import * as dotenv from 'dotenv'; dotenv.config({ override: true });
import fs from 'fs';
import OpenAI from 'openai';
import { CandidateProfile, EvalQuery, EvalQueryRaw } from '../src/types';
import { CONFIG } from '../src/config';
import { profileToText } from '../src/embeddings/profileText';
import { withRetry } from '../src/utils';
import { loadPipelineContext, search } from '../src/pipeline/pipeline';

const PROFILE_BATCH_SIZE = 10;
const MAX_CONCURRENT = 4;

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function buildPrompt(query: string, profiles: CandidateProfile[]): string {
  const profileTexts = profiles.map((p, i) => `[${i}] ${p.name} (${p.id}):\n${profileToText(p)}`);
  return (
    `You are a recruiter assistant creating ground-truth relevance labels. For each candidate below, decide if they are QUALIFIED or NOT_QUALIFIED for this role.\n\n` +
    `Role query: "${query}"\n\n` +
    `Labeling rules:\n` +
    `- For location: when the query names a metro area or region (e.g. "Bay Area", "Greater New York"), any city within that region counts as a match (e.g. Oakland, Berkeley, San Jose all match "Bay Area"). Only mark location as a disqualifier when the candidate is clearly outside the region.\n` +
    `- For soft location language ("preferably", "ideally", "open to remote"): do not disqualify candidates solely on location.\n` +
    `- For skills: only disqualify if a clearly required skill is entirely absent. Use judgment — a candidate strong in adjacent skills may still be qualified.\n\n` +
    `Candidates:\n${profileTexts.join('\n\n---\n\n')}\n\n` +
    `Respond with a JSON object mapping candidate index to "QUALIFIED" or "NOT_QUALIFIED". Example: {"0":"QUALIFIED","1":"NOT_QUALIFIED"}`
  );
}

async function scoreProfileBatch(
  query: string,
  profiles: CandidateProfile[],
): Promise<Map<string, boolean>> {
  const response = await withRetry(
    () => client.chat.completions.create({
      model: CONFIG.openai.chat.guardModel,
      max_tokens: 512,
      response_format: { type: 'json_object' },
      messages: [{ role: 'user', content: buildPrompt(query, profiles) }],
    }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  const text = response.choices[0].message.content ?? '{}';
  let parsed: Record<string, string> = {};
  const match = text.match(/\{[\s\S]*\}/);
  if (match) {
    try { parsed = JSON.parse(match[0]) as Record<string, string>; } catch { /* skip */ }
  }

  const result = new Map<string, boolean>();
  for (const [idx, verdict] of Object.entries(parsed)) {
    const profile = profiles[parseInt(idx)];
    if (profile) result.set(profile.id, verdict === 'QUALIFIED');
  }
  return result;
}

async function withConcurrency<T>(tasks: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let next = 0;
  async function worker() {
    while (next < tasks.length) {
      const i = next++;
      results[i] = await tasks[i]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

async function labelQuery(
  rawQuery: EvalQueryRaw,
  candidates: CandidateProfile[],
): Promise<EvalQuery> {
  const qualifiedIds = new Set<string>();

  const tasks: (() => Promise<void>)[] = [];
  for (let i = 0; i < candidates.length; i += PROFILE_BATCH_SIZE) {
    const batch = candidates.slice(i, i + PROFILE_BATCH_SIZE);
    tasks.push(async () => {
      const scores = await scoreProfileBatch(rawQuery.raw, batch);
      for (const [id, qualified] of scores) {
        if (qualified) qualifiedIds.add(id);
      }
    });
  }
  await withConcurrency(tasks, MAX_CONCURRENT);

  return { id: rawQuery.id, raw: rawQuery.raw, relevantProfileIds: Array.from(qualifiedIds) };
}

async function main() {
  const limitArg = process.argv.find((a) => a.startsWith('--limit='));
  const limit = limitArg ? parseInt(limitArg.split('=')[1], 10) : 5;

  console.log(`MUSE-PoC: Generating eval labels for ${limit} queries (L1 candidates only)...\n`);

  if (!fs.existsSync(CONFIG.data.evalQueriesRawPath)) {
    console.error(`Raw eval queries not found at ${CONFIG.data.evalQueriesRawPath}`);
    process.exit(1);
  }

  const rawQueries = (JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesRawPath, 'utf-8')) as EvalQueryRaw[]).slice(0, limit);

  const existing: EvalQuery[] = fs.existsSync(CONFIG.data.evalQueriesPath)
    ? (() => {
        const raw = JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesPath, 'utf-8')) as EvalQuery[] | EvalQuery[][];
        return (Array.isArray(raw[0]) ? raw[0] : raw) as EvalQuery[];
      })()
    : [];
  const completedIds = new Set(existing.map((q) => q.id));
  const remaining = rawQueries.filter((q) => !completedIds.has(q.id));

  if (remaining.length === 0) {
    console.log('All queries already labeled.');
    process.exit(0);
  }

  console.log('Loading pipeline context...');
  const ctx = loadPipelineContext();

  const allResults = [...existing];

  for (let i = 0; i < remaining.length; i++) {
    const rawQuery = remaining[i];
    console.log(`\n[${i + 1}/${remaining.length}] "${rawQuery.raw.slice(0, 70)}"`);

    // Run pipeline to get the L1 candidate pool for this query
    process.stdout.write('  Running L1 retrieval...\r');
    const { l1CandidateIds } = await search(rawQuery.raw, ctx, { skipGuard: true, rankingMode: 'l1' });

    // Hydrate candidate IDs into full profiles
    const candidates = l1CandidateIds
      .map((id) => ctx.profileMap.get(id))
      .filter((p): p is CandidateProfile => p !== undefined);

    console.log(`  L1 pool: ${candidates.length} candidates — labeling...`);

    // Label only the L1 candidates
    const result = await labelQuery(rawQuery, candidates);

    // Write to disk immediately
    allResults.push(result);
    fs.writeFileSync(CONFIG.data.evalQueriesPath, JSON.stringify(allResults, null, 2));

    console.log(`  ✓ ${result.relevantProfileIds.length} relevant out of ${candidates.length} candidates`);
  }

  const avgRelevant = allResults.reduce((s, q) => s + q.relevantProfileIds.length, 0) / allResults.length;
  console.log(`\n✓ Done → ${CONFIG.data.evalQueriesPath}`);
  console.log(`Average relevant per query: ${avgRelevant.toFixed(1)}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
