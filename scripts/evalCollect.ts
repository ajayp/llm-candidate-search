/**
 * Parse a locally downloaded OpenAI batch output file and merge results into eval_queries.json.
 * Usage: ts-node scripts/evalCollect.ts <output_file.jsonl>
 */
import fs from 'fs';
import { CandidateProfile, EvalQuery, EvalQueryRaw } from '../src/types';
import { CONFIG } from '../src/config';

const PROFILE_BATCH_SIZE = 10;
const STATE_PATH = 'data/eval_batch_state.json';

interface GroupState {
  queryIds: string[];
  groupFile: string;
  batchId: string | null;
  outputFileId: string | null;
  status: 'pending' | 'submitted' | 'completed';
}

interface State {
  groups: GroupState[];
}

function parseOutputFile(
  outputJsonl: string,
  groupQueryIds: string[],
  profiles: CandidateProfile[],
  rawQueries: EvalQueryRaw[],
): EvalQuery[] {
  const queryRawText = new Map(rawQueries.map((q) => [q.id, q.raw]));

  const profileBatches = new Map<string, CandidateProfile[]>();
  for (const queryId of groupQueryIds) {
    const numBatches = Math.ceil(profiles.length / PROFILE_BATCH_SIZE);
    for (let b = 0; b < numBatches; b++) {
      profileBatches.set(
        `${queryId}__b${b}`,
        profiles.slice(b * PROFILE_BATCH_SIZE, (b + 1) * PROFILE_BATCH_SIZE),
      );
    }
  }

  const qualifiedByQuery = new Map<string, Set<string>>();
  for (const queryId of groupQueryIds) qualifiedByQuery.set(queryId, new Set());

  for (const line of outputJsonl.split('\n').filter(Boolean)) {
    const result = JSON.parse(line) as {
      custom_id: string;
      response: { body: { choices: Array<{ message: { content: string } }> } };
    };

    const customId = result.custom_id;
    const queryId = customId.split('__b')[0];
    const batchProfiles = profileBatches.get(customId);
    if (!batchProfiles || !qualifiedByQuery.has(queryId)) continue;

    const content = result.response?.body?.choices?.[0]?.message?.content ?? '{}';
    let parsed: Record<string, string> = {};
    const match = content.match(/\{[\s\S]*\}/);
    if (match) {
      try { parsed = JSON.parse(match[0]) as Record<string, string>; } catch { /* skip */ }
    }

    for (const [idx, verdict] of Object.entries(parsed)) {
      const profile = batchProfiles[parseInt(idx)];
      if (profile && verdict === 'QUALIFIED') {
        qualifiedByQuery.get(queryId)!.add(profile.id);
      }
    }
  }

  return groupQueryIds.map((queryId) => ({
    id: queryId,
    raw: queryRawText.get(queryId)!,
    relevantProfileIds: Array.from(qualifiedByQuery.get(queryId) ?? []),
  }));
}

function main() {
  const outputFile = process.argv[2];
  if (!outputFile) {
    console.error('Usage: ts-node scripts/evalCollect.ts <output_file.jsonl>');
    process.exit(1);
  }

  if (!fs.existsSync(outputFile)) {
    console.error(`File not found: ${outputFile}`);
    process.exit(1);
  }

  if (!fs.existsSync(STATE_PATH)) {
    console.error(`State file not found at ${STATE_PATH}`);
    process.exit(1);
  }

  const rawQueries = JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesRawPath, 'utf-8')) as EvalQueryRaw[];
  const profiles = JSON.parse(fs.readFileSync(CONFIG.data.profilesPath, 'utf-8')) as CandidateProfile[];
  const state = JSON.parse(fs.readFileSync(STATE_PATH, 'utf-8')) as State;

  // Find the first group not yet completed
  const group = state.groups.find((g) => g.status !== 'completed');
  if (!group) {
    console.error('No submitted group found in state file.');
    process.exit(1);
  }

  console.log(`Parsing output for group: ${group.queryIds.join(', ')}`);

  const outputJsonl = fs.readFileSync(outputFile, 'utf-8');
  const groupResults = parseOutputFile(outputJsonl, group.queryIds, profiles, rawQueries);

  // Load existing eval queries and merge
  const existing: EvalQuery[] = fs.existsSync(CONFIG.data.evalQueriesPath)
    ? (() => {
        const raw = JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesPath, 'utf-8')) as EvalQuery[] | EvalQuery[][];
        return (Array.isArray(raw[0]) ? raw[0] : raw) as EvalQuery[];
      })()
    : [];

  const merged = [...existing, ...groupResults];
  fs.writeFileSync(CONFIG.data.evalQueriesPath, JSON.stringify(merged, null, 2));

  // Mark group as completed in state
  group.status = 'completed';
  group.outputFileId = outputFile;
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));

  const avgQualified = groupResults.reduce((s, q) => s + q.relevantProfileIds.length, 0) / groupResults.length;
  console.log(`✓ Merged ${groupResults.length} queries into ${CONFIG.data.evalQueriesPath}`);
  console.log(`  Avg relevant per query: ${avgQualified.toFixed(1)}`);
  console.log(`\nRun "npm run eval:generate" to continue with remaining groups.`);
}

main();
