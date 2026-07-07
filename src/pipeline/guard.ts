import OpenAI from 'openai';
import { FitLevel, StructuredQuery } from '../types';
import { CONFIG, SENIORITY_RANK } from '../config';
import { withRetry } from '../utils';
import { RerankResult } from './reranking';
import { seniorityTrack } from './retrieval';

export interface GuardResult extends RerankResult {
  guardExplanation: string;
  fit: FitLevel;
  facepalm: boolean;
}

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) _client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return _client;
}

export function isFacepalm(result: RerankResult, query: StructuredQuery): boolean {
  if (!query.seniority) return false;
  const queryLevel = SENIORITY_RANK[query.seniority];
  const profileLevel = SENIORITY_RANK[result.profile.seniority];
  if (queryLevel === undefined || profileLevel === undefined) return false;
  // Cross-track (IC ↔ management) is always a facepalm, same as the ABM seniority
  // filter in retrieval.ts — a candidate can reach the guard via the seniority-relaxed
  // tier with a small rank gap but the wrong track (e.g. query "director" vs. a
  // "principal" IC profile), which numeric distance alone wouldn't catch.
  if (seniorityTrack(query.seniority) !== seniorityTrack(result.profile.seniority)) return true;
  return Math.abs(queryLevel - profileLevel) >= 2;
}

function buildPrompt(result: RerankResult, query: StructuredQuery): string {
  const parts: string[] = ['You are a strict recruiter assistant evaluating candidate fit.\n'];

  parts.push('Role requirements:');
  if (query.title) parts.push(`- Title: ${query.title}`);
  if (query.seniority) parts.push(`- Seniority: ${query.seniority}`);
  if (query.location.city || query.location.region || query.location.country) {
    parts.push(`- Location: ${[query.location.city, query.location.region, query.location.country].filter(Boolean).join(', ')}`);
  }
  if (query.requiredQualifications.length > 0) {
    const ambiguous = new Set(query.ambiguousQualifications);
    const annotated = query.requiredQualifications.map((q) =>
      ambiguous.has(q) ? `${q} (ambiguous abbreviation — verify literally, see rule below)` : q,
    );
    parts.push(`- Required qualifications: ${annotated.join(', ')}`);
  }

  parts.push(`\nCandidate profile:\n${result.embeddingRecord.profileText}`);

  parts.push(
    '\nEvaluate fit using these STRICT rules:' +
    '\n- Only count a skill if it is explicitly listed in the profile. Never infer or assume skills from related work.' +
    '\n- For any qualification marked "(ambiguous abbreviation)": its meaning was not resolved upstream because it is genuinely ambiguous (could stand for more than one thing). Check ONLY whether that exact abbreviation appears verbatim in the profile — never guess or substitute what it might stand for (e.g. do not decide "TS" means "TypeScript"). If it does not appear verbatim, mark it not met, but phrase the explanation as unable to confirm the ambiguous requirement, naming it by its written abbreviation — never state a specific technology you inferred for it.' +
    '\n- If no location requirement is listed above, do not evaluate location — it is not a factor for this role.' +
    '\n- If a location requirement is listed, it is a HARD requirement. If the requirement is a metro area or region (e.g. "San Francisco Bay Area", "Greater New York"), any city commonly understood to be within that region counts as a match (e.g. Oakland, Berkeley, San Jose, Fremont all match "San Francisco Bay Area"). Only flag a location gap when the candidate is clearly outside the region.' +
    '\n- Seniority must be an exact match or within ±1 level. Any seniority gap — including overqualified (e.g. VP for a Director role, Staff for a Senior role) — is a gap. A seniority gap of exactly 1 level caps the fit at "partial"; a gap of 2+ levels means "poor".' +
    '\n\nCriteria:' +
    '\n- "poor": wrong location OR seniority mismatch >= 2 levels OR missing 2+ required qualifications' +
    '\n- "partial": seniority off by exactly 1 level (regardless of other factors) OR missing 1 required qualification AND at least one other gap exists' +
    '\n- "good": seniority ✓, location ✓, missing exactly 1 required qualification and no other gaps' +
    '\n- "excellent": meets ALL requirements explicitly' +
    '\n\nExamples:' +
    '\n- Role: Senior, US, requires [Python, NLP]. Candidate: Senior ✓, SF ✓, has Python ✓, no NLP ✗. → "good"' +
    '\n- Role: Senior, US, requires [Python, NLP]. Candidate: Staff ✗, SF ✓, has Python ✓, no NLP ✗. → "partial" (seniority off by 1)' +
    '\n- Role: Director, requires [distributed systems]. Candidate: VP ✗, has distributed systems ✓. → "partial" (seniority off by 1 caps at partial)' +
    '\n- Role: Senior, US, requires [Python, NLP]. Candidate: Senior ✓, SF ✓, no Python ✗, no NLP ✗. → "poor" (2 missing skills)' +
    '\n- Role: Senior, US, requires [Python, NLP]. Candidate: Senior ✓, SF ✓, has Python ✓, has NLP ✓. → "excellent"' +
    '\n\nRespond with a JSON object with keys "qualificationChecks" (an array with exactly one entry per required qualification listed above, in the same order — each entry an object with "qualification" (the exact string) and "met" (boolean: true only if explicitly evidenced in the candidate profile) — evaluate every required qualification individually, one at a time, before deciding fit), ' +
    '"fit" (one of: poor, partial, good, excellent — must be consistent with how many qualificationChecks entries have met=false, per the criteria above), ' +
    'and "explanation" (1-2 sentences stating exactly which requirements are met and which are missing).'
  );

  return parts.join('\n');
}

async function assessCandidate(
  client: OpenAI,
  result: RerankResult,
  query: StructuredQuery,
): Promise<{ fit: FitLevel; explanation: string }> {
  const response = await withRetry(
    () =>
      client.chat.completions.create({
        model: CONFIG.openai.chat.guardModel,
        max_tokens: 300,
        temperature: CONFIG.openai.chat.classificationTemperature,
        seed: CONFIG.openai.chat.classificationSeed,
        response_format: { type: 'json_object' },
        messages: [{ role: 'user', content: buildPrompt(result, query) }],
      }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  const text = response.choices[0].message.content ?? '{}';

  try {
    const parsed = JSON.parse(text) as {
      fit: FitLevel;
      explanation: string;
      qualificationChecks?: { qualification: string; met: boolean }[];
    };
    const validFits: FitLevel[] = ['poor', 'partial', 'good', 'excellent'];
    if (!validFits.includes(parsed.fit) || typeof parsed.explanation !== 'string') {
      throw new Error(`missing/invalid "fit" or "explanation" field: ${text.slice(0, 300)}`);
    }
    // Deterministic override: count missing qualifications ourselves from the model's
    // per-item checklist rather than trusting a self-reported total. A single self-reported
    // count can itself be wrong (undercounted) with nothing to check it against; a per-item
    // checklist is auditable — we only need the model to judge one qualification at a time
    // correctly, and code does the counting and the fit-consistency enforcement.
    const missingRequiredCount = Array.isArray(parsed.qualificationChecks)
      ? parsed.qualificationChecks.filter((c) => c.met === false).length
      : 0;
    const fit = missingRequiredCount >= 2 ? 'poor' : parsed.fit;
    return { fit, explanation: parsed.explanation };
  } catch (err) {
    // A broken/unparseable guard response is a real failure, not a legitimate "partial" fit —
    // log it so it's auditable, and default to excluding the candidate (poor) rather than
    // silently surfacing raw model output as if it were a genuine assessment.
    console.error(
      `[guard] Failed to parse assessment for profile ${result.profile.id}: ${err instanceof Error ? err.message : err}`,
    );
    return {
      fit: 'poor',
      explanation: 'Guard assessment failed (unparseable model output) — treated as poor fit pending review.',
    };
  }
}

export async function runWithConcurrency<T, R>(
  items: T[],
  fn: (item: T) => Promise<R>,
  concurrency: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let idx = 0;
  async function worker() {
    while (idx < items.length) {
      const i = idx++;
      results[i] = await fn(items[i]);
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return results;
}

export async function runGuard(
  query: StructuredQuery,
  candidates: RerankResult[],
): Promise<GuardResult[]> {
  const client = getClient();
  const assessments = await runWithConcurrency(
    candidates,
    (c) => assessCandidate(client, c, query),
    CONFIG.pipeline.guardMaxConcurrency,
  );

  return candidates.map((c, i) => ({
    ...c,
    guardExplanation: assessments[i].explanation,
    fit: assessments[i].fit,
    facepalm: isFacepalm(c, query),
  }));
}
