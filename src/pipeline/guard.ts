import OpenAI from 'openai';
import { FitLevel, StructuredQuery } from '../types';
import { CONFIG, SENIORITY_RANK } from '../config';
import { withRetry } from '../utils';
import { RerankResult } from './reranking';

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

function isFacepalm(result: RerankResult, query: StructuredQuery): boolean {
  if (!query.seniority) return false;
  const queryLevel = SENIORITY_RANK[query.seniority];
  const profileLevel = SENIORITY_RANK[result.profile.seniority];
  if (queryLevel !== undefined && profileLevel !== undefined) {
    if (Math.abs(queryLevel - profileLevel) >= 2) return true;
  }
  return false;
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
    parts.push(`- Required qualifications: ${query.requiredQualifications.join(', ')}`);
  }

  parts.push(`\nCandidate profile:\n${result.embeddingRecord.profileText}`);

  parts.push(
    '\nEvaluate fit using these STRICT rules:' +
    '\n- Only count a skill if it is explicitly listed in the profile. Never infer or assume skills from related work.' +
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
    '\n\nRespond with a JSON object with keys "fit" (one of: poor, partial, good, excellent) and "explanation" (1-2 sentences stating exactly which requirements are met and which are missing).'
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
        response_format: { type: 'json_object' },
        messages: [{ role: 'user', content: buildPrompt(result, query) }],
      }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  const text = response.choices[0].message.content ?? '{}';

  try {
    const parsed = JSON.parse(text) as { fit: FitLevel; explanation: string };
    const validFits: FitLevel[] = ['poor', 'partial', 'good', 'excellent'];
    if (validFits.includes(parsed.fit) && typeof parsed.explanation === 'string') {
      return parsed;
    }
  } catch {
    // fall through to fallback
  }

  return { fit: 'partial', explanation: text };
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
