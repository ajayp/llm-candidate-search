import { CandidateProfile, EmbeddingRecord, StructuredQuery } from '../types';
import { CONFIG, SENIORITY_RANK } from '../config';
import { embedTexts } from '../embeddings/client';
import { FaissIndex, searchIndex } from '../index/faissManager';
import { generateHyDE } from './hyde';


export interface RetrievalResult {
  profile: CandidateProfile;
  l1Score: number;
  embeddingRecord: EmbeddingRecord;
}

export interface RetrievalOutput {
  results: RetrievalResult[];
  fullQueryVector: number[];
}

export function matchesLocation(profile: CandidateProfile, query: StructuredQuery): boolean {
  if (!query.locationStrict) return true;
  if (!query.location.country) return true;
  if (profile.location.country !== query.location.country) return false;
  // Enforce city only for specific-city queries (no region set).
  // Metro area queries (region set, city null) let all country-matched candidates through
  // and rely on the LLM Guard for geographic nuance.
  if (query.location.city && !query.location.region) {
    return profile.location.city === query.location.city;
  }
  return true;
}

const IC_TRACK = new Set(['intern', 'junior', 'mid', 'senior', 'staff', 'principal']);

// Shared with guard.ts's facepalm check so the two stages never disagree about
// what counts as a cross-track (IC ↔ management) seniority mismatch.
export function seniorityTrack(level: string | null | undefined): 'ic' | 'mgmt' {
  return IC_TRACK.has(level ?? '') ? 'ic' : 'mgmt';
}

export function matchesSeniority(profile: CandidateProfile, query: StructuredQuery): boolean {
  if (!query.seniority) return true;
  const queryLevel = SENIORITY_RANK[query.seniority];
  const profileLevel = SENIORITY_RANK[profile.seniority ?? ''];
  if (queryLevel === undefined || profileLevel === undefined) return true;
  // Cross-track matches (IC ↔ management) are never within ±1 regardless of rank proximity
  if (seniorityTrack(query.seniority) !== seniorityTrack(profile.seniority)) return false;
  return Math.abs(queryLevel - profileLevel) <= 1;
}

export function buildScoredTiers(
  hydrated: RetrievalResult[],
  query: StructuredQuery,
  minScore: number,
  abmMinSurvivors: number,
): RetrievalResult[] {
  // Tiers are intentionally NOT re-sorted by score across tier boundaries: a full
  // attribute match (tier1) always outranks a relaxed one, even at a lower L1 score —
  // attribute matching is a stronger relevance signal here than raw cosine similarity.
  // `hydrated` arrives sorted descending by l1Score (FAISS result order), and each
  // tier's filter preserves that order, so within a tier the ranking is still correct.
  const tier1 = hydrated.filter(
    (r) => matchesLocation(r.profile, query) && matchesSeniority(r.profile, query) && r.l1Score >= minScore,
  );

  let scored = tier1;

  if (scored.length < abmMinSurvivors) {
    const relaxed = hydrated.filter((r) => matchesLocation(r.profile, query) && r.l1Score >= minScore);
    const seen = new Set(tier1.map((r) => r.profile.id));
    scored = [...tier1, ...relaxed.filter((r) => !seen.has(r.profile.id))];
  }

  if (scored.length < abmMinSurvivors) {
    const relaxed = hydrated.filter((r) => r.l1Score >= minScore);
    const seen = new Set(scored.map((r) => r.profile.id));
    scored = [...scored, ...relaxed.filter((r) => !seen.has(r.profile.id))];
  }

  return scored;
}

export async function retrieve(
  query: StructuredQuery,
  faissIndex: FaissIndex,
  profileMap: Map<string, CandidateProfile>,
  embeddingCache: Map<string, EmbeddingRecord>,
  hydeText?: string,
  precomputedFullVector?: number[],
): Promise<RetrievalOutput> {
  // Embed HyDE synthetic profile at 512-dim for L1 (aligns query vector with profile-shaped embeddings).
  // Callers that already embedded this text (e.g. eval comparing multiple ranking modes on the
  // same query) can pass precomputedFullVector to skip a redundant embedding API call.
  let fullQueryVector: number[];
  if (precomputedFullVector) {
    fullQueryVector = precomputedFullVector;
  } else {
    const textToEmbed = hydeText ?? await generateHyDE(query);
    [fullQueryVector] = await embedTexts([textToEmbed]);
  }
  const shortQueryVector = fullQueryVector.slice(0, CONFIG.openai.embeddings.shortDims);

  // L1: FAISS search
  const l1Results = searchIndex(faissIndex, shortQueryVector, CONFIG.pipeline.l1TopK);

  // Hydrate profiles
  const hydrated = l1Results
    .map((r) => {
      const profile = profileMap.get(r.profileId);
      const embeddingRecord = embeddingCache.get(r.profileId);
      if (!profile || !embeddingRecord) return null;
      return { profile, l1Score: r.score, embeddingRecord };
    })
    .filter((r): r is RetrievalResult => r !== null);

  if (hydrated.length < l1Results.length) {
    console.warn(
      `[retrieval] Dropped ${l1Results.length - hydrated.length} FAISS hit(s) with no matching profile/embedding-cache entry — index and data may be out of sync.`,
    );
  }

  const scored = buildScoredTiers(hydrated, query, CONFIG.pipeline.l1MinScore, CONFIG.pipeline.abmMinSurvivors);

  return {
    results: scored.slice(0, CONFIG.pipeline.l2TopN),
    fullQueryVector,
  };
}
