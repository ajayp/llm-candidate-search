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
  if (query.location.city && CONFIG.pipeline.strictCityFilter) {
    return profile.location.city === query.location.city;
  }
  return true;
}

const IC_TRACK = new Set(['intern', 'junior', 'mid', 'senior', 'staff', 'principal']);

export function matchesSeniority(profile: CandidateProfile, query: StructuredQuery): boolean {
  if (!query.seniority) return true;
  const queryLevel = SENIORITY_RANK[query.seniority];
  const profileLevel = SENIORITY_RANK[profile.seniority ?? ''];
  if (queryLevel === undefined || profileLevel === undefined) return true;
  // Cross-track matches (IC ↔ management) are never within ±1 regardless of rank proximity
  const queryTrack = IC_TRACK.has(query.seniority) ? 'ic' : 'mgmt';
  const profileTrack = IC_TRACK.has(profile.seniority ?? '') ? 'ic' : 'mgmt';
  if (queryTrack !== profileTrack) return false;
  return Math.abs(queryLevel - profileLevel) <= 1;
}

export function buildScoredTiers(
  hydrated: RetrievalResult[],
  query: StructuredQuery,
  minScore: number,
  abmMinSurvivors: number,
): RetrievalResult[] {
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
): Promise<RetrievalOutput> {
  // Embed HyDE synthetic profile at 512-dim for L1 (aligns query vector with profile-shaped embeddings)
  const textToEmbed = hydeText ?? await generateHyDE(query);
  const [fullQueryVector] = await embedTexts([textToEmbed]);
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

  const scored = buildScoredTiers(hydrated, query, CONFIG.pipeline.l1MinScore, CONFIG.pipeline.abmMinSurvivors);

  return {
    results: scored.slice(0, CONFIG.pipeline.l2TopN),
    fullQueryVector,
  };
}
