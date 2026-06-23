import { CandidateProfile, EmbeddingRecord, SearchResult, StructuredQuery } from '../types';
import { CONFIG } from '../config';
import { loadCache } from '../embeddings/cache';
import { loadIndex } from '../index/faissManager';
import { understandQuery } from './queryUnderstanding';
import { retrieve } from './retrieval';
import { rerank, rerankCosine } from './reranking';
import { runGuard } from './guard';
import fs from 'fs';

export interface PipelineContext {
  profileMap: Map<string, CandidateProfile>;
  embeddingCache: Map<string, EmbeddingRecord>;
  faissIndex: ReturnType<typeof loadIndex>;
}

export interface ExtendedSearchResult {
  results: SearchResult[];
  // L1 candidate IDs before re-ranking — used to compute Recall@50 in eval
  l1CandidateIds: string[];
}

export function loadPipelineContext(): PipelineContext {
  const profiles = JSON.parse(
    fs.readFileSync(CONFIG.data.profilesPath, 'utf-8'),
  ) as CandidateProfile[];

  const profileMap = new Map(profiles.map((p) => [p.id, p]));
  const embeddingCache = loadCache();
  const faissIndex = loadIndex(CONFIG.data.indexPath, CONFIG.data.profileIdMapPath);

  return { profileMap, embeddingCache, faissIndex };
}

export type RankingMode = 'l1' | 'cosine' | 'cohere';

export interface SearchOptions {
  skipGuard?: boolean;
  rankingMode?: RankingMode;
  structuredQuery?: StructuredQuery;
  hydeText?: string;
}

export async function search(
  rawQuery: string,
  ctx: PipelineContext,
  options: SearchOptions = {},
): Promise<ExtendedSearchResult> {
  // Stage 1: Query understanding — skip if caller already parsed the query
  let structuredQuery: StructuredQuery;
  if (options.structuredQuery) {
    structuredQuery = options.structuredQuery;
    console.log('[Stage 1] Query understanding skipped (pre-parsed).');
  } else {
    console.log('[Stage 1] Query understanding...');
    structuredQuery = await understandQuery(rawQuery);
    console.log('[Stage 1] Done:', JSON.stringify(structuredQuery, null, 2));
  }

  // Stage 2 + ABM: L1 EBR retrieval with attribute-based post-filter
  console.log('[Stage 2] L1 retrieval + ABM...');
  const { results: l1Results, fullQueryVector } = await retrieve(
    structuredQuery,
    ctx.faissIndex,
    ctx.profileMap,
    ctx.embeddingCache,
    options.hydeText,
  );
  console.log(`[Stage 2] Done: ${l1Results.length} candidates after ABM filter`);

  const l1CandidateIds = l1Results.map((r) => r.profile.id);

  if (l1Results.length === 0) {
    return { results: [], l1CandidateIds: [] };
  }

  // Stage 3: L2 re-ranking — mode determines strategy
  const rankingMode = options.rankingMode ?? 'cohere';
  console.log(`[Stage 3] L2 reranking (mode: ${rankingMode})...`);
  const l2Results =
    rankingMode === 'l1'
      ? l1Results.slice(0, CONFIG.pipeline.finalTopK).map((r) => ({ ...r, l2Score: r.l1Score }))
      : rankingMode === 'cosine'
        ? rerankCosine(fullQueryVector, l1Results)
        : await rerank(structuredQuery, l1Results);
  console.log(`[Stage 3] Done: ${l2Results.length} candidates after rerank`);

  // L2 floor — drop weak rerank scores before guard to avoid wasting tokens
  const l2Qualified = l2Results.filter((r) => r.l2Score >= CONFIG.pipeline.l2MinScore);

  // Stage 4: LLM guard — structured fit assessment + filter poor fits
  let results: SearchResult[];
  if (options.skipGuard) {
    console.log('[Stage 4] LLM guard skipped.');
    results = l2Qualified.map((r) => ({
      profile: r.profile,
      l1Score: r.l1Score,
      l2Score: r.l2Score,
      guardExplanation: '',
      fit: 'partial' as const,
      facepalm: false,
    }));
  } else {
    console.log('[Stage 4] LLM guard...');
    const guarded = await runGuard(structuredQuery, l2Qualified);
    const filtered = guarded.filter((r) => r.fit !== 'poor');
    console.log(`[Stage 4] Done: ${filtered.length} candidates after guard filter (${guarded.length - filtered.length} poor fits removed)`);
    results = filtered.map((r) => ({
      profile: r.profile,
      l1Score: r.l1Score,
      l2Score: r.l2Score,
      guardExplanation: r.guardExplanation,
      fit: r.l2Score < CONFIG.pipeline.l2MinScore ? 'poor' : r.fit,
      facepalm: r.facepalm,
    }));
  }

  return { results, l1CandidateIds };
}
