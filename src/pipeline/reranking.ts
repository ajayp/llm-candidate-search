import { CandidateProfile, CohereRerankResponse, EmbeddingRecord, StructuredQuery } from '../types';
import { CONFIG } from '../config';
import { withRetry } from '../utils';
import { RetrievalResult } from './retrieval';

export interface RerankResult {
  profile: CandidateProfile;
  l1Score: number;
  l2Score: number;
  embeddingRecord: EmbeddingRecord;
}

export async function rerank(
  query: StructuredQuery,
  candidates: RetrievalResult[],
): Promise<RerankResult[]> {
  if (candidates.length === 0) return [];

  const documents = candidates.map((c) => c.embeddingRecord.profileText);

  const body = {
    model: CONFIG.cohere.rerankingModel,
    query: query.queryText,
    top_n: CONFIG.pipeline.finalTopK,
    return_documents: false,
    documents,
  };

  const response = await withRetry(
    async () => {
      const res = await fetch(CONFIG.cohere.rerankEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.COHERE_API_KEY}`,
        },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Cohere rerank failed (${res.status}): ${text}`);
      }

      return res.json() as Promise<CohereRerankResponse>;
    },
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  return response.results
    .sort((a, b) => b.relevance_score - a.relevance_score)
    .map((r) => ({
      profile: candidates[r.index].profile,
      l1Score: candidates[r.index].l1Score,
      l2Score: r.relevance_score,
      embeddingRecord: candidates[r.index].embeddingRecord,
    }));
}

// 3072-dim cosine reranking — bi-encoder baseline (not equivalent to cross-encoder DCNv2).
// Used in eval to show the Matryoshka dimension effect: cosine_3072 > cosine_512.
export function rerankCosine(fullQueryVector: number[], candidates: RetrievalResult[]): RerankResult[] {
  const norm = (v: number[]) => { const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)); return n === 0 ? v : v.map((x) => x / n); };
  const qNorm = norm(fullQueryVector);
  return candidates
    .map((c) => {
      const dNorm = norm(c.embeddingRecord.fullVector);
      const score = qNorm.reduce((s, v, i) => s + v * dNorm[i], 0);
      return { profile: c.profile, l1Score: c.l1Score, l2Score: score, embeddingRecord: c.embeddingRecord };
    })
    .sort((a, b) => b.l2Score - a.l2Score)
    .slice(0, CONFIG.pipeline.finalTopK);
}
