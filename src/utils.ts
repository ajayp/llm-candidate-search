// OpenAI/Cohere SDK errors carry an HTTP `status`. 429 (rate limit) and 5xx (server-side)
// are transient — worth retrying. Other 4xx (bad request, auth, schema violation) are
// permanent — retrying just burns ~30s of exponential backoff before failing anyway.
// Errors with no status (network failures, timeouts) are treated as retryable.
function isRetryableError(err: unknown): boolean {
  const status = (err as { status?: number } | null)?.status;
  if (status === undefined) return true;
  return status === 429 || status >= 500;
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts = 5,
  baseDelayMs = 1000,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt === maxAttempts || !isRetryableError(err)) break;
      const delay = baseDelayMs * Math.pow(2, attempt - 1);
      console.warn(`Attempt ${attempt} failed (${err instanceof Error ? err.message : err}), retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }
  throw lastError;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function normalizeVector(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
  if (norm === 0) return v;
  return v.map((x) => x / norm);
}

export function flattenVectors(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const d = vectors[0].length;
  const flat = new Array<number>(vectors.length * d);
  for (let i = 0; i < vectors.length; i++) {
    for (let j = 0; j < d; j++) {
      flat[i * d + j] = vectors[i][j];
    }
  }
  return flat;
}

export function computeNDCG(ranked: string[], relevant: Set<string>, k: number): number {
  const dcg = ranked.slice(0, k).reduce((sum, id, i) => {
    const rel = relevant.has(id) ? 1 : 0;
    return sum + rel / Math.log2(i + 2);
  }, 0);

  const idealCount = Math.min(relevant.size, k);
  const idcg = Array.from({ length: idealCount }, (_, i) => 1 / Math.log2(i + 2)).reduce(
    (a, b) => a + b,
    0,
  );

  return idcg === 0 ? 0 : dcg / idcg;
}

export function computeRecall(ranked: string[], relevant: Set<string>, k: number): number {
  if (relevant.size === 0) return 0;
  const hits = ranked.slice(0, k).filter((id) => relevant.has(id)).length;
  return hits / relevant.size;
}
