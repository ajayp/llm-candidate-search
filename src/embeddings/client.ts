import OpenAI from 'openai';
import { CONFIG } from '../config';
import { withRetry } from '../utils';

let _client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!_client) {
    _client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return _client;
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  const client = getClient();
  const response = await withRetry(
    () =>
      client.embeddings.create({
        model: CONFIG.openai.embeddings.model,
        input: texts,
        dimensions: CONFIG.openai.embeddings.fullDims,
      }),
    CONFIG.retry.maxAttempts,
    CONFIG.retry.baseDelayMs,
  );

  if (response.data.length !== texts.length) {
    throw new Error(`OpenAI returned ${response.data.length} embeddings for ${texts.length} inputs`);
  }

  return response.data
    .sort((a, b) => a.index - b.index)
    .map((item) => item.embedding);
}
