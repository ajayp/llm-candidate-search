import fs from 'fs';
import path from 'path';
import { EmbeddingRecord } from '../types';
import { CONFIG } from '../config';
import { embedTexts } from './client';

export function loadCache(): Map<string, EmbeddingRecord> {
  const cachePath = CONFIG.data.embeddingsCachePath;
  if (!fs.existsSync(cachePath)) {
    return new Map();
  }
  const raw = JSON.parse(fs.readFileSync(cachePath, 'utf-8')) as EmbeddingRecord[];
  return new Map(raw.map((r) => [r.profileId, r]));
}

export function saveCache(cache: Map<string, EmbeddingRecord>): void {
  const cachePath = CONFIG.data.embeddingsCachePath;
  const dir = path.dirname(cachePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(Array.from(cache.values()), null, 2));
}

export async function getOrEmbed(
  profileId: string,
  profileText: string,
  cache: Map<string, EmbeddingRecord>,
): Promise<EmbeddingRecord> {
  const cached = cache.get(profileId);
  if (cached) return cached;

  const [fullVector] = await embedTexts([profileText]);
  const shortVector = fullVector.slice(0, CONFIG.openai.embeddings.shortDims);

  const record: EmbeddingRecord = {
    profileId,
    fullVector,
    shortVector,
    profileText,
    embeddedAt: new Date().toISOString(),
  };
  cache.set(profileId, record);
  return record;
}

export async function embedAllProfiles(
  profiles: Array<{ id: string; text: string }>,
  cache: Map<string, EmbeddingRecord>,
): Promise<void> {
  const missing = profiles.filter((p) => !cache.has(p.id));
  if (missing.length === 0) {
    console.log(`All ${profiles.length} profiles already cached.`);
    return;
  }

  console.log(`Embedding ${missing.length} new profiles (${profiles.length - missing.length} cached)...`);
  const batchSize = CONFIG.openai.embeddings.batchSize;

  for (let i = 0; i < missing.length; i += batchSize) {
    const batch = missing.slice(i, i + batchSize);
    const vectors = await embedTexts(batch.map((p) => p.text));

    for (let j = 0; j < batch.length; j++) {
      const { id, text } = batch[j];
      const fullVector = vectors[j];
      const shortVector = fullVector.slice(0, CONFIG.openai.embeddings.shortDims);
      cache.set(id, {
        profileId: id,
        fullVector,
        shortVector,
        profileText: text,
        embeddedAt: new Date().toISOString(),
      });
    }

    saveCache(cache);
    console.log(`  Saved ${Math.min(i + batchSize, missing.length)}/${missing.length} embeddings`);
  }
}
