import fs from 'fs';
import { CandidateProfile } from '../src/types';
import { CONFIG } from '../src/config';
import { profileToText } from '../src/embeddings/profileText';
import { loadCache, saveCache, embedAllProfiles } from '../src/embeddings/cache';
import { buildIndex, saveIndex } from '../src/index/faissManager';

async function main() {
  console.log('MUSE-PoC: Building FAISS index...\n');

  if (!fs.existsSync(CONFIG.data.profilesPath)) {
    console.error(`Profiles not found at ${CONFIG.data.profilesPath}. Run "npm run generate" first.`);
    process.exit(1);
  }

  const profiles = JSON.parse(
    fs.readFileSync(CONFIG.data.profilesPath, 'utf-8'),
  ) as CandidateProfile[];
  console.log(`Loaded ${profiles.length} profiles.`);

  // Build profile texts
  const profileTexts = profiles.map((p) => ({ id: p.id, text: profileToText(p) }));

  // Load embedding cache and embed any missing profiles
  const cache = loadCache();
  await embedAllProfiles(profileTexts, cache);
  saveCache(cache);
  console.log(`Embeddings cached → ${CONFIG.data.embeddingsCachePath}`);

  // Build FAISS index over 512-dim short vectors
  const records = profiles.map((p) => {
    const record = cache.get(p.id);
    if (!record) throw new Error(`Missing embedding for profile ${p.id}`);
    return record;
  });

  const faissIndex = buildIndex(records);
  saveIndex(faissIndex, CONFIG.data.indexPath, CONFIG.data.profileIdMapPath);

  console.log(`\n✓ Index ready: ${faissIndex.profileIdMap.length} vectors at ${CONFIG.data.indexPath}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
