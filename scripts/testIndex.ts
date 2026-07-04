import fs from 'fs';
import { CandidateProfile } from '../src/types';
import { CONFIG } from '../src/config';
import { profileToText } from '../src/embeddings/profileText';
import { loadCache, saveCache, embedAllProfiles } from '../src/embeddings/cache';
import { buildIndex, saveIndex, searchIndex } from '../src/index/faissManager';
import { embedTexts } from '../src/embeddings/client';

async function main() {
  const profiles = JSON.parse(
    fs.readFileSync(CONFIG.data.profilesPath, 'utf-8'),
  ) as CandidateProfile[];

  const sample = profiles.slice(0, 2);
  console.log('Testing with 2 profiles:');
  sample.forEach((p) => console.log(` - ${p.id} | ${p.title} | ${p.seniority} | ${p.location.city}`));

  // Embed the 2 profiles
  const profileTexts = sample.map((p) => ({ id: p.id, text: profileToText(p) }));
  const cache = loadCache();
  await embedAllProfiles(profileTexts, cache);
  saveCache(cache);

  const records = sample.map((p) => {
    const r = cache.get(p.id);
    if (!r) throw new Error(`Missing embedding for ${p.id}`);
    return r;
  });

  console.log(`\nEmbeddings OK — fullVector dims: ${records[0].fullVector.length}, shortVector dims: ${records[0].shortVector.length}`);

  // Build a tiny index over just the 2 profiles
  const faissIndex = buildIndex(records);
  console.log(`Index built: ${faissIndex.profileIdMap.length} vectors`);

  // Query: embed the first profile's text and search — it should come back as top-1
  const [queryVec] = await embedTexts([profileTexts[0].text]);
  const shortQuery = queryVec.slice(0, CONFIG.openai.embeddings.shortDims);
  const results = searchIndex(faissIndex, shortQuery, 2);

  console.log('\nSearch results (querying with profile[0] text):');
  results.forEach((r, i) => console.log(` ${i + 1}. profileId=${r.profileId}  score=${r.score.toFixed(4)}`));

  const topId = results[0]?.profileId;
  if (topId === sample[0].id) {
    console.log('\n✓ Top result matches the query profile — index working correctly.');
  } else {
    console.error('\n✗ Top result does NOT match — something is wrong.');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
