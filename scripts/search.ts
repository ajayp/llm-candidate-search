import * as dotenv from 'dotenv'; dotenv.config({ override: true });
import { loadPipelineContext, search } from '../src/pipeline/pipeline';

async function main() {
  const rawQuery = process.argv.slice(2).join(' ').trim();
  if (!rawQuery) {
    console.error('Usage: npm run search "<query>"');
    console.error('Example: npm run search "Senior ML engineer, Python, NLP background, Bay Area"');
    process.exit(1);
  }

  console.log(`\nQuery: "${rawQuery}"\n`);
  console.log('Loading index...');
  const ctx = loadPipelineContext();

  console.log('Running pipeline...\n');
  const startMs = Date.now();
  const { results } = await search(rawQuery, ctx);
  const elapsedMs = Date.now() - startMs;

  if (results.length === 0) {
    console.log('No results found.');
    return;
  }

  console.log(`Found ${results.length} candidates in ${(elapsedMs / 1000).toFixed(1)}s\n`);
  console.log('='.repeat(80));

  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const flag = r.facepalm ? ' ⚠ FACEPALM' : '';
    console.log(`\n#${i + 1}${flag}  ${r.profile.name} — ${r.profile.title}`);
    console.log(
      `     ${r.profile.seniority} | ${r.profile.location.city}, ${r.profile.location.country} | ${r.profile.yearsOfExperience} yrs exp`,
    );
    console.log(`     Skills: ${r.profile.skills.join(', ')}`);
    console.log(`     L1 score: ${r.l1Score.toFixed(4)}  |  L2 score: ${r.l2Score.toFixed(4)}  |  fit assessment: ${r.fit}`);
    console.log(`\n     ${r.guardExplanation}`);
    console.log('-'.repeat(80));
  }

  console.log(`\nTotal pipeline latency: ${(elapsedMs / 1000).toFixed(2)}s`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
