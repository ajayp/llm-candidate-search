import * as dotenv from 'dotenv'; dotenv.config({ override: true });
import fs from 'fs';
import { generateProfiles } from '../src/synthetic/generator';
import { CONFIG } from '../src/config';

async function main() {
  console.log('MUSE-PoC: Generating synthetic candidate profiles...\n');

  if (fs.existsSync(CONFIG.data.profilesPath)) {
    const existing = JSON.parse(fs.readFileSync(CONFIG.data.profilesPath, 'utf-8'));
    if (existing.length >= CONFIG.data.profileCount) {
      console.log(`Already complete (${existing.length} profiles). Delete ${CONFIG.data.profilesPath} to regenerate.`);
      return;
    }
  }

  const profiles = await generateProfiles(CONFIG.data.profilesPath);


  console.log(`\n✓ Generated ${profiles.length} profiles → ${CONFIG.data.profilesPath}`);

  // Print summary stats
  console.log('\nDistribution by seniority:');
  const bySeniority = new Map<string, number>();
  for (const p of profiles) {
    bySeniority.set(p.seniority, (bySeniority.get(p.seniority) ?? 0) + 1);
  }
  for (const [level, count] of [...bySeniority.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    console.log(`  ${level.padEnd(12)}: ${count}`);
  }

  console.log('\nDistribution by location (top 10):');
  const byLocation = new Map<string, number>();
  for (const p of profiles) {
    const key = `${p.location.city}, ${p.location.country}`;
    byLocation.set(key, (byLocation.get(key) ?? 0) + 1);
  }
  const sortedLocations = [...byLocation.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
  for (const [loc, count] of sortedLocations) {
    console.log(`  ${loc.padEnd(30)}: ${count}`);
  }
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
