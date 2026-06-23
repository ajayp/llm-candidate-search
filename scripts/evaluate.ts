import * as dotenv from 'dotenv'; dotenv.config({ override: true });
import fs from 'fs';
import { EvalQuery } from '../src/types';
import { CONFIG } from '../src/config';
import { loadPipelineContext, RankingMode, search } from '../src/pipeline/pipeline';
import { understandQuery } from '../src/pipeline/queryUnderstanding';
import { generateHyDE } from '../src/pipeline/hyde';
import { computeNDCG, computeRecall } from '../src/utils';

const MODES: RankingMode[] = ['l1', 'cosine', 'cohere'];
const MODE_LABELS: Record<RankingMode, string> = {
  l1:     'L1 (cosine-512)',
  cosine: 'L2 (cosine-3072)',
  cohere: 'L2 (Cohere x-enc)',
};

interface ModeMetrics {
  recall10: number;
  ndcg10: number;
}

interface QueryResult {
  queryId: string;
  query: string;
  recall50: number;
  relevantCount: number;
  byMode: Record<RankingMode, ModeMetrics>;
}

async function main() {
  console.log('MUSE-PoC: Running Matryoshka comparison evaluation...\n');
  console.log('Modes: L1 baseline → cosine-3072 rerank → Cohere cross-encoder\n');

  if (!fs.existsSync(CONFIG.data.evalQueriesPath)) {
    console.error(`Eval queries not found at ${CONFIG.data.evalQueriesPath}. Run "npm run eval:generate" first.`);
    process.exit(1);
  }

  const evalQueriesRaw = JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesPath, 'utf-8'));
  const evalQueries = (Array.isArray(evalQueriesRaw[0]) ? evalQueriesRaw[0] : evalQueriesRaw) as EvalQuery[];

  const limitArg = process.argv.find((a) => a.startsWith('--limit='));
  const limit = limitArg ? parseInt(limitArg.split('=')[1], 10) : evalQueries.length;
  const queries = evalQueries.slice(0, limit);
  if (limit < evalQueries.length) console.log(`Running ${limit}/${evalQueries.length} queries (--limit=${limit}).\n`);

  console.log('Loading pipeline context...');
  const ctx = loadPipelineContext();

  const queryResults: QueryResult[] = [];

  for (let i = 0; i < queries.length; i++) {
    const eq = queries[i];
    const relevant = new Set(eq.relevantProfileIds);
    console.log(`\n[${i + 1}/${queries.length}] "${eq.raw.slice(0, 70)}"`);

    const byMode = {} as Record<RankingMode, ModeMetrics>;
    let recall50 = 0;

    console.log('[Stage 1] Query understanding...');
    const structuredQuery = await understandQuery(eq.raw);
    console.log('[Stage 1] Done:', JSON.stringify(structuredQuery, null, 2));
    console.log('[Stage 1b] Generating HyDE...');
    const hydeText = await generateHyDE(structuredQuery);
    console.log('[Stage 1b] HyDE:', hydeText);

    for (const mode of MODES) {
      const { results, l1CandidateIds } = await search(eq.raw, ctx, { skipGuard: true, rankingMode: mode, structuredQuery, hydeText });
      const rankedIds = results.map((r) => r.profile.id);

      byMode[mode] = {
        recall10: computeRecall(rankedIds, relevant, 10),
        ndcg10: computeNDCG(rankedIds, relevant, 10),
      };

      // Recall@50 is the same across modes (L1 pool is identical); capture it once
      if (mode === 'l1') {
        recall50 = computeRecall(l1CandidateIds, relevant, 50);
      }

      process.stdout.write(`  ${MODE_LABELS[mode].padEnd(18)} R@10=${byMode[mode].recall10.toFixed(3)}  NDCG@10=${byMode[mode].ndcg10.toFixed(3)}\n`);
    }

    queryResults.push({ queryId: eq.id, query: eq.raw, recall50, relevantCount: relevant.size, byMode });
  }

  // Macro averages per mode
  const avg = (mode: RankingMode, key: keyof ModeMetrics) =>
    queryResults.reduce((s, r) => s + r.byMode[mode][key], 0) / queryResults.length;
  const avgRecall50 = queryResults.reduce((s, r) => s + r.recall50, 0) / queryResults.length;

  const W = 100;
  console.log('\n' + '='.repeat(W));
  console.log('MATRYOSHKA COMPARISON — MACRO AVERAGES');
  console.log('='.repeat(W));

  const hdr = `${''.padEnd(52)} ${'R@10'.padStart(8)} ${'NDCG@10'.padStart(10)} ${'R@50(L1)'.padStart(10)}`;
  console.log(hdr);
  console.log('-'.repeat(W));

  for (const mode of MODES) {
    const r10 = avg(mode, 'recall10');
    const n10 = avg(mode, 'ndcg10');
    const r50 = mode === 'l1' ? avgRecall50 : 0;
    const r50str = mode === 'l1' ? r50.toFixed(3).padStart(10) : '          ';
    console.log(`  ${MODE_LABELS[mode].padEnd(50)} ${r10.toFixed(3).padStart(8)} ${n10.toFixed(3).padStart(10)} ${r50str}`);
  }

  console.log('-'.repeat(W));

  // Delta rows
  const l1n = avg('l1', 'ndcg10');
  const cosn = avg('cosine', 'ndcg10');
  const cohn = avg('cohere', 'ndcg10');
  console.log(`\n  Matryoshka effect (cosine-3072 vs cosine-512): NDCG@10 ${cosn >= l1n ? '+' : ''}${((cosn - l1n) * 100).toFixed(1)}pp`);
  console.log(`  Cross-encoder gain (Cohere vs cosine-3072):    NDCG@10 ${cohn >= cosn ? '+' : ''}${((cohn - cosn) * 100).toFixed(1)}pp`);
  console.log(`  Full pipeline gain (Cohere vs L1 baseline):    NDCG@10 ${cohn >= l1n ? '+' : ''}${((cohn - l1n) * 100).toFixed(1)}pp`);

  console.log('\n' + '='.repeat(W));
  console.log('PER-QUERY DETAIL');
  console.log('='.repeat(W));
  const qhdr = `${'Query'.padEnd(52)} ${'L1 N@10'.padStart(8)} ${'Cos N@10'.padStart(10)} ${'Coh N@10'.padStart(10)}`;
  console.log(qhdr);
  console.log('-'.repeat(W));
  for (const r of queryResults) {
    console.log(
      `  ${r.query.slice(0, 50).padEnd(50)} ` +
      `${r.byMode.l1.ndcg10.toFixed(3).padStart(8)} ` +
      `${r.byMode.cosine.ndcg10.toFixed(3).padStart(10)} ` +
      `${r.byMode.cohere.ndcg10.toFixed(3).padStart(10)}`,
    );
  }
  console.log('='.repeat(W));

  console.log(`\nTargets: Recall@10 > 0.50, NDCG@10 > 0.45 (Cohere L2)`);
  console.log(`Cohere:  Recall@10 = ${avg('cohere', 'recall10').toFixed(3)}, NDCG@10 = ${avg('cohere', 'ndcg10').toFixed(3)}`);

  const evalResults = {
    runAt: new Date().toISOString(),
    macroAverages: Object.fromEntries(
      MODES.map((m) => [m, { recall10: avg(m, 'recall10'), ndcg10: avg(m, 'ndcg10') }]),
    ),
    recall50L1: avgRecall50,
    perQuery: queryResults,
  };

  fs.writeFileSync(CONFIG.data.evalResultsPath, JSON.stringify(evalResults, null, 2));
  console.log(`\nResults written → ${CONFIG.data.evalResultsPath}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
