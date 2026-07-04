import fs from 'fs';
import OpenAI from 'openai';
import { CandidateProfile, EvalQuery, EvalQueryRaw } from '../src/types';
import { CONFIG } from '../src/config';

const PROFILE_BATCH_SIZE = 10;
const QUERIES_PER_GROUP = 5;
const POLL_INTERVAL_MS = 30_000;
const STATE_PATH = 'data/eval_batch_state.json';
const INPUT_JSONL_PATH = 'data/eval_batch_input.jsonl';

interface GroupState {
  queryIds: string[];
  groupFile: string;
  batchId: string | null;
  outputFileId: string | null;
  status: 'pending' | 'submitted' | 'completed';
}

interface State {
  groups: GroupState[];
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

function splitInputIntoGroups(rawQueries: EvalQueryRaw[]): State {
  if (!fs.existsSync(INPUT_JSONL_PATH)) {
    console.error(`Input JSONL not found at ${INPUT_JSONL_PATH}. Run eval:generate:build first.`);
    process.exit(1);
  }

  // Index all lines by queryId
  const linesByQuery = new Map<string, string[]>();
  for (const line of fs.readFileSync(INPUT_JSONL_PATH, 'utf-8').split('\n').filter(Boolean)) {
    const obj = JSON.parse(line) as { custom_id: string };
    const queryId = obj.custom_id.split('__b')[0];
    if (!linesByQuery.has(queryId)) linesByQuery.set(queryId, []);
    linesByQuery.get(queryId)!.push(line);
  }

  const groups: GroupState[] = [];
  for (let i = 0; i < rawQueries.length; i += QUERIES_PER_GROUP) {
    const groupQueryIds = rawQueries.slice(i, i + QUERIES_PER_GROUP).map((q) => q.id);
    const groupIndex = Math.floor(i / QUERIES_PER_GROUP) + 1;
    const groupFile = `data/eval_batch_group_${groupIndex}.jsonl`;

    const groupLines: string[] = [];
    for (const qid of groupQueryIds) {
      groupLines.push(...(linesByQuery.get(qid) ?? []));
    }
    fs.writeFileSync(groupFile, groupLines.join('\n'));
    console.log(`  Written ${groupFile} (${groupLines.length} requests)`);

    groups.push({ queryIds: groupQueryIds, groupFile, batchId: null, outputFileId: null, status: 'pending' });
  }

  const state: State = { groups };
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));
  return state;
}

function saveState(state: State): void {
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2));
}

async function submitGroup(group: GroupState): Promise<string> {
  const fileContent = fs.readFileSync(group.groupFile);
  const uploaded = await client.files.create({
    file: new File([fileContent], group.groupFile.split('/').pop()!, { type: 'application/jsonl' }),
    purpose: 'batch',
  });
  console.log(`  Uploaded file: ${uploaded.id}`);

  const batch = await client.batches.create({
    input_file_id: uploaded.id,
    endpoint: '/v1/chat/completions',
    completion_window: '24h',
  });
  console.log(`  Batch created: ${batch.id}`);
  return batch.id;
}

async function pollUntilDone(batchId: string): Promise<string> {
  while (true) {
    const batch = await client.batches.retrieve(batchId);
    const completed = batch.request_counts?.completed ?? 0;
    const total = batch.request_counts?.total ?? 0;
    process.stdout.write(`  Polling: ${batch.status} (${completed}/${total} complete)\r`);

    if (batch.status === 'completed') {
      console.log(`\n  Done: ${completed}/${total} requests completed.`);
      return batch.output_file_id!;
    }

    if (batch.status === 'failed' || batch.status === 'cancelled' || batch.status === 'expired') {
      console.error(`\n  Batch ${batch.status}.`);
      if (batch.errors?.data?.length) {
        console.error('  Errors:', JSON.stringify(batch.errors.data, null, 2));
      }
      process.exit(1);
    }

    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
}

function parseGroupResults(
  outputJsonl: string,
  groupQueryIds: string[],
  profiles: CandidateProfile[],
  rawQueries: EvalQueryRaw[],
): EvalQuery[] {
  const queryRawText = new Map(rawQueries.map((q) => [q.id, q.raw]));

  // Build customId → profile slice map using the same PROFILE_BATCH_SIZE and profiles order
  const profileBatches = new Map<string, CandidateProfile[]>();
  for (const queryId of groupQueryIds) {
    const numBatches = Math.ceil(profiles.length / PROFILE_BATCH_SIZE);
    for (let b = 0; b < numBatches; b++) {
      profileBatches.set(
        `${queryId}__b${b}`,
        profiles.slice(b * PROFILE_BATCH_SIZE, (b + 1) * PROFILE_BATCH_SIZE),
      );
    }
  }

  const qualifiedByQuery = new Map<string, Set<string>>();
  for (const queryId of groupQueryIds) qualifiedByQuery.set(queryId, new Set());

  for (const line of outputJsonl.split('\n').filter(Boolean)) {
    const result = JSON.parse(line) as {
      custom_id: string;
      response: { body: { choices: Array<{ message: { content: string } }> } };
    };

    const customId = result.custom_id;
    const queryId = customId.split('__b')[0];
    const batchProfiles = profileBatches.get(customId);
    if (!batchProfiles || !qualifiedByQuery.has(queryId)) continue;

    const content = result.response?.body?.choices?.[0]?.message?.content ?? '{}';
    let parsed: Record<string, string> = {};
    const match = content.match(/\{[\s\S]*\}/);
    if (match) {
      try { parsed = JSON.parse(match[0]) as Record<string, string>; } catch { /* skip */ }
    }

    for (const [idx, verdict] of Object.entries(parsed)) {
      const profile = batchProfiles[parseInt(idx)];
      if (profile && verdict === 'QUALIFIED') {
        qualifiedByQuery.get(queryId)!.add(profile.id);
      }
    }
  }

  return groupQueryIds.map((queryId) => ({
    id: queryId,
    raw: queryRawText.get(queryId)!,
    relevantProfileIds: Array.from(qualifiedByQuery.get(queryId) ?? []),
  }));
}

async function main() {
  console.log('MUSE-PoC: Generating eval ground-truth labels (Batch API, grouped)...\n');

  if (!fs.existsSync(CONFIG.data.evalQueriesRawPath)) {
    console.error(`Raw eval queries not found at ${CONFIG.data.evalQueriesRawPath}`);
    process.exit(1);
  }

  const rawQueries = JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesRawPath, 'utf-8')) as EvalQueryRaw[];
  const profiles = JSON.parse(fs.readFileSync(CONFIG.data.profilesPath, 'utf-8')) as CandidateProfile[];

  // Load already-completed labels
  const evalQueries: EvalQuery[] = fs.existsSync(CONFIG.data.evalQueriesPath)
    ? (() => {
        const raw = JSON.parse(fs.readFileSync(CONFIG.data.evalQueriesPath, 'utf-8')) as EvalQuery[] | EvalQuery[][];
        return (Array.isArray(raw[0]) ? raw[0] : raw) as EvalQuery[];
      })()
    : [];
  const completedQueryIds = new Set(evalQueries.map((q) => q.id));

  // Load or create state (splits input JSONL into group files on first run)
  let state: State;
  if (fs.existsSync(STATE_PATH)) {
    state = JSON.parse(fs.readFileSync(STATE_PATH, 'utf-8')) as State;
    console.log(`Resuming from saved state (${state.groups.length} groups).\n`);
  } else {
    console.log(`Splitting input JSONL into groups of ${QUERIES_PER_GROUP} queries...`);
    state = splitInputIntoGroups(rawQueries);
    console.log(`\nCreated ${state.groups.length} group files.\n`);
  }

  for (let g = 0; g < state.groups.length; g++) {
    const group = state.groups[g];
    const label = `[Group ${g + 1}/${state.groups.length}]`;

    // Skip if this group was already completed in this batch run
    if (group.status === 'completed') {
      console.log(`${label} Already complete, skipping.`);
      continue;
    }

    // Skip if every query in this group already has labels from a prior run
    // (state.json is deleted on full completion, so a re-invocation would
    // otherwise resubmit and duplicate every query with fresh, non-deterministic labels)
    if (group.queryIds.every((id) => completedQueryIds.has(id))) {
      console.log(`${label} All queries already labeled, skipping.`);
      continue;
    }

    console.log(`\n${label} Queries: ${group.queryIds.join(', ')}`);

    // Submit if not yet submitted
    if (group.status === 'pending') {
      group.batchId = await submitGroup(group);
      group.status = 'submitted';
      saveState(state);
    } else {
      console.log(`  Resuming batch: ${group.batchId}`);
    }

    // Check for a pre-downloaded local output file (batch_{id}_output.jsonl)
    const localOutputPath = `data/batch_${group.batchId}_output.jsonl`;
    let outputJsonl: string;
    if (fs.existsSync(localOutputPath)) {
      console.log(`  Using local output file: ${localOutputPath}`);
      outputJsonl = fs.readFileSync(localOutputPath, 'utf-8');
      group.outputFileId = localOutputPath;
      group.status = 'completed';
      saveState(state);
    } else {
      // Poll until the batch completes
      const outputFileId = await pollUntilDone(group.batchId!);
      group.outputFileId = outputFileId;
      group.status = 'completed';
      saveState(state);

      const fileContent = await client.files.content(outputFileId);
      outputJsonl = await fileContent.text();
    }
    const groupResults = parseGroupResults(outputJsonl, group.queryIds, profiles, rawQueries);

    for (const r of groupResults) {
      completedQueryIds.add(r.id);
      const existingIdx = evalQueries.findIndex((q) => q.id === r.id);
      if (existingIdx >= 0) evalQueries[existingIdx] = r;
      else evalQueries.push(r);
    }
    fs.writeFileSync(CONFIG.data.evalQueriesPath, JSON.stringify(evalQueries, null, 2));

    const avgQualified = groupResults.reduce((s, q) => s + q.relevantProfileIds.length, 0) / groupResults.length;
    console.log(`  ✓ Results written. Avg ${avgQualified.toFixed(1)} relevant per query.`);
  }

  const avgRelevant = evalQueries.reduce((s, q) => s + q.relevantProfileIds.length, 0) / evalQueries.length;
  console.log(`\n✓ All groups complete → ${CONFIG.data.evalQueriesPath}`);
  console.log(`Average relevant per query: ${avgRelevant.toFixed(1)}`);

  fs.unlinkSync(STATE_PATH);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
