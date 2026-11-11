# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A proof-of-concept semantic candidate search system ("HASS") that replicates the architecture of LinkedIn's public Hiring Assistant blog posts at small scale (724 synthetic profiles). It turns a free-text recruiter query into a ranked candidate list via a 4-stage LLM + retrieval pipeline. See README.md for the full design rationale, measured results (HyDE recall improvement, Matryoshka dimensionality experiment), and architecture diagrams — read it before making pipeline changes, since most design decisions there are backed by specific before/after numbers that should stay valid.

## Commands

```bash
npm install
cp .env.example .env        # requires OPENAI_API_KEY and COHERE_API_KEY

npm run search "<query>"    # run one query through the live pipeline (interactive demo)
npm run evaluate             # full offline eval (NDCG@10, R@50) against data/eval/eval_queries.json
npm run evaluate -- --limit=5   # quick smoke test, fewer queries

npm test                     # jest, runs src/__tests__/**/*.test.ts
npx jest src/__tests__/matchStats.test.ts   # run a single test file
npx jest -t "test name"      # run tests matching a name

npm run tscheck               # tsc --noEmit type check

# One-time / regenerate-the-data-directory scripts:
npm run generate              # regenerate the 724 synthetic profiles (data/profiles.json is pre-committed, so rarely needed)
npm run index                 # build data/generated/ (embeddings cache + FAISS index) from profiles — required after a fresh clone, since data/generated/ is gitignored
npm run eval:generate          # regenerate ground-truth labels via OpenAI Batch API (14,480 judgments — costs money/time)
npm run eval:generate:quick    # faster/cheaper ground-truth generation for iteration
npm run eval:collect            # collect batch API results once ready
```

Scripts are run directly via `ts-node/register/transpile-only` (no build step) with `.env` auto-loaded — see `scripts/loadEnvOverride.js` and the `--env-file-if-exists=.env` flag in `package.json`.

## Architecture

The pipeline (`src/pipeline/pipeline.ts`, entry point `search()`) runs four stages plus one side computation, in order:

1. **Query understanding** (`queryUnderstanding.ts`) — GPT-4o extracts a `StructuredQuery` (title, seniority, location, required/ambiguous qualifications) from raw text. Can be skipped by passing a pre-parsed `structuredQuery` (used by eval to parse once and reuse across ranking-mode comparisons).
2. **HyDE generation** (`hyde.ts`) — GPT-4o-mini synthesizes a fake "ideal candidate" profile from the structured query; this is what actually gets embedded, not the raw query, to close the embedding-asymmetry gap between short queries and long profile documents (see README's "Key Finding" section for the measured recall impact — this is the single most load-bearing decision in the system). `queryUnderstanding.ts` and `hyde.ts` both derive role/skills/location phrasing via the shared `deriveQueryFields()` helper (`queryFields.ts`) so the two never independently drift on that derivation.
3. **L1 retrieval + ABM** (`retrieval.ts`) — FAISS exact search (`IndexFlatL2`, 512-dim Matryoshka slice) over the HyDE-embedded query, then a rule-based attribute filter (seniority/location) narrows ~600 → ~100 candidates. `matchStats.ts` branches off this same output to compute `distinguishingSkills` (term-frequency enrichment vs. the full corpus) — pure computation, no LLM, never blocks or fails the main path.
4. **L2 reranking** (`reranking.ts`) — mode-dependent (`RankingMode`: `'l1' | 'cosine' | 'cohere'`). Default `'cohere'` uses Cohere's cross-encoder reranker on the full 3072-dim text; `'cosine'` and `'l1'` exist mainly for eval comparisons.
5. **LLM guard** (`guard.ts`) — GPT-4o-mini gives each surviving candidate a structured `fit` verdict (`poor|partial|good|excellent`) with a per-qualification checklist; `poor` fits are dropped. Code — not the model — enforces that the `fit` label is consistent with the qualification-check count, and separately flags `facepalm` cases (seniority mismatch ≥ 2 ranks, or an IC↔management track mismatch regardless of rank gap) via `SENIORITY_RANK` in `config.ts`.

`loadPipelineContext()` loads all shared state once (profiles, embedding cache, FAISS index, background skill frequencies) and is meant to be reused across multiple `search()` calls (e.g. across eval queries), not reconstructed per-query.

**Acronym handling** (`src/acronyms.ts`) is a single deterministic lookup table, resolved once in Stage 1 (`resolveQualification()`, called from `queryUnderstanding.ts`). Its output — expanded qualification text plus which ones are flagged `ambiguous` (genuinely ambiguous ones like `TS` are left unexpanded rather than guessed) — rides along in `StructuredQuery` into the Stage 4 guard prompt, which is what lets the guard tell the LLM to verify an ambiguous abbreviation literally instead of inferring what it means. Guard never imports `acronyms.ts` directly; if you touch query understanding, that's still the only place resolution happens, so don't duplicate it in the guard prompt.

**Config** (`src/config.ts`) centralizes all model names, dimensions, thresholds (`l1MinScore`, `l2MinScore`, `finalTopK`, etc.), and file paths — check here first before hardcoding a tunable value elsewhere. Classification-style LLM calls (query understanding, guard) use `temperature: 0` + a fixed `seed` for reproducibility; HyDE generation deliberately does not, since it's generative-by-design.

**Data directory** (`data/`) splits committed source data from generated build artifacts. `data/profiles.json` and `data/eval/` (`eval_queries_raw.json`, `eval_queries.json`) are checked in. `data/generated/` (`embeddings_cache.json` — 67MB, `index.faiss`, `profile_id_map.json`, `eval_results.json`) is entirely gitignored and rebuilt via `npm run index`; run it once after cloning before `npm run search` or `npm run evaluate` will work.
