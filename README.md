# Hiring Assistant Search System

Hiring Assistant Search System (HASS) is an AI-powered candidate search system that converts natural language recruiter job descriptions into structured, ranked candidate matches using multi-stage semantic retrieval, reranking, and LLM-based evaluation.

In 2023 I built a recruiting assistant and hit these limits directly. HASS is the version that addresses them.

**Stack:** TypeScript · FAISS · OpenAI `text-embedding-3-large` · GPT-4o / 4o-mini · Cohere `rerank-v4.0-pro` · OpenAI Batch API

## What This PoC Tries to Prove

That the retrieval-architecture shape described in LinkedIn's public Hiring Assistant engineering blogs (query understanding → embedding retrieval → attribute filter → rerank → LLM evaluation) can be replicated at small scale with off-the-shelf models, and that the *engineering judgment calls* behind it — not just the pipeline shape — hold up under measurement. Two claims are empirically tested here, not asserted:

1. **HyDE-based retrieval fixes a real, measured embedding-asymmetry recall gap** — R@50 went from 0.287 → 0.442 (see [Results & Deep Dive](#results--deep-dive)).
2. **Reducing embedding dimensionality for first-stage retrieval costs nothing in ranking quality on a generic embedding model** — +0.5pp NDCG (noise-level), which is consistent with — and arrived at independently before reading — LinkedIn's own rationale for why their fine-tuned MUSE model needs task-specific training to make higher dimensions worth the latency (see [Matryoshka Experiment](#matryoshka-experiment-diminishing-returns-on-embedding-dimensions)).

This is a research PoC on retrieval tradeoffs, not a production system: 724 synthetic profiles, and **exact (not approximate) nearest-neighbor search**, chosen deliberately because the corpus is far below the scale where ANN's recall/speed tradeoff would actually matter — see the FAISS decision below.

## Table of Contents

- [The Problem](#the-problem)
- [Key Technical Decisions](#key-technical-decisions)
- [Results & Deep Dive](#results--deep-dive)
  - [Matryoshka Experiment](#matryoshka-experiment-diminishing-returns-on-embedding-dimensions)
- [Architecture](#architecture)
- [Distinguishing Skills: Statistical Explainability (No LLM)](#distinguishing-skills-statistical-explainability-no-llm)
- [Acronym Handling: A Documented Limitation, Not a Fix](#acronym-handling-a-documented-limitation-not-a-fix)
- [Evaluation Methodology](#evaluation-methodology)
- [Example Sourcing Walkthrough](#example-sourcing-walkthrough)
- [Future Roadmap](#future-roadmap)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)

---

## The Problem

Recruiters rarely search using a handful of isolated keywords. Instead, they describe their ideal candidate in natural language:

> "Director of Engineering with 8+ years, distributed systems background, New York only."

Traditional search engines struggle with this—keyword search misses semantically equivalent profiles, faceted filters are brittle to construct from free-text, and nuanced qualification fit rarely maps cleanly to structured database fields.

---

## Key Technical Decisions

| Decision | Why |
|-----------|-----|
| **HyDE Retrieval** | Resolves embedding asymmetry; transforms short queries into profile-shaped documents to maximize recall before reranking. |
| **FAISS Exact Search (`IndexFlatL2`)** | Deliberately not ANN. At 724 profiles, brute-force cosine search is sub-millisecond with zero recall loss. Approximate indexing (IVF/IVFPQ) only pays off once brute-force becomes too slow — millions of vectors, not hundreds — so it's the wrong tool to reach for at this scale. |
| **Cohere Reranker** | Utilizes a powerful cross-encoder to compute deep, interactive qualification alignment far better than raw cosine similarity. |
| **Ground Truth via Batch API** | Maximizes cost and time efficiency to generate a dense evaluation dataset for quantitative performance benchmarking. |
| **Synthetic Dataset** | Allowed full control over skill, seniority, and location distributions to reliably stress-test edge cases. |
| **512-dim at L1 / 3072-dim at L2** | Search time and index memory scale with embedding dimension regardless of exact-vs-approximate search. 512-dim keeps first-stage retrieval fast across large corpora; the full dimension is only applied at L2 where you're scoring a small top-N, not the entire index. |
| **Term-Frequency Explainability (no LLM)** | A second "why this candidate" signal — which of a candidate's skills are statistically overrepresented in the retrieved pool vs. the full corpus — computed with plain term-frequency math, not a model call. Zero marginal cost, zero hallucination risk, sits alongside the LLM guard's explanation rather than replacing it. See [Distinguishing Skills](#distinguishing-skills-statistical-explainability-no-llm). |
| **Deterministic, Shared Acronym Resolution** | A small static table (not an LLM judgment call) resolves unambiguous abbreviations (`K8s`→Kubernetes) and explicitly flags genuinely ambiguous ones (`TS`) as unresolved — used identically by Stage 1 and Stage 4 so the two can never silently disagree about what an abbreviation means. Does not resolve genuine ambiguity, only makes not-knowing consistent and visible. See [Acronym Handling](#acronym-handling-a-documented-limitation-not-a-fix). |

---

## Results & Deep Dive

| Configuration      |   NDCG@10 |      R@50 |
| ------------------ | --------: | --------: |
| Baseline Retrieval |     0.581 |     0.287 |
| **+ HyDE Retrieval**   | **0.715** | **0.442** |

### Key Finding: Fix Retrieval Before Optimizing Precision

The largest performance leap did not come from a heavier reranking model; it came from fixing early-stage retrieval.

The baseline system embedded short recruiter queries directly (e.g., *"Senior ML engineer, expertise in Python, NLP"*). Because candidate profiles are long, comprehensive documents, their vectors lived in a different region of the embedding space. Many highly qualified candidates were missed entirely in Stage 1.

By introducing **HyDE**, GPT-4o-mini generates a synthetic "ideal profile" from the query first. Embedding this profile-shaped text aligns perfectly with the FAISS index structure. 

**Result:** R@50 shot up from **0.287 → 0.442 (+54%)** purely by optimizing query-side geometry without altering the underlying index.

### Matryoshka Experiment: Diminishing Returns on Embedding Dimensions

`text-embedding-3-large` supports Matryoshka Representation Learning (MRL) — lower-dimensional prefixes of the 3072-dim vector remain valid embeddings. The motivation: search time and index memory scale with dimension, so using 512-dim at L1 keeps retrieval fast, while reserving the full 3072-dim for L2 where you're scoring a small top-N.

**Hypothesis:** The extra dimensions encode finer-grained qualification signals that would improve ranking precision at L2.

**Result:** +0.5pp NDCG@10 across multiple runs — effectively noise.

**Why:** For a general-purpose model without domain-specific fine-tuning, most semantic signal is already captured in the first 512 dimensions. A task-specific MRL model trained on (query, profile, relevance) triplets would front-load coarse signals in lower dimensions and reserve higher dimensions for nuanced qualification reasoning — that's when the larger budget matters. Without that training, the extra dimensions add latency without meaningful signal.

---

## Architecture

```mermaid
flowchart TD
    A([Free-text recruiter query]) --> B

    B["[Stage 1] Query Understanding 🤖
    ─────────────────────────
    GPT-4o extracts title, seniority,
    location, and required skills"]

    B --> C["[Stage 1b] HyDE Generation 🤖
    ─────────────────────────
    GPT-4o-mini generates a synthetic
    ideal-candidate profile for embedding"]

    C --> D["[Stage 2] Retrieval
    ─────────────────────────
    FAISS exact search (IndexFlatL2, 512-dim)
    + seniority &amp; location ABM filter
    600 → ~100 candidates"]

    D --> E["[Stage 3] Reranking 🤖
    ─────────────────────────
    Cohere cross-encoder scores
    deep candidate relevance
    ~100 → 25 candidates"]

    D --> M["Distinguishing Skills 📊
    ─────────────────────────
    Term-frequency stats: ~100 retrieved
    candidates (foreground) vs. full
    724-profile corpus (background)
    No LLM call"]

    E --> F["[Stage 4] LLM Fit Assessment 🤖
    ─────────────────────────
    GPT-4o-mini evaluates each candidate:
    poor / partial / good / excellent
    Filters poor fits"]

    M --> G
    F --> G([Ranked candidate list:
    fit + guardExplanation + distinguishingSkills])
```

🤖 = LLM call · 📊 = pure computation, no model call. Notice `Distinguishing Skills` branches off the *same* Stage 2 output that feeds Stage 3 — it runs independently, in parallel, and never touches an LLM; it only rejoins the pipeline at the very end, attached to whichever candidates survive Stage 3/4. See [below](#distinguishing-skills-statistical-explainability-no-llm) for why, and the sequence diagram for exactly how data threads through.

---

## Distinguishing Skills: Statistical Explainability (No LLM)

Stage 4's LLM guard already explains *why a candidate matches the query* (title/seniority/location/required skills). This second signal answers a different question — *what makes this candidate statistically typical of the pool that got retrieved* — using no model call at all. It's inspired by Solr's Semantic Knowledge Graph (`relatedness()`): compare how common a term is in a "foreground" set vs. a "background" set; a term that's disproportionately common in the foreground is distinguishing.

Applied here:
- **Foreground** = the ~100 candidates Stage 2 already retrieved for this query.
- **Background** = the full 724-profile corpus.
- **Enrichment** per skill = `(fraction of foreground with the skill) / (fraction of background with the skill)`. Since the foreground is always a subset of the background, this ratio is always defined — no smoothing needed.
- Per surviving candidate, keep only *their own* skills that score > 1 (enriched), sorted descending, top 3. Ties (common for skills that appear only once in the whole corpus) break by how many of the retrieved pool shares the skill, then alphabetically — so output is always deterministic, never arbitrary insertion order.

The sequence diagram below makes explicit which stages call an LLM and which don't, and exactly where this signal branches off and rejoins:

```mermaid
sequenceDiagram
    actor R as Recruiter
    participant Q as Stage 1: Query Understanding
    participant H as Stage 1b: HyDE Generation
    participant L1 as Stage 2: L1 Retrieval + ABM
    participant MS as Distinguishing Skills
    participant L2 as Stage 3: L2 Reranking
    participant G as Stage 4: LLM Guard

    R->>Q: "Senior ML engineer, Python, NLP, Bay Area"
    Note over Q: 🤖 LLM call (GPT-4o)
    Q->>H: structured query (title, seniority, location, required skills)
    Note over H: 🤖 LLM call (GPT-4o-mini)
    H->>L1: synthetic ideal-candidate profile text (for embedding)
    Note over L1: 📊 No LLM — FAISS exact search (IndexFlatL2) + rule-based location/seniority filter
    L1->>L2: ~100 ABM-filtered candidates
    L1->>MS: same ~100 candidates, as the "foreground" set
    Note over MS: 📊 No LLM — term-frequency ratio vs. the full 724-profile "background" corpus
    L2->>G: top 25 reranked candidates
    Note over G: 🤖 LLM call (GPT-4o-mini), one call per candidate
    MS-->>G: distinguishingSkills[] per candidate (computed independently, attached at the end)
    G->>R: ranked list — fit, guardExplanation, distinguishingSkills
```

The key thing this diagram is meant to clarify: `Distinguishing Skills` is **not** a step in the LLM chain (Stage 1 → 1b → 4). It's a side branch off Stage 2's output, computed once per query in plain JavaScript, and merged back in only when the final result list is assembled — it can never fail, retry, hallucinate, or cost an API call, because it never talks to a model.

### Validated Across Job Categories

Ran three live queries spanning very different skill vocabularies to confirm the signal isn't secretly tech-biased or fragile outside the ML-engineer happy path:

| Query | Job category | Distinguishing skills surfaced |
| --- | --- | --- |
| *"Senior ML engineer, Python, NLP background, Bay Area"* | Tech / IC | Hugging Face, RAG, Feature Stores, MLflow — niche ML tooling, never the shared basics (Python, Machine Learning) |
| *"Director of Engineering with 8+ years, distributed systems background"* | Tech / management | Bioinformatics, Compliance (SOC2, HIPAA), Channel Partner Management — domain-specific leadership skills, correctly varied per candidate |
| *"Senior marketing manager with SEO and content strategy experience"* | Non-tech | marketing automation, Brand Voice, Attribution Modeling |

Across all three: no crashes, no `NaN`/undefined values, sensible multipliers throughout (no arbitrary-looking ties — the tiebreak logic held up on real, varied data, not just hand-built test fixtures).

---

## Acronym Handling: A Documented Limitation, Not a Fix

Recruiter queries are full of abbreviations, and some are genuinely ambiguous — a query like *"knows TS"* could mean TypeScript or Time Series, and there's no way to know which without more context. Testing live surfaced two real failure modes from this:

1. **Stage 1 leaving an abbreviation unresolved is fine — Stage 4 silently guessing one is not.** In one run, Stage 1 correctly left `"TS"` as written, but Stage 4's guard, in a separate LLM call with no visibility into Stage 1's caution, confidently decided on its own that `"TS"` meant TypeScript and penalized every candidate for lacking it — an interpretation nobody asked for, invisible to whoever reads the result.
2. **A single LLM call isn't reliably consistent even with itself.** Before this fix, the guard could write an explanation naming 2+ missing qualifications while still labeling the candidate `good` — the label and the reasoning behind it simply disagreed.

**What's implemented (`src/acronyms.ts`):** a small, deterministic, shared lookup — not an LLM judgment call — used identically by Stage 1 and Stage 4:
- Genuinely unambiguous abbreviations (`K8s`→Kubernetes) are expanded in code, once, so both stages see the same resolved term.
- Abbreviations already standard-as-written (`NLP`, `RAG`, `LLM`, ...) are left alone — expanding these would break matches against profiles that use the same short form.
- Anything else abbreviation-shaped and unrecognized (`TS`) is flagged `ambiguous` and passed through to Stage 4 with an explicit instruction: verify the literal string only, never substitute a guessed technology, and phrase the explanation around the unresolved abbreviation rather than a confident but invented interpretation.

Separately, Stage 4 now must output a per-qualification checklist (`qualificationChecks`) rather than a single self-reported "missing count" — code computes the count and enforces the `fit` label's consistency with it, rather than trusting the model to self-apply its own stated rules.

**Determinism:** testing this also surfaced a third issue — the same query could classify a qualification as "required" on one run and not on the next, purely from GPT-4o's own sampling randomness, unrelated to any of the above. Both classification calls (Stage 1 query understanding, Stage 4 guard) now run with `temperature: 0` and a fixed `seed`, config-driven via `CONFIG.openai.chat.classificationTemperature`/`classificationSeed`. HyDE generation (Stage 1b) is deliberately left untouched — it's generative-by-design, not a classification call, so determinism isn't the goal there. Note this is best-effort reproducibility, not a hard guarantee.

**Live example — typo correction, filler stripping, compound splitting, ambiguity flagging, and determinism, all together:**

```bash
npm run search "Find me an AI RAG LLM enginner who kows psytorch, TS, and sematic serch"
```

```json
[Stage 1] Done: {
  "raw": "Find me an AI RAG LLM enginner who kows psytorch, TS, and sematic serch",
  "title": "AI RAG LLM engineer",
  "qualifications": ["RAG", "LLM", "PyTorch", "TS", "semantic search"],
  "requiredQualifications": ["RAG", "LLM", "PyTorch", "TS", "semantic search"],
  "ambiguousQualifications": ["TS"],
  "queryText": "AI RAG LLM engineer, with expertise in RAG, LLM, PyTorch, TS, semantic search"
}
```

Note everything this one query exercises: `"psytorch"` and `"sematic serch"` are corrected to `"PyTorch"`/`"semantic search"` (typo fix, not a guess); `"AI RAG LLM"` in the title is split into separate `"RAG"`/`"LLM"` entries (compound splitting), not left as one ungrammatical string; `"knows"` never appears in the output (filler stripped); `"TS"` is left exactly as written and explicitly flagged in `ambiguousQualifications` rather than silently guessed as TypeScript (ambiguity handling). Re-ran this exact query 3 times back-to-back — identical output every time, character for character, confirming the `temperature: 0` + fixed `seed` fix actually holds in practice, not just in theory.

**What this does not do: actually resolve ambiguity.** `"TS"` is still `"TS"` — we've made not-knowing consistent and honest across both stages, not resolved. Two paths forward if that's not enough:

- **Interactive clarification** — surface flagged-ambiguous terms back to the recruiter before running the rest of the pipeline (*"did you mean TypeScript or Time Series?"*), matching the real Hiring Assistant's "interactive UX" philosophy of resolving ambiguity with the human before large-scale execution. Requires an actual clarification loop — `scripts/search.ts` is currently one-shot, and `evaluate.ts` runs unattended with no human in the loop.
- **A skill ontology/knowledge graph** — instead of a static two-way table (known-safe vs. known-expansion vs. ambiguous), a structured ontology could use surrounding query context (title, other required qualifications) to disambiguate with actual confidence — e.g. inferring "TS" likely means TypeScript when co-occurring with "React", or Time Series when co-occurring with "forecasting". This is the next PoC direction if the deterministic-table approach proves too coarse in practice.

---

## Evaluation Methodology

### The Corpus & Ground Truth

To move away from anecdotal validation, I built a controlled testing ecosystem:

* **724 synthetic candidate profiles** distributed across 10 job functions and 9 seniority levels.
* **20 complex recruiter search queries**.
* To establish ground truth, the **OpenAI Batch API** evaluated every single query against every single profile (20 × 724 = 14,480 deterministic qualification judgments), classifying pairings strictly as `QUALIFIED` or `NOT_QUALIFIED`.

### Metrics Tracked

* **R@50 (Recall @ 50):** Measures the percentage of total qualified candidates captured in the top 50 results. This represents our pipeline's ceiling; if a candidate is missed here, a downstream reranker cannot save them.
* **NDCG@10 (Normalized Discounted Cumulative Gain):** Evaluates ranking quality, ensuring the absolute best matches are heavily penalized if they do not appear at the very top of the recruiter's feed.

---

## Example Sourcing Walkthrough

```bash
npm run search "Senior ML engineer, Python, NLP background, Bay Area"

```

```text
Loading index...
Running pipeline...

[Stage 1] Query understanding...
[Stage 1] Done: {
  "title": "ML engineer",
  "seniority": "senior",
  "location": { "region": "San Francisco Bay Area", "country": "United States" },
  "requiredQualifications": ["Python", "NLP background"],
  "queryText": "senior ML engineer, with expertise in Python, NLP background, based in San Francisco Bay Area"
}

================================================================================

#1  Priya Nair — Senior Machine Learning Engineer
     senior | Oakland, United States | 7 yrs exp
     Skills: Python, NLP, PyTorch, Transformers, spaCy, Hugging Face, MLflow, SQL, Docker, AWS
     Distinguishing: Hugging Face (7.4x), MLflow (7.4x), spaCy (7.4x)
     L1 score: 0.8061  |  L2 score: 0.9817  |  fit assessment: excellent

     The candidate meets all requirements: they have seniority as 'Senior',
     their location in Oakland is within the 'San Francisco Bay Area', they
     have the required skills in Python and NLP.
--------------------------------------------------------------------------------

#2  Aisha Kamara — Senior ML Engineer
     senior | Berkeley, United States | 6 yrs exp
     Skills: Python, NLP, Deep Learning, PyTorch, BERT, LLMs, RAG, Vector Databases, FastAPI, GCP
     Distinguishing: RAG (7.4x), Vector Databases (7.4x), GCP (3.7x)
     L1 score: 0.7654  |  L2 score: 0.9734  |  fit assessment: excellent

     The candidate is a Senior ML Engineer, located in the Berkeley area which
     is part of the San Francisco Bay Area, and meets all required
     qualifications with expertise in Python and NLP.
```

`Distinguishing` (from the [term-frequency stage](#distinguishing-skills-statistical-explainability-no-llm) above) is computed and rendered alongside `guardExplanation` but comes from a completely separate, LLM-free code path — see the sequence diagram for exactly where it branches off.

---

## Future Roadmap

| Current Implementation | Target Scalability |
| --- | --- |
| General-purpose embeddings | Fine-tuned dual-tower retrieval model |
| Generic cross-encoder | Recruiter behavioral-specific ranking model |
| HyDE generation latency cost | Learned query encoder / distillation |
| Synthetic profiles | Anonymized production-grade profiles |
| Static acronym table + ambiguity flagging | Skill ontology / knowledge graph for context-aware disambiguation, or interactive recruiter clarification (see [Acronym Handling](#acronym-handling-a-documented-limitation-not-a-fix)) |

---

## Quickstart

The profiles, FAISS index snapshots, and evaluation labels are pre-compiled and committed to the repository. Only API keys are required to run an instant demo.

### 1. Installation

```bash
npm install
cp .env.example .env

```

Configure your environment variables:

```bash
OPENAI_API_KEY=your_key_here
COHERE_API_KEY=your_key_here

```

### 2. Run an Active Search Pipeline

```bash
npm run search "Senior ML engineer, Python, NLP background, Bay Area"

```

### 3. Run the Evaluation Harness

```bash
# Full evaluation
npm run evaluate

# Rapid smoke test
npm run evaluate -- --limit=5

```

---

## Project Structure

```text
src/
  pipeline/         # the 4 pipeline stages + orchestrator + matchStats.ts (no-LLM explainability)
  embeddings/       # OpenAI embedding client + disk cache
  index/            # FAISS exact-search (IndexFlatL2) index build/load/search
  data/             # synthetic profile generator
  config.ts         # all tuning parameters and model choices
  types.ts          # TypeScript interfaces
  __tests__/        # unit tests, including matchStats.test.ts
scripts/
  generate.ts       # one-time: generate 724 synthetic profiles
  index.ts          # one-time: build embeddings + FAISS index
  search.ts         # interactive search demo
  evaluate.ts       # full eval harness (Matryoshka comparison)
  evalGenerate.ts   # generate ground-truth labels via Batch API
data/
  profiles.json             # 724 synthetic profiles (committed)
  embeddings_cache.json     # OpenAI embedding vectors for all profiles — 3072-dim floats per profile,
                            # cached to disk so re-runs don't re-call the API (67MB, not committed)
  index.faiss               # serialized FAISS exact-search (IndexFlatL2) index built from the 512-dim slice of each
                            # embedding — what gets searched at query time (committed)
  profile_id_map.json       # maps FAISS index positions → profile IDs (committed)
  eval_queries_raw.json     # the 20 hardcoded recruiter query strings (committed)
  eval_queries.json         # ground-truth relevance labels: for each query, which profile IDs are
                            # qualified — generated by asking an LLM to judge all 724 profiles
                            # against each query via OpenAI Batch API (committed)
  eval_results.json         # output from the last evaluate run — per-query NDCG@10, R@10, R@50
```
