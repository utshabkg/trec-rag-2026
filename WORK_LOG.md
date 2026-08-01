# TREC RAG 2026 — Work Log

Running record of what was built, what was measured, and why each decision was
made. Kept in enough detail to write the notebook paper from, including the
numbers behind every claim. **Append as work happens; do not rewrite history.**

- **Team:** `iris-mst` (Missouri S&T) · **Deadline:** Aug 8, 2026
- **Corpus:** ClimbMix (`climbmix-400b`) via the Pyserini REST API
- **Tasks entered:** Retrieval (`R`) and RAG

---

## 1. Current state

| | |
|---|---|
| Submission candidate | `submissions/CURRENT_bruteforce-cascade_20260801_0709/` |
| Validation | 0 violations on both files |
| Coverage | 119/119 narratives, 0 failures |
| Tests | 131 passing |
| Branch | `nahid/bruteforce-approach` |

**Pipeline (baseline run):**

```
BM25(1000) ─┐
            ├─ RRF(500) ─ LTR(500) ─ [ColBERT off] ─ CE(50) ─ LLM listwise(50) ─ generator(top-10)
e5 rescore ─┘
                    ▲
        query-focused passage selection
```

Retrieval depth 50/narrative, except `rag2026-70` at 10 (see §4.1).

---

## 2. Environment

| Component | Choice | Why |
|---|---|---|
| Retrieval | Pyserini REST, `climbmix-400b` | Only route to the corpus; see §3.1 |
| Biencoder | `intfloat/e5-base-v2` (GPU) | Cached locally; needs `query:`/`passage:` prefixes |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` (GPU) | Cached; 500 candidates in 0.24s once batched |
| Listwise + generation | `llama3.3:70b-instruct-q4_K_M` via Ollama | Local, free, non-reasoning (see §4.4) |
| Decomposition (step 1) | `Qwen2.5-72B-Instruct-AWQ` via vLLM | AWQ 4-bit, has a chat template |

**Two conda envs, deliberately separate.** `trec-rag-2026` runs torch 2.11.0+cu128
with sentence-transformers; `trec-rag-2026-vllm` runs torch 2.9.0+cu128 with vLLM
0.13.0. vLLM pins torch, so installing it into the main env would downgrade torch
underneath a validated submission pipeline.

Hardware: 2× RTX 6000 Ada (48 GB each), shared with other users.

---

## 3. Findings about the task setup

### 3.1 The index is sparse-only — this bounds the whole system

The Pyserini API exposes exactly two endpoints (`/search`, `/doc/{docid}`), zero
mentions of dense/FAISS/vector/embedding in its OpenAPI spec, self-describes as
"Lucene indexes via Pyserini", and its tunable parameters are `k1` and `b` —
BM25 hyperparameters, documented as returning 400 "on non-sparse indexes".

**Consequence:** there is no dense index to search, so the biencoder can only
*re-score* the BM25 pool, never add to it. BM25 recall at depth 1000 is a hard
ceiling on the entire cascade — every later stage only reorders what BM25 found.
This is why query decomposition (§6) is the highest-value direction: more,
better-targeted queries are the only lever on recall.

Building our own index was considered and rejected: 6,543 parquet shards / 600 GB,
a strict docid contract (`shard_01789_3390` = filename + row index, any reordering
invalidates every ID), and the official docid validator itself requires the API
token. Not reachable inside the deadline.

### 3.2 ClimbMix documents are whole documents, not passages

Measured over one real 1000-candidate response: **mean 66,685 chars, median
42,948, max 834,116 — and not one candidate under 2,000 chars.**

Every stage after BM25 truncates its input to a prefix, so before passage
selection the entire cascade was reranking and citing each document's title page.
Measured over 200 real candidates for `rag2026-0`:

- the leading window was the best window in only **10/200** documents
- median best window began at char offset **15,900**
- selecting it raised query-token overlap **+32%**
- symptom: **all 50** surviving cross-encoder scores were negative (max −1.84)

`rag-task.md` explicitly permits the fix: *"Apply its own chunking or passage
selection internally."*

### 3.3 Unverified spec details that were wrong

Shubham's planning doc (flagged in its own text as a secondhand snapshot) said
answers must be **< 400 words**. The official `rag-task.md` says **1024**. The
config was already right; the doc was stale. Topics file verified byte-identical
to the official release (SHA-256 `72dc2fd3…`).

---

## 4. Bugs found and fixed

### 4.1 ClimbMix API: individual documents kill the response stream

`GET /v1/climbmix-400b/doc/shard_02798_23484` returns HTTP 200 then drops the
connection mid-body (`ChunkedEncodingError`). **One request, one document** — no
search, no concurrency, no volume.

Ruled out: response size (`rag2026-73` returns 47.9 MB fine), server load (a fresh
request succeeded while the pipeline's failed), stale connection pools (a restarted
process failed on its first request), and the parse path (`parse=false` fails
identically). Thresholds were byte-identical across two sessions hours apart under
very different load (2s vs 22s response times). A 4.5 MB request fails while a
27 MB one succeeds — volume-based shedding cannot produce that ordering.

**This is distinct from the 429/rate-limit issue discussed in the track Slack.**
That is load-shedding on high QPS or large top-k; this is a single deterministic
document. `max_doc_length=2000` makes both affected narratives return all 1000
candidates at 2.1 MB, which isolates the fault to emitting one document's *full*
contents.

Affected: `rag2026-70` (fails at hits ≥14, offending doc at rank 14) and
`rag2026-72` (fails at hits ≥800). **Workaround:** `search_adaptive` walks the
depth down until the offending document falls outside the window. Reported to the
organizers (`plans/climbmix_api_bug_report.md`).

*Paper note:* `rag2026-70` is retrieved at depth 10 and `rag2026-72` at 500,
versus 1000 for the other 117. This asymmetry should be stated.

### 4.2 Passage selection returned windows in document order

With `window_chars: 2000` and `windows_per_doc: 2` joined in document order, but
every consumer truncating at 2000 chars, the rerankers only ever saw the *first*
window — **the lower-scoring one in 52% of two-window documents** (measured over
299 real candidates). Selecting two windows was worse than selecting one, and
invisibly so. Fixed by ordering on score: **52% → 1%**.

### 4.3 Listwise reranker parsed stray integers as ranks

`_INT_RE = r"\d+"` matched every bare integer, and missing indices were
back-filled, so the result was always a *valid* permutation — nothing errored.
`"1. [7]"` ranked passage 1 first, not 7. It had displaced 6 of the
cross-encoder's top 10 in a real checkpoint. Fixed to bracketed indices with a
coverage guard.

### 4.4 Reasoning tokens corrupt both parsers

The listwise parser reads every integer; the generator takes the first bracketed
span as JSON. A `<think>` block supplies plenty of both. Now stripped. `qwen3.5:35b`
returned empty through Ollama's OpenAI route — `llama3.3:70b` (non-reasoning) is
used instead.

### 4.5 Three silent paths to a zero-scoring record

1. `postprocess` used `break` on the word budget: one oversized "sentence"
   discarded every valid sentence after it → no sentences → no references → zero.
2. `LLMGenerator` only fell back when JSON failed to *parse*; well-formed JSON with
   rejected citations produced an empty answer and no fallback.
3. `validate.py` reported 0 violations for `{"references": [], "answer": []}`.

### 4.6 Operational

- **`.env.local` was not gitignored** despite the README saying it was (`.env`
  does not match `.env.local`). A `git add -A` would have pushed the API token.
- **`--limit` overwrote the official filenames** — a smoke test would truncate a
  finished submission and exit 0. Now writes `*.partial`.
- **Secret scanner false-positived** on `"risk-perception commons"` (`ri|sk-…`),
  failing a valid submission.
- **Transport errors after HTTP 200** are invisible to status-code retry; added
  application-level retry.
- **Shared Ollama contention** caused a 300s timeout and one 578s topic; timeout
  raised to 900s with retry.

---

## 5. Results

### 5.1 Baseline v1 → v2 (after §4.2 and prompt hardening)

| metric | v1 (superseded) | v2 (current) |
|---|---|---|
| rerankers shown lower-scoring window | 52% | **1%** |
| `"Passage [N]"` leaking into prose | 43 sentences | **0** |
| answers opening by restating the question | 21 narratives | **0** |
| "evidence does not provide…" punts | 20 narratives | **0** |
| mean answer words | 348 | **373** |

Both runs: 119/119, 0 validation violations.

### 5.2 The open problem: coverage

Answers average **373 words against a 1024-word budget**, and sentence count is
near-constant (~9.5) **regardless of whether a narrative asks 3 things or 18**
(correlation −0.065). The generator prompt already demands 614–921 words and
already says breadth matters more than depth.

The missing material is in documents the answers *already cite*:

- `rag2026-102`: "Hamburg" in 5 of 9 cited refs, "Tokyo"/"firestorm"/"Gomorrah" in
  one each — **none in the answer**, which spends 40% of its words on Hiroshima.
- `rag2026-0`: the CNO explicitly wants "recruitment and **retention**";
  "retention" appears in 4 of 7 cited refs and never in the answer.

So this is neither a retrieval gap nor a prompting gap. Prompting for length has
been tried twice and plateaued. **The fix is structural: one generation pass per
sub-question, so length is emitted by construction rather than requested.**

### 5.3 First-stage depth: 1000 is load-bearing, not padding

Proposal raised (Debayan): cut BM25 depth 1000 → 500, on the grounds that
downstream stages only consume a small top-k anyway. Measured over all 119
baseline checkpoints, using the `bm25_rank` feature each candidate carries:

| BM25 origin rank | share of submitted top-50 | share of generator top-10 |
|---|---|---|
| 1–100 | 26.3% | 30.3% |
| 101–250 | 24.2% | 21.4% |
| 251–500 | 23.8% | 23.8% |
| **501–1000** | **25.7%** | **24.5%** |

Cutting to 500 would discard **25.7% of submitted documents and 24.5% of the
evidence the generator actually cites**, affecting **112 of 119 narratives**.

The distribution is near-uniform, which is the real finding: **BM25 rank barely
predicts final relevance.** Spearman correlation with the final submitted order
is **+0.032 for BM25 rank** versus **+0.209 for the e5 biencoder** — the
biencoder is ~6.5× the predictor BM25 is, and BM25's own ordering is close to
noise with respect to the end result. This is consistent with §3.2: BM25 scores
whole 66k-char documents, so its ranking is dominated by length and term
frequency across irrelevant material.

The biencoder is also demonstrably doing work, not producing "incremental gain":
**BM25 and e5 agree on only 19% of their top-10**, so RRF is fusing two largely
disjoint views rather than smoothing one.

**Decision: keep depth 1000.** Under a sparse-only index the first-stage pool is
the recall ceiling (§3.1), and a weak first-stage ranker is precisely the case
where a wide pool matters most. Note also that BM25 depth does not affect run
file size — the submitted run is 50 docs/narrative (375 KB) regardless.

Costs of the change, for the record: 500 would halve API load and the response
cache (4.9 GB → ~2.5 GB) and would dodge one instance of the §4.1 API bug
(`rag2026-72` succeeds at 500). Changing `hits` also invalidates the entire cache
and forces a re-fetch of all 119 narratives. Not worth 25% of the evidence.

*Open question:* the flat distribution weakly suggests depth >1000 could still
help. Untested — response size (~60 MB at 1000) and §4.1 exposure make it
impractical to push much further.

### 5.4 Cascade width: LTR is a no-op, CE width is load-bearing, listwise was over-run

Three proposals were raised together: drop LTR, shrink the cross-encoder's input
(k1), shrink the listwise reranker's input (k2). Measured separately, they do not
have the same answer.

**LTR — remove it. Provably a no-op.** It reranked 500 → 500 (no truncation,
since hybrid emits 500) and the cross-encoder then re-sorted every candidate by
its own score, discarding LTR's ordering wholesale. Verified end to end: **5/5
topics produce byte-identical final rankings with LTR enabled or disabled.** Now
`enabled: false`. This also retires the "train the LTR model" item — under this
topology a trained ranker would change nothing downstream.

**Cross-encoder input — keep it wide.** Measured over 12 topics, tracking where
each CE top-50 document sat in the RRF ordering it was handed:

| RRF rank of doc | share of CE top-50 | share of top-10 evidence |
|---|---|---|
| 1–50 | 20.3% | 26.7% |
| 51–100 | 12.7% | 12.5% |
| 101–200 | 21.2% | 21.7% |
| 201–350 | 24.2% | 20.8% |
| 351–500 | 21.7% | 18.3% |

Capping the CE's input would cost:

| cap | lost from top-50 | lost from evidence |
|---|---|---|
| 100 | **67.0%** | 60.8% |
| 200 | **45.8%** | 39.2% |
| 350 | **21.7%** | 18.3% |

Again near-uniform, and consistent with §5.3: **first-stage rank barely predicts
final relevance**, so the cross-encoder — the only stage with real
discrimination — finds its winners spread evenly through the pool. There is no
cost argument either: batched, it scores 500 candidates in **0.24s** (~0.5 min
for all 119 narratives). Width here is nearly free and demonstrably load-bearing.

*Untested and cheap:* the distribution is flat all the way to 500, so `hybrid.top_k`
(currently 500, out of a 1000-document BM25 pool) may itself be the limiting cut.
Raising it to 1000 would cost ~0.5s/topic of CE time.

**Listwise input — genuinely too wide.** This stage is ~85% of runtime, one LLM
call per sliding window. It was reranking all 50 candidates (4 windows) when only
the head becomes the generator's evidence. Added `rerank_depth`, set to 20: the
LLM reranks the top 20 (**1 window**) and the tail is spliced back in
cross-encoder order. **4 LLM calls → 1**, with the run still submitting 50
documents and every ordering invariant intact (all candidates present, ranks
1..N, scores non-increasing across the splice).

Net: the two stages that were removed or narrowed are the ones that could not be
shown to contribute; the one kept wide is the one measurement says is doing the
work.

### 5.5 Grounding

Of 1,135 checkable sentences: 309 verbatim-match a cited doc, 33 match an
uncited-but-present doc, the rest are paraphrase. **The failure mode is
misattribution within the retrieved set, not fabrication.**

---

## 6. Next: query decomposition

The narratives are composite — situational metadata plus 2–3 sub-questions.
Decomposition addresses both open problems at once: it is the only lever on
recall under a sparse-only index (§3.1), and per-sub-question generation is the
structural fix for coverage (§5.2).

- **Step 1 — built and tested** (`scripts/decompose/`, spec in
  `plans/decompose/preprocessing_step_1.md`). vLLM offline batch, XML tags with a
  deterministic parser, one fallback retry, independent pre-write gate, flagged
  file so no query can silently vanish.
- **Step 2 — to build:** retrieve per sub-question, RRF-fuse with the base run.
- **Step 3 — to build:** one generation pass per sub-question, assembled.

Give it a **distinct `run_id`** so both runs can be submitted and told apart.

---

## 7. Instrumentation

`logs/` holds one run log and one `stages_*.jsonl` per run. Each stage records
`n_in`, `n_out`, score distribution, `overlap@10` and `mean_displacement@10`
against its input, which separates:

- **overlap ≈ 1.0, displacement 0** → the stage is a no-op
- **overlap ≈ 0** → the stage overrides everything upstream (intended for a strong
  cross-encoder; a bug for a parser reading noise)
- **near-flat score spread** → the model cannot discriminate, so its ordering is
  close to arbitrary regardless of overlap

This is how to attribute a score change to a stage instead of guessing.

---

## 8. Reproduction

```bash
conda activate trec-rag-2026
python scripts/check_api.py --config configs/pipeline.gpu.yaml      # first-contact check
python scripts/run_pipeline.py \
    --topics trec_rag_2026_test_query_set.tsv \
    --config configs/pipeline.gpu.yaml \
    --team-id iris-mst --run-id bruteforce-cascade \
    --out-dir output_baseline/ --workers 1
python scripts/validate_submission.py \
    --retrieval output_baseline/r_output_trec_rag_2026.tsv \
    --rag output_baseline/rag_output_trec_rag_2026.jsonl \
    --topics trec_rag_2026_test_query_set.tsv
```

Resumable at two layers: per-topic checkpoints and the Pyserini response cache
(~4.9 GB, keyed by index+query+hits). Re-runs cost no API calls. Use
`--workers 1`: the neural stages share one model instance each.

Runtime: ~80 min with GPUs free; 512 min observed under Ollama contention.

---

## 9. Open items

- [ ] Send the ClimbMix bug report (drafted; needs Shubham's review)
- [ ] Run step-1 decomposition (needs GPUs free — a 39 GB AWQ load needs both)
- [ ] Build steps 2 and 3 of the decomposition pipeline
- [ ] Consider tuning BM25 `k1`/`b` — defaults (0.9/0.4) are MS MARCO-tuned for
      short passages; ClimbMix documents average 66k chars, so `b` likely
      under-penalises length. Verified tunable; ~10 API calls to test.
- [ ] No dev-set signal is wired up (RAGDoll/nuggets). Every quality judgement so
      far is inspection, not measurement — the largest methodological gap.
