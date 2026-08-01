# TREC RAG 2026 — Work Log

Running record of what was built, what was measured, and why each decision was
made, across **all approaches**. Kept in enough detail to write the notebook
paper from, including the numbers behind every claim. **Append as work happens;
do not rewrite history.**

- **Team:** `iris-mst` (Missouri S&T) · **Deadline:** Aug 8, 2026
- **Corpus:** ClimbMix (`climbmix-400b`) via the Pyserini REST API
- **Tasks entered:** Retrieval (`R`) and RAG

## 0. Approaches, branches, and where their outputs live

Outputs are kept in separate directories so runs never overwrite each other and
any result can be traced back to the approach that produced it.

| Approach | Branch | Output dir | `run_id` | Status |
|---|---|---|---|---|
| Brute-force cascade | `nahid/bruteforce-approach` | `output_baseline/` | `bruteforce-cascade` | **Complete, validated, banked** (§1) |
| Query decomposition (step 1) | `nahid/bruteforce-approach` | `output_decomp/` | — (preprocessing only) | **Step 1 done** (§6) |
| QDER | `nahid/qder-approach` | `output_qder/` | `qder-cascade` | **Stage is currently inert** (§5.6) |

Banked immutable snapshots (both submission files + config + log + checksums)
live outside the repo at
`/media/12TB/shared/datasets/trec/trec-rag-2026/submissions/`, prefixed
`CURRENT_` / `SUPERSEDED_`.

**Rule:** give every approach its own `--out-dir` and its own `--run-id`. The
`run_id` is what identifies a run to the organizers, and two runs sharing one is
how the wrong file gets submitted.

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

### 5.6 QDER: the stage is currently inert (branch `nahid/qder-approach`)

**Measured, and this is the headline result for this branch: turning QDER off
changes nothing.** Final rankings are byte-identical with QDER enabled or
disabled, on 5/5 topics. Same for LTR, and for both together:

| variant | identical to as-configured |
|---|---|
| QDER off | 5/5 topics |
| LTR off | 5/5 topics |
| both off | 5/5 topics |

**Why.** `configs/pipeline.gpu.yaml` sets `qder.top_k: 500` while `hybrid.top_k`
is also 500 — so QDER reorders 500 candidates and discards none. The
cross-encoder then scores every one of those 500 and re-sorts by its own score,
which makes its input order irrelevant. A reranker that neither truncates nor
survives into the final ordering cannot affect the output. This is exactly the
structure that made LTR a no-op (§5.4).

The config comment is candid about the cause — *"reorder the whole pool; don't
discard on an untrained scorer"* — and that caution is reasonable in itself. But
its consequence is that the stage this branch exists to evaluate is not being
exercised at all.

Compounding it: `backend: hashing` with `model_path: null`, i.e. an **untrained**
bilinear scorer over hashed features, and `use_entities: false`, so the
entity channel — the "E" in QDER — is off. A trained LTR model
(`models/ltr_lambdamart.txt`, 4000 rows / 22 groups) now exists, but under this
topology it too changes zero bytes of either submission file.

**What has to be true for QDER to contribute** (all three, not any one):

1. **Train it.** `scripts/train_qder.py` + `scripts/prepare_qder_data.py` exist;
   no checkpoint has been produced yet.
2. **Point at the checkpoint** — `backend: transformer`, `model_path: <ckpt>`.
3. **Let it prune.** `qder.top_k` must be < the pool it receives (e.g. 500 -> 150),
   or its score must be fused into the final ordering rather than overwritten by
   the cross-encoder.

Note the tension in (3): pruning on an *untrained* scorer is worse than not
pruning, so steps 1 and 3 have to land together. Until then, any "QDER run" is
really the brute-force cascade under a different `run_id`.

**Verification method** (re-run after any change): build the pipeline twice with
`qder.enabled` true/false and compare `retrieve()` docid lists per topic. If they
match, the stage is still inert.

### 5.6b QDER training: NO-GO, and why

Audited the training path before spending GPU time. Four findings, each verified
independently rather than taken from the audit.

**1. A trained checkpoint cannot improve the ranking — it can only delete.**
`reranker.py:252,258` writes `qder_score` / `qder_hybrid_score` into
`Candidate.features`, and `grep -rn "qder_score" src/ scripts/ configs/` finds
**no consumer anywhere** (only `tests/test_qder.py:145`). QDER's output feeds
straight into the cross-encoder, which scores every candidate it receives and
re-sorts by its own score. So QDER's *ordering* is discarded; the only thing
that survives is *which candidates it removed*. Training it buys the right to
override `ms-marco-MiniLM-L-6-v2` — distilled from ~500k MS MARCO queries —
with a model fit on 22 topics, and truncation is monotone: it can only lose
recall. This is the same argument already written into the config for ColBERT
(`pipeline.gpu.yaml:70-79`), minus the timing rationale.

**2. The BM25 score channel is mathematically inert in our configuration.**
`third_party/qder_training/qder_model.py:219-221` multiplies each pooled text
feature by `doc_score`, and `:316` then applies `LayerNorm` over the
concatenation. LayerNorm is scale-invariant, so a uniform scaling of everything
it normalises is erased. Verified numerically:

| | doc_score 0.5 | doc_score 7.0 |
|---|---|---|
| `use_entities: false` (our config) | max diff **4e-05** | **1e-05** |
| `use_entities: true` | max diff 1.08 | 1.87 |

With the entity channel off — which is our config, since there is no
WAT/Wikipedia2Vec linker for ClimbMix — *every* concatenated feature is scaled
by the same scalar, so `doc_score` contributes nothing. `prepare_qder_data.py`
carefully min-max-normalises that field to match serve time; it is wasted work.
Worse, `LayerNorm(0) == layer_norm.bias` exactly, so any candidate whose
normalised `doc_score` is 0 (the weakest-BM25 candidate of every topic) produces
a **constant, text-independent** vector — and backfilled positives are hard-coded
to 0. Those rows teach a positive label on an input carrying no information.

The finding is configuration-scoped: with entities enabled the scaling is
relative and does matter. That is not a path available here.

**3. The validation metric optimises the discarded quantity.**
`third_party/qder_training/train.py:174` hard-codes `map`, and `train_qder.py`
exposes no `--metric`. MAP scores QDER's ordering — thrown away by the
cross-encoder. The only quantity the pipeline consumes is `recall@qder.top_k`.

**4. It would have crashed anyway.** `pytrec_eval` is absent from *both* conda
envs (verified), and `train_qder.py:175` uses `subprocess.run(..., check=True)`
— so it raises **after** step 1 has already spent 22 fresh Pyserini searches.
And **0 of the 22 dev topics are in the response cache** (verified by recomputing
the cache keys: the cache holds exactly the 119 test topics). Those are the same
uncached `hits=1000` searches whose 429 storm killed `train_ltr.py`.

**Decision: do not train QDER.** The stage cannot help in this topology, and the
model's score channel is inert in this configuration.

### 5.6c Fixing the no-op: feature-level rank fusion

The root cause behind both the QDER and LTR no-ops was structural, not a bug in
either stage: **a cascade keeps only its last scorer's opinion.** Each stage
re-sorted a pool that the next stage re-sorted from scratch, so an earlier
stage's ordering was always discarded and only its *truncation* survived.

Added `src/trecrag/rerank/fusion.py`: a weighted RRF over the per-stage scores
already accumulated in `Candidate.features`, running after the cross-encoder.
The cross-encoder now scores the pool without truncating (`top_k: 500`) and
fusion selects the final 50, so no candidate is dropped before its earlier
scores can count.

**Fusion is by reciprocal rank, not by score.** Cross-encoder logits were all
negative in a real run (max −1.84) while an untrained QDER's scale is arbitrary;
min-max normalising them would let one stage's spread dominate for reasons
unrelated to its accuracy. RRF needs only each signal's ordering, which is the
part worth trusting.

Verified on 5 topics, against the current CE-only ranking:

| config | rankings differ |
|---|---|
| `qder_score: 0.0`, QDER enabled | 0/5 |
| `qder_score: 0.0`, QDER disabled | 0/5 |
| `qder_score: 0.5` | **5/5** |
| `qder_score: 1.0` | **5/5** |

Two things this establishes. A weight of 0 is **exactly** inert — the shipped
config reproduces the validated cascade byte for byte, so this is safe to have
in place before anything is trained. And a non-zero weight makes QDER genuinely
live: it is now a dial, not a dead stage.

**The weight ships at 0.0 deliberately.** The scorer is still untrained
(`backend: hashing`, `model_path: null`), and giving noise a vote would degrade
a validated cascade. What changed is that the question is now answerable by
turning a dial rather than blocked by the topology — and that a trained QDER
would be able to *contribute* rather than only prune (§5.6b finding 1).

Caveat worth stating plainly: with no dev-set evaluation wired up, there is
still no way to choose a non-zero weight on evidence. Fusion removes the
structural blocker; it does not remove the measurement gap.

### 5.7 Grounding

Of 1,135 checkable sentences: 309 verbatim-match a cited doc, 33 match an
uncited-but-present doc, the rest are paraphrase. **The failure mode is
misattribution within the retrieved set, not fabrication.**

---

## 6. Next: query decomposition

The narratives are composite — situational metadata plus 2–3 sub-questions.
Decomposition addresses both open problems at once: it is the only lever on
recall under a sparse-only index (§3.1), and per-sub-question generation is the
structural fix for coverage (§5.2).

- **Step 1 — DONE** (`scripts/decompose/`, spec in
  `plans/decompose/preprocessing_step_1.md`). vLLM offline batch, XML tags with a
  deterministic parser, one fallback retry, independent pre-write gate, flagged
  file so no query can silently vanish.

  **Run:** Qwen2.5-72B-Instruct-AWQ, tensor_parallel_size=2, temperature 0.
  Output banked at `datasets/trec/trec-rag-2026/decompositions/step1_20260801_0841/`.

  | | |
  |---|---|
  | parsed on first attempt | **119/119** (0 fallbacks, 0 flagged) |
  | `comp_query` preserved exactly | yes, all 119 |
  | sub-questions per narrative | 1:16, 2:67, 3:18, 4:11, 5:6, 8:1 — mean **2.40** |
  | narratives with no framing (empty metadata) | 16/119 |
  | content-word recall of the original | mean **0.991**, median 1.000, min 0.823 |

  Verbatim fidelity measured 67.5% exact / 15.0% near-verbatim / 17.5%
  "materially rewritten", but inspection shows the last bucket is almost entirely
  the prompt's own allowance — merging a sentence the asker split ("Write an
  investment plan… Specifically, the timespan…") or fixing a dangling connective
  so the question stands alone. The strict test required the whole normalised
  string to appear contiguously, which any sentence join fails. Content-word
  recall of 0.991 confirms nothing substantive is dropped; the missing tokens are
  function words (`have`, `any`, `even`, `furthermore`).

  Spot-check: `rag2026-0` reproduces the worked decomposition in Debayan's spec
  exactly — the two framing sentences as metadata, and the "what should we
  prioritize" and "how should we think about tradeoffs" questions as the two
  sub-questions.

  *Note for the paper:* mean 2.40 sub-questions per narrative is the direct
  justification for per-facet generation (§5.2) — the baseline emits a
  near-constant 9.5 sentences regardless of how many questions were asked.
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

**QDER — current focus**

- [x] ~~Train a QDER checkpoint~~ — **NO-GO**, see §5.6b. The stage cannot improve
      the ranking in this topology (its scores are discarded; it can only delete
      candidates ahead of a far stronger cross-encoder), and the model's BM25
      channel is mathematically inert with the entity channel off.
- [x] ~~Make QDER's score survive into the final ordering~~ — **done** (§5.6c).
      Feature-level RRF fusion after the cross-encoder. Verified: weight 0 is
      exactly inert, weight 0.5 changes 5/5 topics. QDER is now a dial.
- [ ] **Choose a `qder_score` weight on evidence.** Ships at 0.0 because the
      scorer is untrained and there is no dev signal to justify anything else.
      Needs either a trained checkpoint or a dev-set metric — see below.
- [ ] Consider entity linking for ClimbMix, which would both restore the "E" and
      un-break the `doc_score` channel that LayerNorm currently erases (§5.6b).
- [ ] Independent of QDER: `pip install pytrec-eval-terrier` in `trec-rag-2026`
      if any vendored trainer is ever run (absent in both envs today).

**Carried over**

- [ ] Send the ClimbMix bug report (drafted; needs Shubham's review)
- [ ] Build steps 2 and 3 of the decomposition pipeline
- [ ] Consider tuning BM25 `k1`/`b` — defaults (0.9/0.4) are MS MARCO-tuned for
      short passages; ClimbMix documents average 66k chars, so `b` likely
      under-penalises length. Verified tunable; ~10 API calls to test.
- [ ] No dev-set signal is wired up (RAGDoll/nuggets). Every quality judgement so
      far is inspection, not measurement — the largest methodological gap. This
      bites QDER hardest: without it, "did QDER help?" stays unanswerable even
      after the stage is made live.

---

## 10. Changelog

Newest first. One line per substantive change, so the paper can reconstruct what
was true when.

- **Aug 1 — QDER branch.** Added feature-level rank fusion (§5.6c), removing the
  structural reason QDER and LTR were no-ops: earlier stages' scores now survive
  into the final ordering instead of being overwritten by the cross-encoder.
  Ships with `qder_score` weight 0.0 (exactly inert, cascade unchanged) because
  the scorer is untrained.
- **Aug 1 — QDER branch.** Audited the training path: **NO-GO** (§5.6b). QDER's
  scores have no consumer, so training buys only the right to prune ahead of a
  stronger zero-shot cross-encoder; the model's BM25 channel is erased by a
  LayerNorm whenever the entity channel is off; the trainer's metric optimises
  the discarded ordering; and it would have crashed on a missing `pytrec_eval`
  after spending 22 uncached Pyserini searches. Also measured QDER to be a
  **no-op** in the shipped config (§5.6). Restored the missing `output/checkpoint.py` that left
  `run_pipeline.py` unimportable on this branch, and anchored the `.gitignore`
  rule that had made that module untrackable. Fixed a 429 `RetryError` escaping
  the retry loop and killing `train_ltr.py` mid-run. Trained
  `models/ltr_lambdamart.txt` (4000 rows / 22 groups) — inert under the current
  topology.
- **Aug 1 — Decomposition.** Step-1 run over all 119 topics: 119 parsed on first
  attempt, 0 fallbacks, 0 flagged. Median 2 sub-questions/topic (range 1–8).
- **Aug 1 — Brute-force.** Removed LTR (byte-identical output); narrowed listwise
  reranking from 4 LLM calls to 1; kept cross-encoder width after measuring that
  capping its input at 100 would cost 67% of the submitted top-50 (§5.4).
- **Aug 1 — Brute-force.** Baseline banked as the submission candidate: 119/119,
  0 validation violations, all three prose defects eliminated (§5.1).
- **Jul 31 — Brute-force.** Passage-selection ordering bug: rerankers were handed
  the lower-scoring window in 52% of documents; fixed to 1% (§4.2).
- **Jul 29 — Infrastructure.** First full run. ClimbMix API response-stream bug
  isolated to specific documents (§4.1); `.env.local` found untracked-but-not-
  ignored, i.e. one `git add -A` from leaking the API token (§4.6).
