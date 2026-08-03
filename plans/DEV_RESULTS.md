# Development-set experiments — TREC RAG 2026

**Team:** iris-mst · **Deadline:** Aug 8, 2026 (5 days from Aug 3) · **Status:** a
validated submission is banked; these experiments are about improving it.

This document states what was run, what was being tested, what came out, and what
it does and does not license us to claim.

---

## 0. Direct answers to the questions raised

| Question | Answer |
|---|---|
| Are these for the RAG 2026 submission? | Yes. They are development diagnostics for the system whose test-set run is already banked. |
| Did you train a model? | **No.** Everything is zero-shot, off-the-shelf checkpoints. No fine-tuning, no training data, no learned weights anywhere in the pipeline. The one learnable component (LTR) was never trained and is now disabled — measured as a no-op (§4.1). |
| Did you test on RAG 2025? | Partly, and the distinction matters. The dev topics **are** 22 RAG25 narratives (`rag25-topics-dev.tsv`), but scored against **projected UMBRELA judgments over ClimbMix**, not the original RAG25 qrels over MS MARCO. So it is RAG25 *topics* on the RAG2026 *corpus*, not a RAG25 replication. |
| What's the baseline? | BM25 over ClimbMix via the Pyserini API — the track's own first-stage retriever. Measured for the first time in §3; before that we had no reference point, which was a real gap. |
| How do we know we're doing well? | On retrieval, we now know the cascade beats BM25 by +0.08 to +0.13 nDCG@10. On the **actual RAG task** — answer quality — we do **not** yet know (§6). |
| When is the deadline? | **Aug 8, 2026.** |

---

## 1. What was run

**Data.** 22 RAG25 development narratives, three independent projected-UMBRELA
qrels variants over ClimbMix (26,341 judgments each, grades 0–3). All shipped in
`TREC-RAG/trec-rag-data`.

**System.** Zero-shot cascade, no trained components:

```
BM25(1000) ──┐
             ├─ RRF(500) ─ cross-encoder(500→50) ─ LLM listwise(top-20) ─ generation
e5 rescore ──┘
      ↑ query-focused passage selection
```

Components: `intfloat/e5-base-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`llama3.3:70b-instruct-q4_K_M` via Ollama. Hardware: 2× RTX 6000 Ada, Xeon
w9-3595X, 754 GB RAM.

**Metrics.** nDCG@{10,20,100}, recall@{100,1000}, MAP via `pytrec_eval`. The
evaluator was validated before use: a run ranked by true relevance grade scores
nDCG@10 = 1.0000, shuffled 0.5668, reverse-ranked 0.0030.

**Reproduce:** `scripts/eval/evaluate_retrieval.py`, `scripts/eval/report_latency.py`.

---

## 2. Hypotheses tested

- **H1** — Each cascade stage improves ranking over the stage before it. *(§3)*
- **H2** — Reranking only the top-20 with the LLM, rather than all 50, costs
  ranking quality. *(§4.2)*
- **H3** — The LLM listwise stage contributes at all. *(§4.3)*
- **H4** — LTR contributes. *(§4.1)*
- **H5** — Added stages cost latency, but buy quality; is the tradeoff
  favourable, and where are the marginal returns? *(§5)* — motivated in §5.0:
  quality figures alone cannot distinguish a stage that buys +0.02 nDCG for 0.5 s
  from one that buys +0.02 for 30 s, and the notebook paper needs that
  distinction.

---

## 3. H1: stage ablation against the BM25 baseline

Five arms, identical except for which stages are active, all on the same 22
topics. **Condensed-list nDCG@10** (see §3.1 — this is the correct measure here):

| arm | | codex | ministral | qwen |
|---|---|---|---|---|
| **E0** | BM25 only *(baseline)* | 0.5940 | 0.6840 | 0.6205 |
| **E1** | + passage selection | 0.5940 | 0.6840 | 0.6205 |
| **E2** | + e5 biencoder & RRF | 0.6345 | 0.7336 | 0.6883 |
| **E3** | + cross-encoder | 0.6449 | 0.7938 | 0.7167 |
| **E4** | + LLM listwise *(full system)* | **0.6719** | **0.8136** | **0.7270** |

**Monotonically increasing across all three assessors.** Full system over BM25:
**+0.078 / +0.130 / +0.107** — far above the ~0.04 resolution of this dev set
(§7). H1 supported.

E1 ≡ E0 exactly, which is a **harness sanity check, not a null result**: with no
reranker active, passage selection rewrites document *text* but nothing consumes
it, so BM25's ordering cannot change. That it is bit-identical confirms the
ablation is isolating what it claims to.

### 3.1 Why condensed lists — the finding that nearly produced a wrong conclusion

Scored naively, **BM25 appears to beat the full cascade by a wide margin**
(0.6840 vs 0.4957 on ministral). That result is an artifact and would have been
badly wrong to report.

Judged-rate diagnostic — what fraction of each run's top-10 the assessors
actually saw:

| arm | judged@10 | judged@50 |
|---|---|---|
| E0 BM25 | **100.0%** | 100.0% |
| E2 +e5/RRF | 92.7% | 72.7% |
| E4 full | **56.4%** | 47.6% |

The judgment pool was built from BM25 output (~1,197 judged docs/topic ≈ a
top-1000 pool). Unjudged documents score 0 in nDCG, so **any reranker is
penalised for surfacing documents the pool never contained** — a property of the
pool, not of the system. BM25 scores 100% judged *because the pool is its own
output*.

Removing unjudged documents before scoring (Sakai's condensed lists,
`--condensed`) compares systems only on documents assessors actually saw, and
reverses the ordering. Both views are reported because both are informative: the
condensed numbers say our reranking orders judged documents better; the
judged-rate says our system surfaces many documents outside the pool, whose
relevance is **unknown** — potentially good (new relevant material) or bad
(junk). This dev set cannot distinguish those, and we should not claim otherwise.

---

## 4. H2–H4: individual stage contributions

Paired per-topic tests (same topics across arms, so topic difficulty cancels).

### 4.1 H4 — LTR contributes: **refuted**

LTR reranked 500→500 with no truncation, and the cross-encoder then re-sorted
everything by its own score, discarding LTR's ordering. **5/5 topics produced
byte-identical output with LTR enabled or disabled.** Now disabled. This also
retires "train the LTR model" — under this topology a trained ranker would change
nothing downstream.

### 4.2 H2 — narrowing listwise costs quality: **refuted (good news)**

| assessor | mean paired Δ nDCG@10 | t |
|---|---|---|
| codex | −0.005 | −0.33 |
| ministral | −0.004 | −0.25 |
| qwen | +0.001 | +0.05 |

t-values near zero, sign flips between assessors. Reranking the top-20 instead of
all 50 cut LLM calls 4→1 and ~31% of wall clock **at no measurable cost**.

### 4.3 H3 — listwise contributes: **weak positive, not significant**

| assessor | mean paired Δ vs no-listwise | t |
|---|---|---|
| codex | +0.027 | +1.54 |
| ministral | +0.021 | +0.93 |
| qwen | +0.013 | +0.61 |

All three positive; 40 wins / 26 losses pooled (sign test p = 0.109 — indicative
only, and the three assessors are not independent). Individually below
significance. Kept because the sign is consistent and it now costs ~9 s/narrative
rather than the ~32 s it cost at all-50.

---

## 5. H5: latency, and the cost/quality tradeoff

### 5.0 Why we measure this

A cascade is a sequence of increasingly expensive stages, and the honest question
about any of them is not "does it help?" but "does it help *enough to be worth
its cost?*". Reporting quality alone hides that: a stage that buys +0.02 nDCG@10
for 0.5 s and a stage that buys +0.02 for 30 s are very different engineering
propositions, and a reader cannot tell them apart from a quality table.

So latency is instrumented per stage on the *same runs* as the quality metrics,
which lets us state the tradeoff directly and, for the notebook paper, plot
quality against cost as stages are added. It also makes the system reproducible
in a way a quality number alone is not — the same code on different hardware, or
with a cold cache, is a materially different system.

Measured on: **2× NVIDIA RTX 6000 Ada (49 GB each, driver 575.57.08), Intel Xeon
w9-3595X (120 logical cores), 754 GB RAM, Linux 6.1.0-47.** Single process, one
narrative at a time (`--workers 1`) — the neural stages share one model instance
each, so these are serial per-narrative latencies, not a throughput-tuned
deployment.

### 5.1 Phase A — preprocessing (query decomposition)

| step | s/narrative | notes |
|---|---|---|
| Step 1: decomposition | **0.92** | 119 narratives in 1 m 49 s, batched vLLM (Qwen2.5-72B-AWQ, tp=2) |
| Step 2: per-sub-question retrieval | *not built* | planned |
| Step 3: per-facet generation | *not built* | planned |
| **Phase A total (current)** | **0.92** | + ~4.7 min one-time model load, amortised over the run |

**The most actionable number in this document.** Decomposition costs 0.92 s per
narrative under *batched* vLLM, against 34 s for a single *serial* Ollama
generation call — a ~37× difference in per-narrative cost for comparable LLM
work. The gap is batching, not model size: vLLM sustained 793 input tok/s and
150 output tok/s across a 119-prompt batch. Steps 2 and 3 should therefore be
built on batched vLLM rather than per-narrative calls, or they will dominate
total runtime unnecessarily.

### 5.2 Phase B — retrieval pipeline

| stage | kind | s/narrative | share of Phase B |
|---|---|---|---|
| bm25 | network (Pyserini) | 0.23 cached / **4.23 cold** | 1.4% |
| passage_selection | CPU | 0.96 | 5.8% |
| biencoder (e5) | GPU | 4.96 | 30.1% |
| rrf | CPU | ~0.00 | 0.0% |
| cross_encoder | GPU | 0.45 | 2.7% |
| llm_listwise | LLM (Ollama) | 9.89 | 60.0% |
| **Phase B total** | | **16.49** | 100% |

**The Pyserini API is not the bottleneck** — 4.23 s cold against 9.89 s for
listwise reranking and 34 s for generation. Worth stating explicitly because it
is the intuitive suspect for a system built on a remote index.

### 5.3 Phase C — answer generation

| stage | kind | s/narrative | share |
|---|---|---|---|
| generation | LLM (Ollama, serial) | **34.10** | 100% |

### 5.4 End to end

| phase | s/narrative | share of total |
|---|---|---|
| A — preprocessing | 0.92 | 1.8% |
| B — retrieval pipeline | 16.49 | 32.0% |
| C — generation | 34.10 | **66.2%** |
| **total** | **51.51** | ≈70 narratives/hour |

≈100 min for the 119-narrative test set cached; ≈108 min with live retrieval.

### 5.5 The tradeoff (H5)

Quality against cost as retrieval stages are added:

| arm | cumulative s/narrative | condensed nDCG@10 (ministral) | Δ nDCG per added second |
|---|---|---|---|
| BM25 | ~1.2 | 0.6840 | — |
| + e5/RRF | ~6.2 | 0.7336 | +0.0099 |
| + cross-encoder | ~6.6 | 0.7938 | **+0.1505** |
| + listwise | ~16.5 | 0.8136 | +0.0020 |

**Monotonic in both quality and cost — the tradeoff is favourable overall, so H5
is supported.** But the marginal returns differ by two orders of magnitude: the
cross-encoder is by far the best value (+0.060 nDCG for 0.45 s), while listwise
buys +0.020 for 9.89 s. Both are kept, for different reasons — the cross-encoder
because it is nearly free, listwise because it is consistently positive and the
budget currently allows it.

### 5.6 Caveats for any published latency figure

- The `bm25` row is a **cached local disk read** (0.23 s), not live API latency
  (**4.23 s**, measured against the real service) — an ~18× difference for
  identical code. Any table must say which condition it describes.
- Serial, single-narrative measurement. A batched or concurrent deployment would
  look very different, as §5.1 shows.
- Phase A covers only step 1; steps 2 and 3 are not built, so the current total
  is not the projected final system's cost.

---

## 6. What we still cannot claim

Stated plainly, because these are the gaps that matter:

1. **We have not evaluated the actual task.** TREC RAG 2026's RAG task is judged
   on *answer* quality — pairwise battles and nugget coverage — not ranking. All
   of the above is retrieval. `scripts/eval/evaluate_nuggets.py` is built and
   unit-tested but **has not been run**. This is the most important missing
   experiment.
2. **No external baseline.** We compare our configurations against each other and
   against BM25. The five official baseline runs cannot be scored: they are over
   the 119 *test* narratives, for which no qrels exist. We do not know how we
   compare to other participants.
3. **Projected, not human, judgments.** Model-generated UMBRELA labels, described
   by the organizers as development diagnostics. Not NIST assessments.
4. **n = 22 topics**, resolution ≈0.04 nDCG@10 (§7). Adequate for the ablation
   effects above; not for fine tuning.
5. **Unknown material outside the pool.** ~44% of our top-10 is unjudged (§3.1).
   Whether that is a strength or a weakness is not determinable from this data.
6. **Answer coverage is known to be weak.** Answers average 373 words against a
   1024-word budget, with a near-constant ~9.5 sentences regardless of whether a
   narrative asks 1 question or 8 (correlation −0.065). Coverage is the axis the
   RAG task competes on, and this is the open problem.

---

## 7. Statistical note

An earlier reading of this data used the *unpaired* standard error
(0.198/√22 ≈ 0.042 ⇒ ~0.084 detectable). That is the wrong test: all arms are
scored on the same topics, so topic difficulty cancels and the **paired**
difference SE applies — measured at 0.014–0.023, giving a resolution of ≈0.04
nDCG@10. The dev set is roughly twice as sensitive as first stated.

---

## 8. Artifacts

| what | where |
|---|---|
| Banked submission (test set, validated) | `datasets/.../submissions/CURRENT_bruteforce-cascade_20260801_0709/` |
| Retrieval evaluator | `scripts/eval/evaluate_retrieval.py` |
| Nugget evaluator (built, not yet run) | `scripts/eval/evaluate_nuggets.py` |
| Latency reporter | `scripts/eval/report_latency.py` |
| Per-run stage logs | `logs/stages_*.jsonl` |
| Full engineering history | `WORK_LOG.md` |

Reproduce the headline table:

```bash
python scripts/eval/evaluate_retrieval.py /tmp/E0_bm25.trec /tmp/E4_full.trec \
    --condensed --judged-rate
```
