# Development experiments, TREC RAG 2026

**Deadline:** Aug 8, 2026, which is five days out from Aug 3
**Status:** a validated submission is already banked, so everything here is about trying to beat it.

This is the record of what we ran, what we were testing, what came back, and what we can honestly say off the back of it.

---

## 0. Short answers to the questions Dr. Chatterjee raised

| Question | Answer |
|---|---|
| Are these for the RAG 2026 submission? | Yes. They are development diagnostics for the same system whose test set run is already banked. |
| Did you train a model? | **No.** Everything is zero-shot with off the shelf checkpoints. No fine tuning, no training data, no learned weights anywhere in the pipeline. The one component that could have been trained (LTR) never was, and it is switched off now because we measured it doing nothing (§4.1). |
| Did you test on RAG 2025? | Partly, and the difference matters. The dev topics **are** 22 RAG25 narratives (`rag25-topics-dev.tsv`), but we score them against **projected UMBRELA judgments over ClimbMix**, not the original RAG25 qrels over MS MARCO. So it is RAG25 topics on the RAG2026 corpus, not a RAG25 replication. |
| What's the baseline? | BM25 over ClimbMix through the Pyserini API, which is the track's own first stage retriever. We measured it for the first time in §3. Before that we had no reference point at all, wich was a real gap. |
| How do we know we're doing well? | On retrieval, we now know the cascade beats BM25 by +0.08 to +0.13 nDCG@10. On the **actual RAG task**, meaning answer quality, we still do **not** know (§6). |
| When is the deadline? | **Aug 8, 2026.** |

---

## 1. What we ran

**Data.** 22 development narratives from RAG25. Three independent sets of projected UMBRELA judgments over ClimbMix, 26,341 judgments each, graded 0 to 3. All of it ships in `TREC-RAG/trec-rag-data`.

**System.** A zero-shot cascade with nothing trained inside it:

```
BM25(1000) ──┐
             ├─ RRF(500) ─ cross-encoder(500→50) ─ LLM listwise(top-20) ─ generation
e5 rescore ──┘
      ↑ query-focused passage selection
```

Components: `intfloat/e5-base-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, and `llama3.3:70b-instruct-q4_K_M` served through Ollama. Hardware: 2× RTX 6000 Ada, Xeon w9-3595X, 754 GB RAM.

**Metrics.** nDCG@{10,20,100}, recall@{100,1000} and MAP, all computed with `pytrec_eval`. We sanity checked the evaluator before trusting it. A run sorted by true relevance grade scores nDCG@10 of 1.0000, a shuffled run gets 0.5668, and a reversed run gets 0.0030. It behaves.

**Reproduce with:** `scripts/eval/evaluate_retrieval.py` and `scripts/eval/report_latency.py`.

---

## 2. What we were testing

- **H1** Each stage of the cascade ranks better than the stage before it. *(§3)*
- **H2** Reranking only the top 20 with the LLM, instead of all 50, costs us ranking quality. *(§4.2)*
- **H3** The LLM listwise stage does anything at all. *(§4.3)*
- **H4** LTR does anything at all. *(§4.1)*
- **H5** The extra stages cost time but buy quality. Is that trade worth it, and where do the returns dry up? *(§5)*

H5 is on the list because quality numbers on their own hide cost. A stage that buys +0.02 nDCG for half a second and a stage that buys +0.02 for thirty seconds look identical in a quality table, and the notebook paper needs to tell them apart.

---

## 3. H1: every stage against the BM25 baseline

Five arms, identical apart from which stages are switched on, all run over the same 22 topics. The numbers below are condensed list nDCG@10, and §3.1 explains why.

| arm | what's active | codex | ministral | qwen |
|---|---|---|---|---|
| **E0** | BM25 only *(baseline)* | 0.5940 | 0.6840 | 0.6205 |
| **E1** | + passage selection | 0.5940 | 0.6840 | 0.6205 |
| **E2** | + e5 biencoder and RRF | 0.6345 | 0.7336 | 0.6883 |
| **E3** | + cross-encoder | 0.6449 | 0.7938 | 0.7167 |
| **E4** | + LLM listwise *(full system)* | **0.6719** | **0.8136** | **0.7270** |

The score climbs at every step for all three assessors. Full system over BM25: **+0.078 / +0.130 / +0.107**. That sits well above the roughly 0.04 this dev set can resolve (§7), so H1 holds.

E1 comes out exactly equal to E0, and thats a harness check rather than a null result. With no reranker running, passage selection rewrites document *text* but nothing downstream reads it, so BM25's ordering cannot move. The fact that it is bit identical tells us the ablation is isolating what it claims to isolate.

### 3.1 Why condensed lists, and the wrong conclusion we nearly shipped

Score these runs naively and BM25 looks like it beats the full cascade by a mile, 0.6840 against 0.4957 on ministral. That number is an artifact. Reporting it would have been badly wrong.

Here is the diagnostic. What share of each run's top 10 did the assessors actually look at?

| arm | judged@10 | judged@50 |
|---|---|---|
| E0 BM25 | **100.0%** | 100.0% |
| E2 +e5/RRF | 92.7% | 72.7% |
| E4 full | **56.4%** | 47.6% |

The judgment pool was built from BM25 output, around 1,197 judged docs per topic, so roughly a top 1000 pool. Unjudged documents count as zero in nDCG. That means any reranker gets punished for surfacing documents the pool never contained, which is a property of the pool rather than of the system. BM25 scores 100% judged because the pool is literally its own output.

Sakai's condensed lists (`--condensed`) drop the unjudged documents before scoring, so systems are compared only on documents assessors actually saw. That reverses the ordering. We report both views because both say something useful. The condensed numbers say our reranking puts judged documents in a better order. The judged rate says our system pulls up a lot of material from outside the pool, and nobody knows whether that material is relevant. It could be new relevant content, or it could be junk. This dev set cannot settle it and we shouldn't pretend otherwise.

---

## 4. H2 to H4: what each stage is worth on its own

All of these are paired per topic tests. Same topics across arms, so topic difficulty cancels out.

### 4.1 H4, does LTR help? No

LTR reranked 500 documents down to 500 with no truncation, and then the cross-encoder re-sorted everything by its own score and threw LTR's ordering away. 5 out of 5 topics produced byte identical output with LTR on or off. It is disabled now. This also kills off the "train the LTR model" task, because under this topology a trained ranker would still change nothing downstream.

### 4.2 H2, does narrowing listwise to the top 20 cost quality? No, and thats good news

| assessor | mean paired Δ nDCG@10 | t |
|---|---|---|
| codex | −0.005 | −0.33 |
| ministral | −0.004 | −0.25 |
| qwen | +0.001 | +0.05 |

The t values sit near zero and the sign flips between assessors. Going from 50 down to 20 cut LLM calls from 4 to 1 and took about 31% off the wall clock, at no measurable cost.

### 4.3 H3, does listwise contribute? Probably, but not significantly

| assessor | mean paired Δ vs no-listwise | t |
|---|---|---|
| codex | +0.027 | +1.54 |
| ministral | +0.021 | +0.93 |
| qwen | +0.013 | +0.61 |

All three assessors come out positive. Pooled, that is 40 wins against 26 losses, sign test p = 0.109, which is indicative at best given the three assessors are not independent. Taken individually, none of them clears significance. We are keeping the stage because the sign is consistant across assessors, and because it now costs about 9 s per narrative rather than the 32 s it cost back when it reranked all 50.

---

## 5. H5: latency, and whether the cost is worth paying

### 5.0 Why we bother measuring this

The honest question about any stage in a cascade is not whether it helps. It is whether it helps enough to justify what it costs. A quality table can't answer that. +0.02 nDCG for 0.5 s and +0.02 nDCG for 30 s are very different engineering propositions and they look the same on paper.

So we instrumented latency per stage on the *same runs* that produced the quality numbers. That lets us state the trade directly, and plot quality against cost as stages get added, which is what the notebook paper needs. It also makes the system reproducible in a way a quality score alone is not, because the same code on different hardware or with a cold cache is a materially different system.

Measured on 2× NVIDIA RTX 6000 Ada (49 GB each, driver 575.57.08), an Intel Xeon w9-3595X with 120 logical cores, 754 GB RAM, Linux 6.1.0-47. Single process, one narrative at a time (`--workers 1`). The neural stages share one model instance each, so these are serial per narrative latencies rather than a throughput tuned deployment.

### 5.1 Phase A, preprocessing (query decomposition)

| step | s/narrative | notes |
|---|---|---|
| Step 1: decomposition | **0.92** | 119 narratives in 1 m 49 s, batched vLLM (Qwen2.5-72B-AWQ, tp=2) |
| Step 2: per-sub-question retrieval | *not built* | planned |
| Step 3: per-facet generation | *not built* | planned |
| **Phase A total (current)** | **0.92** | plus a one time model load of about 4.7 min, amortised over the run |

**This is the most actionable number in the document.** Decomposition costs 0.92 s per narrative under *batched* vLLM. A single *serial* Ollama generation call costs 34 s. That is a factor of roughly 37 for comparable LLM work, and the gap is batching, not model size. vLLM held 793 input tok/s and 150 output tok/s across a 119 prompt batch. Steps 2 and 3 should therefore be built on batched vLLM. Build them as per narrative calls and they will eat the runtime for no good reason.

### 5.2 Phase B, retrieval pipeline

| stage | kind | s/narrative | share of Phase B |
|---|---|---|---|
| bm25 | network (Pyserini) | 0.23 cached / **4.23 cold** | 1.4% |
| passage_selection | CPU | 0.96 | 5.8% |
| biencoder (e5) | GPU | 4.96 | 30.1% |
| rrf | CPU | ~0.00 | 0.0% |
| cross_encoder | GPU | 0.45 | 2.7% |
| llm_listwise | LLM (Ollama) | 9.89 | 60.0% |
| **Phase B total** | | **16.49** | 100% |

**The Pyserini API is not the bottleneck.** 4.23 s cold, against 9.89 s for listwise reranking and 34 s for generation. Worth saying out loud, because a remote index is the obvious thing to blame when a system feels slow.

### 5.3 Phase C, answer generation

| stage | kind | s/narrative | share |
|---|---|---|---|
| generation | LLM (Ollama, serial) | **34.10** | 100% |

### 5.4 End to end

| phase | s/narrative | share of total |
|---|---|---|
| A, preprocessing | 0.92 | 1.8% |
| B, retrieval pipeline | 16.49 | 32.0% |
| C, generation | 34.10 | **66.2%** |
| **total** | **51.51** | about 70 narratives/hour |

That works out at roughly 100 min for the 119 narrative test set with a cached index, or about 108 min with live retrieval.

### 5.5 The trade (H5)

Quality against cost as retrieval stages get added:

| arm | cumulative s/narrative | condensed nDCG@10 (ministral) | Δ nDCG per added second |
|---|---|---|---|
| BM25 | ~1.2 | 0.6840 | n/a |
| + e5/RRF | ~6.2 | 0.7336 | +0.0099 |
| + cross-encoder | ~6.6 | 0.7938 | **+0.1505** |
| + listwise | ~16.5 | 0.8136 | +0.0020 |

**Quality and cost both climb monotonically, so the trade is favourable overall and H5 is supported.** The marginal returns are another story. They differ by two orders of magnitude. The cross-encoder is far and away the best value at +0.060 nDCG for 0.45 s, while listwise buys +0.020 for 9.89 s. We keep both, for seperate reasons. The cross-encoder because it is nearly free, listwise because the sign is consistent and the budget currently allows it.

### 5.6 Caveats on any latency figure we publish

- the `bm25` row is a **cached local disk read** at 0.23 s, not live API latency, which measured **4.23 s** against the real service. Same code, roughly 18× apart. Any published table has to say which condition it describes.
- everything here is serial and single narrative. Batch it or run it concurrently and the picture changes a lot, as §5.1 shows.
- Phase A only covers step 1. Steps 2 and 3 do not exist yet, so the current total is not what the finished system will cost.

---

## 6. What we still cannot claim

Putting these plainly, because they are the gaps that matter.

1. **We have not evaluated the actual task.** TREC RAG 2026 judges *answers*, through pairwise battles and nugget coverage, not rankings. Everything above is retrieval. `scripts/eval/evaluate_nuggets.py` is written and unit tested but has never been run. This is the biggest missing experiment.
2. **No external baseline.** We compare our own configurations against each other and against BM25. The five official baseline runs cannot be scored, because they cover the 119 *test* narratives and no qrels exist for those. So we have no idea how we compare to other participants.
3. **The judgments are projected, not human.** Model generated UMBRELA labels, which the organizers describe as development diagnostics. These are not NIST assessments.
4. **n = 22 topics**, resolution around 0.04 nDCG@10 (§7). Fine for the ablation effects above. Not fine for tuning.
5. **We don't know what sits outside the pool.** About 44% of our top 10 is unjudged (§3.1). Whether that is a strength or a weakness is not something this data can settle.
6. **Answer coverage is weak and we know it.** Answers average 373 words against a 1024 word budget, and they sit at a near constant 9.5 sentences whether the narrative asks 1 question or 8. The correlation is −0.065. Coverage is the axis the RAG task competes on, so this is the open problem.

---

## 7. A note on the statistics

An earlier reading of this data used the *unpaired* standard error, 0.198/√22 which is about 0.042, giving a detectable difference near 0.084. That is the wrong test. Every arm is scored on the same topics, so topic difficulty cancels and the **paired** difference SE is what applies. Measured, that comes to 0.014 to 0.023, so the resolution is about 0.04 nDCG@10. The dev set is roughly twice as sensitive as we first said.

---

## 8. Where everything lives

| what | where |
|---|---|
| Banked submission (test set, validated) | `/media/12TB/shared/datasets/trec/trec-rag-2026/submissions/CURRENT_bruteforce-cascade_20260801_0709/` |
| Retrieval evaluator | `/media/12TB/shared/shared_projects/trec-rag-2026/scripts/eval/evaluate_retrieval.py` |
| Nugget evaluator (built, not yet run) | `/media/12TB/shared/shared_projects/trec-rag-2026/scripts/eval/evaluate_nuggets.py` |
| Latency reporter | `/media/12TB/shared/shared_projects/trec-rag-2026.scripts/eval/report_latency.py` |
| Per-run stage logs | `/media/12TB/shared/shared_projects/trec-rag-2026/logs/stages_*.jsonl` |
| Full engineering history | `/media/12TB/shared/shared_projects/trec-rag-2026/WORK_LOG.md` |

To reproduce the headline table:

```bash
python /media/12TB/shared/shared_projects/trec-rag-2026/scripts/eval/evaluate_retrieval.py /tmp/E0_bm25.trec /tmp/E4_full.trec \
    --condensed --judged-rate
```