# Decomposed vs composite — dev-set result

**TL;DR: the decomposed arm does not beat the composite cascade. Submit composite as
primary, decomposed as a contrastive second run.**

Measured 2026-08-07 on the 22-topic dev set, condensed lists, all three projected
UMBRELA assessor variants. The decomposed dev run retrieved **171/171 atomic needs
(100%)**, 0 verifier rejections — so the arm ran at full strength and this is not a
measurement artefact.

## nDCG@10 (condensed)

| assessor | composite | decomposed | Δ |
|---|---|---|---|
| codex-gpt5.5-medium-reasoning-v1 | 0.6971 | 0.7093 | **+0.012** |
| ministral-3-14b-instruct-2512-v2 | 0.8238 | 0.8210 | **−0.003** |
| qwen3.5-9b-v2 | 0.7416 | 0.7292 | **−0.012** |

Detectable difference on this set is **~0.04** (paired SE 0.014–0.023). Every delta is
inside ±0.013 and the sign flips across assessors. That is a wash, not a win.

## Deeper cutoffs are consistently worse

Small, but the direction holds across all three assessors and four metrics, which
noise does not do:

| metric | codex | ministral | qwen3.5 |
|---|---|---|---|
| nDCG@20 | −0.007 | −0.006 | −0.011 |
| nDCG@100 | −0.008 | −0.013 | −0.011 |
| MAP | −0.002 | −0.002 | −0.002 |
| R@100 | −0.001 | −0.002 | −0.001 |

## The core hypothesis failed its own test

Decomposition should help most on narratives that ask many things. It doesn't:

| | codex | ministral | qwen3.5 |
|---|---|---|---|
| Pearson r (per-topic Δ vs sub-question count) | 0.11 | 0.08 | 0.02 |
| wins / losses | 13W/9L | 11W/9L | 9W/13L |

Split by narrative width, averaged over assessors: **1–2 sub-q −0.003 · 3–4 sub-q
+0.001 · 5+ sub-q −0.002**. There is no subset where it pays.

## Suspected cause — tested, and ruled out

The deeper-cutoff shortfall pointed at the assumption flagged as untested in
`configs/pipeline.decomposed.yaml`: `per_ain.bm25_hits: 100` against the composite's
1000. The arm builds its deep ranks from a much shallower pool, which is exactly
where it loses, and the config says "raise this first if recall looks thin".

**Raised it to 1000 and it got worse, not better** (`pipeline.decomposed.hits1000.yaml`,
one substantive line different; it also lifts the per-need RRF pool 100 -> 500 via
`hybrid_top_k = min(500, bm25_hits)`). Dev run: 22 topics, 171/171 needs
retrieved over two sessions (the second resumed with 3 topics checkpointed and ran
the other 19, 146 needs, in 60 min), so the arm ran at full strength.

| metric | Δ vs hits=100 (3 assessors) | Δ vs composite |
|---|---|---|
| nDCG@10 | −0.029 −0.020 −0.018 | −0.017 −0.023 −0.031 |
| nDCG@20 | −0.050 −0.061 −0.056 | −0.057 −0.067 −0.067 |
| nDCG@100 | −0.026 −0.032 −0.030 | −0.034 −0.045 −0.041 |
| R@100 | −0.004 −0.004 −0.004 | −0.005 −0.006 −0.005 |
| MAP | −0.003 −0.004 −0.003 | −0.005 −0.006 −0.005 |

**Worse in 15/15 (assessor x metric) cells against both baselines**, and the nDCG@20
loss exceeds the ~0.04 detection threshold — a real degradation, not noise. So the
shallow per-need pool is NOT the explanation for the deep-cutoff shortfall; feeding
the cross-encoder 10x more candidates per need makes it promote worse documents,
which is the wrong-aspect failure mode this project already documents.

One caveat on how far to trust the direction: `judged@10` falls from 55.9% to 44.1%,
so the deeper run diverges much further from the judgment pool. Condensed scoring
drops unjudged documents before scoring and so controls for the obvious form of that
bias, but it cannot credit genuinely new relevant material the pool never saw. The
measurement says "worse among judged documents"; it cannot say "worse, full stop".
The consistency across 15/15 cells still puts the weight of evidence on a real loss.

**Do not raise `bm25_hits`.** The submitted decomposed run uses 100, which is the
better of the two.

## Caveats

- Projected UMBRELA judgments, not NIST assessments. A win here is a signal, not a result.
- 22 topics. Underpowered for anything below ~0.04.
- `--condensed` is mandatory: the pool was built from BM25 output, so a decomposed run
  surfaces even more unjudged material. Scored naively BM25 appears to beat the full
  cascade (0.6840 vs 0.4957) purely from judged@10 of 100% vs 56.4%.
- **Answer quality is unmeasured on both arms.** These are retrieval metrics only;
  `evaluate_nuggets.py` has not been run.

## Tooling bug found while doing this

`scripts/eval/evaluate_retrieval.py` keys its run table on `path.name`
(line 153, `runs[path.name] = load_run(path)`). Two runs that both use the standard
`r_output_trec_rag_2026.tsv` filename silently overwrite each other — it prints one
row and scores one run twice, no warning. Any comparison of runs living in separate
per-experiment directories is affected. Worked around here with distinctly-named
symlinks; the script should key on the full path or reject duplicate basenames.
