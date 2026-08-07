# Changes on this branch

Work added to `nahid/bruteforce-approach` on top of `545832a`, **not yet
committed**. Written 2026-08-08.

For the 13 commits already on the branch (dev-set evaluation, latency
instrumentation, stage ablation, decomposition steps 1b/2/3), see
`.venv/rnd/RECENT_WORK.md`. This document covers only what came after them.

Two things were built:

1. **The per-AIN retrieval arm** — the join that finally consumes the
   decomposition output. Before this, three preprocessors produced 1,270
   expanded queries that nothing read.
2. **Generation rework** — prompts as Jinja templates, optional `<think>`
   reasoning, and an answer verifier that can reject and regenerate a draft.

**413 tests pass, up from 267.** Every edit to an existing file defaults to
prior behaviour when the new config keys are absent; `configs/pipeline.gpu.yaml`
— the banked submission path — is byte-identical.

---

## Inventory

New files:

| File | Lines | Purpose |
|---|---:|---|
| `src/trecrag/decomposed/schema.py` | 129 | `AtomicNeed` / `SubQuestion` / `DecomposedTopic` |
| `src/trecrag/decomposed/loader.py` | 198 | step-3 JSONL → objects, with the alignment gate |
| `src/trecrag/decomposed/fusion.py` | 101 | cross-AIN weighted RRF + provenance features |
| `src/trecrag/decomposed/evidence.py` | 106 | coverage-aware evidence selection |
| `src/trecrag/decomposed/orchestrator.py` | 247 | `DecomposedPipeline` |
| `src/trecrag/decomposed/__init__.py` | 34 | public surface |
| `src/trecrag/generation/prompts.py` | 133 | Jinja environment + both renderers |
| `src/trecrag/generation/verifier.py` | 402 | hallucination / omission checks |
| `src/trecrag/generation/templates/rag_answer.jinja` | 61 | generator prompt |
| `src/trecrag/generation/templates/answer_verification.jinja` | 46 | judge prompt |
| `configs/pipeline.decomposed.yaml` | 292 | the per-AIN config |

Modified files, all backward-compatible:

| File | Δ | Change |
|---|---:|---|
| `src/trecrag/factory.py` | +198 | `build_decomposed_pipeline_from_config`, `build_verifier` |
| `src/trecrag/generation/generator.py` | ±153 | retry loop, thinking, prompt moved to template |
| `scripts/run_pipeline.py` | +39 | `--decomposition`, `--per-ain-diagnostics`, early validation |
| `src/trecrag/retrieval/hybrid.py` | +19 | optional `weights=` on RRF |
| `src/trecrag/clients/llm_client.py` | +16 | `strip_reasoning` handles unterminated blocks |
| `pyproject.toml` | +3 | `Jinja2>=3.1` as a base dependency |

Tests: 166 new across six files —
`test_decomposed_loader` (17), `test_decomposed_fusion` (20),
`test_decomposed_evidence` (16), `test_decomposed_pipeline` (24),
`test_generator_prompt` (18), `test_generator_verifier` (51).

---

## 1. Per-AIN retrieval arm

### What it does

Runs the existing cascade once per atomic information need, fuses the per-need
runs with a full-depth composite run, reranks the pool once, and hands the
generator evidence chosen to span the narrative's sub-questions.

Mode is selected by the config carrying a `decomposition:` block. Nothing else
changes.

### Design decisions worth knowing

**`DecomposedPipeline` exposes the same `run_topic(topic) -> (ranked, RagAnswer)`
contract as `RagPipeline`.** That one choice meant `scripts/run_pipeline.py`'s
worker loop, per-topic checkpointing, canonical-order assembly, `.partial` guard
and failure handling all work unchanged — no duplication of the validated run
harness.

**Listwise reranking runs once, on the pooled list, never per need.** It costs
~9.9 s a call against ~0.5 s for the cross-encoder, and a narrative averages
~10.7 needs. Per-need listwise would make one stage most of a multi-hour run to
order candidate sets that are about to be merged anyway. Implemented by building
the per-need cascade with `llm_listwise=None` and holding the reranker at the
orchestrator.

**The composite run is fused in as a floor.** Whatever the baseline already
finds stays in the pool, so decomposition can add recall but not silently lose
it. `reciprocal_rank_fusion` gained an optional `weights=` argument for this —
one composite list against ~11 per-need lists would otherwise hold a ~1/12 vote
purely by being outnumbered.

**Both cascades share model instances.** The factory builds e5 and the
cross-encoder once and constructs two `RagPipeline` objects over them; calling
`build_pipeline_from_config` twice would put a second copy of each on the GPU.

**`per_ain.bm25_hits: 100`, not 1000.** A ClimbMix response at full depth is
~63 MB; 1,270 needs would mean ~80 GB of new cache. This is flagged in the
config as an **untested assumption** — `WORK_LOG.md` §5.3 measured depth 1000 as
load-bearing for the *composite* query, and nobody has checked whether that
transfers to a narrow need.

### Two failure modes this had to be built around

**The step-2/step-3 join.** `ain` and `expanded_queries` come from separate
script runs. A mismatched join retrieves a well-formed query for the wrong
information need, every downstream stage behaves normally, and the run answers a
different question than it reports — nothing downstream can detect it. The
loader keys on `(sub_question_id, ain_id)` and raises on any disagreement.

**The 3C field is populated even though PRF never ran.** With
`step3c_prf_terms: []` the term-append returns the 3B string verbatim, so
reading `step3c_expanded_query` alone would report a PRF run that did not
happen. The loader gates on `step3c_status`, and a test asserts PRF is absent
across all 1,270 committed records.

---

## 2. Generation rework

### Prompts are Jinja templates

`src/trecrag/generation/templates/`, rendered by `prompts.py` with
`StrictUndefined` — a mistyped variable fails loudly rather than rendering a
prompt quietly missing its word target or citation cap, which nothing downstream
would catch. Every instruction in the answer template was an observed failure
over 119 narratives, and a test pins them so they cannot be trimmed for brevity.

`Jinja2` moves into base dependencies, not an extra: without it the package
cannot generate an answer. It was already installed but unlisted, so a clean
install would have failed.

### Thinking

Both prompts optionally ask for a `<think>` block with scaffolded steps —
enumerate the asks, map passages to each, plan conditional guidance where
evidence is thin, budget sentences across asks, confirm each claim traces to a
citation. The judge's scaffold is written for judging instead.

**This required hardening `strip_reasoning` first.** `WORK_LOG.md` §4.4 records
reasoning tokens corrupting both parsers, which is why the project runs a
non-reasoning model. The existing regex matched only **closed**
`<think>…</think>`; a block truncated by the token budget kept no closing tag,
passed through untouched, and `_JSON_ARRAY_RE` then read brackets out of the
reasoning — a silently wrong answer. Deliberately prompting for a think block
makes that likely rather than rare. It now also drops an unterminated block to
end-of-string, which yields an empty string and the extractive fallback.

Token budgets rise when thinking is on — generator 4096 → 6144, judge 512 →
1536 — because the block is spent *before* the payload.

### Verification and regeneration

`verifier.py`. **Off by default in code**; enabled in
`configs/pipeline.decomposed.yaml` only.

- `LexicalGroundingVerifier` — free, no API call. Hallucination: per-sentence
  content-word overlap against the passage it *cites*, aimed at the measured
  failure, which §5.5 shows is **misattribution, not fabrication** (33 sentences
  matched a retrieved-but-uncited document). Omission: uncited evidence, plus
  sub-questions with a passage in the pool but no citation.
- `LLMVerifier` — opt-in, one judging call per answer.
- `CompositeVerifier` — cheapest-first, short-circuiting.

A rejected draft is regenerated **once**, with the verifier's specific findings
injected into the prompt's revision block. "Try again" reliably reproduces the
same answer.

**The failure mode this had to be protected against.** An unreadable judgement is
*passed through*, not failed — rejecting on the verifier's own parse errors would
regenerate good answers and ship the last draft anyway. That made "I checked and
it's clean" and "I couldn't check" the same value, so a broken judge produced a
run of unverified answers that read as a run of clean ones.
`VerificationResult.checked` separates them; every unverified answer is logged by
topic id; three consecutive unreadable judgements log an ERROR naming the likely
cause and stop the calls; `strict: true` raises instead.

### Coverage-aware evidence selection

Taking the pool's top 10 in rank order would have left the original problem in
place. Fusion rewards agreement and needs cluster inside sub-questions, so the
head of the pool skews toward whichever sub-question spawned the most needs — a
narrative can decompose into eight sub-questions, retrieve well for all eight,
and still hand the generator ten documents about the first two. Demonstrated on
the real `rag2026-0` decomposition with a deterministic retriever:

```
rank-order    : [sq1_doc0, sq1_doc1, sq1_doc2, sq1_doc3]  → covers sq1 only
coverage-aware: [sq1_doc0, sq1_doc1, sq1_doc2, sq2_only]  → covers sq1 + sq2
```

Selection round-robins across sub-questions, rarest coverage first, then fills
remaining slots by global rank.

`effective_evidence_k()` floors `evidence_k` at the sub-question count. **At the
default this is inert** — the widest narrative has 8 sub-questions against
`evidence_k: 10` — and it is documented as such. It exists so the floor is
principled rather than a flat bump, which `ARCHITECTURE.md` §4.3 deliberately
held off on.

---

## Compatibility

- `configs/pipeline.gpu.yaml` is **byte-identical**. The banked submission path
  runs exactly as before: no thinking, no verification, single-pass generation.
- `reciprocal_rank_fusion(weights=None)` reproduces its previous output exactly;
  a test asserts it.
- `LLMGenerator` with no verifier makes exactly one LLM call, as before.
- `scripts/run_pipeline.py` without a `decomposition:` block in the config takes
  the original path.
- `strip_reasoning` on text with no `<think>` block, or a closed one, behaves as
  before. Only the unterminated case changed, and it changed from
  "silently corrupt" to "fall back".

---

## What this does not do

- **The decomposed arm has never run against the live service.** Everything
  above was verified with stubbed retrieval and unit tests.
- **It cannot be scored yet.** The committed decomposition covers the 119 test
  narratives; the 22 dev topics have integer ids and **0 of 22** are covered.
  Steps 1–3 must be run over the dev topics first — see `RUNNING.md` §4.
- **Answer quality is unmeasured**, for either arm. The verifier is a guardrail,
  not a metric: it checks a draft against its own evidence, which says nothing
  about whether the answer is good.
- **Step 3C / PRF is still not run** — `status: "disabled"` on all 1,270
  records. Independent of this work; `decomposition.query_variant` picks it up
  for free once populated.
- **Sub-question dependencies are loaded and carried but not acted on.**
  Sequential retrieval is a different architecture.
