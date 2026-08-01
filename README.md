# trec-rag-2026

TREC RAG 2026 submission pipeline for the RAG (`RAG`) task, implementing:

```
Query --> (BM25 + Biencoder) --> LTR --> ColBERT --> QDER --> CE --> LLM Listwise --> Final Ranked-List --> Generator
```

Retrieves from NVIDIA's ClimbMix-400b corpus via the official Pyserini REST
API, cascades through six reranking stages, and generates a sentence-cited,
word/citation-bounded answer per the TREC RAG 2026 track guidelines. See
`../plans/trec-rag-2026-participation-guide.md` for the full task/format
writeup this implementation targets, and `../plans/qder-approach.md` for the
QDER reranking stage's approach.

## Design principle: every stage runs with zero setup

Each neural stage (biencoder, ColBERT, cross-encoder, LLM listwise, LLM
generation) has a dependency-free fallback backend (hashing-trick encoders,
lexical scorers, or an identity no-op LLM client) alongside a real backend
behind a lazy, optional import. The pipeline is fully runnable end to end
with only `requests` + `PyYAML` + `numpy` and no API keys — swap
`backend:` in `configs/pipeline.yaml` per stage as real models/credentials
become available; nothing else changes.

| Stage | Fallback (default) | Real backend |
|---|---|---|
| Biencoder | hashing-trick bag-of-words cosine | `sentence-transformers` (`neural` extra) |
| LTR | per-feature min-max weighted linear sum | LightGBM LambdaMART (`ltr` extra), see `scripts/train_ltr.py` |
| ColBERT | hashed one-hot token MaxSim | HF transformer token embeddings + MaxSim (`neural` extra) |
| QDER | hashed text (+ optional heuristic entity) channels, untrained bilinear scorer | vendored `QDERModel` (BERT text channel, `neural` extra), see below |
| Cross-encoder | lexical token-overlap scorer | `sentence_transformers.CrossEncoder` (`neural` extra) |
| LLM listwise | identity (keeps input order) | OpenAI-compatible or Anthropic client (`llm` extra) |
| Generator | extractive (best query-overlap sentence per doc) | LLM JSON-cited generator with extractive fallback on parse failure |

## Project layout

```
configs/pipeline.yaml         Zero-setup default: stage depths, backend switches, env var names
configs/pipeline.gpu.yaml     Submission config: real backends on GPU + Ollama (see "Two configs")
src/trecrag/
  schema.py                   Topic / Candidate / RagAnswer dataclasses
  config.py                   YAML config + .env.local loader (never logs secrets)
  text_utils.py                tokenize() / split_sentences() shared by fallback backends
  chunking.py                  Query-focused passage selection (PassageSelector)
  logging_utils.py             Console + optional file logging, quiets noisy HTTP libraries
  clients/
    pyserini_client.py         ClimbMix search + doc fetch (Pyserini REST API); response cache,
                                transport-level retry, and adaptive-depth fallback
    llm_client.py              LLMClient interface + Identity/OpenAI-compatible/Anthropic backends;
                                retries, strips reasoning-model <think> blocks
  retrieval/
    bm25.py                     First-stage sparse retrieval
    biencoder.py                 Dense re-scoring of the BM25 pool (hashing | sentence-transformer)
    hybrid.py                   Reciprocal Rank Fusion
  rerank/
    ltr.py                       Learning-to-rank fusion of upstream features
    colbert.py                   Late-interaction MaxSim reranking
    qder/                        Query-specific Doc/Entity Representations reranking (see below)
    cross_encoder.py            Joint (query, passage) scoring
    llm_listwise.py             RankGPT-style sliding-window listwise reranking
  generation/
    generator.py                 Extractive / LLM answer generation
    postprocess.py               Enforces word/citation/reference submission rules
  output/
    writers.py                   Writes the two official submission files
    checkpoint.py                Per-topic checkpointing so a long run is resumable/crash-safe
  validate.py                  Offline checks against the track's validation rules
  pipeline.py                  RagPipeline orchestrator (stage order + depths)
  factory.py                   Builds a RagPipeline from configs/pipeline.yaml
scripts/
  run_pipeline.py              CLI: topics -> submission files (resumable, checkpointed)
  check_api.py                 First-contact smoke test for the Pyserini API before a real run
  validate_submission.py       CLI: validate submission files
  train_ltr.py                 CLI: train the LTR LightGBM booster from a qrels file
  prepare_qder_data.py         CLI: build QDER training/dev data from this pipeline's own cascade
  train_qder.py                 CLI: prepare data -> train -> unwrap checkpoint, one command
data/dev/                      RAG25 dev topics/qrels/nuggets/rubrics (in-repo, no sibling-repo needed)
third_party/qder_training/     Vendored SIGIR2025-QDER doc-ranking trainer (train.py etc., MIT-licensed)
tests/                         unittest suite (network-free; fakes/hashing backends only)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core deps: requests, PyYAML, numpy
pip install -e ".[neural,ltr,llm]"   # optional: real model backends
```

### Secrets

Never pass tokens as CLI arguments or commit them. Create a repo-local
`.env.local` (already gitignored):

```
PYSERINI_API_TOKEN=...
# only if using real LLM backends (configs/pipeline.yaml `llm` section):
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
```

Request a Pyserini API token by emailing `get-pyserini@googlegroups.com`.
`scripts/*.py` load `.env.local` automatically via `trecrag.config.load_dotenv_local`.

## Two configs

| Config | Purpose |
|---|---|
| `configs/pipeline.yaml` | Zero-setup default. Every neural stage runs its dependency-free fallback, so it executes anywhere with no GPU, models, or keys. Useful for plumbing tests — **not** for a real submission: listwise reranking is a literal no-op and answers are extracted sentences. |
| `configs/pipeline.gpu.yaml` | The submission config. Real E5 biencoder + cross-encoder on GPU, Ollama-backed listwise reranking and answer generation, on-disk caching of Pyserini responses, and query-focused passage selection. ColBERT is deliberately disabled (no real checkpoint available, and the batched cross-encoder handles the full pool cheaply enough that pruning ahead of it with a weak proxy only costs recall); QDER stays enabled on its hashing fallback since a real training path exists (`scripts/train_qder.py`) even though no checkpoint has been trained yet. |

Before a real run, check the API is reachable and behaving as expected:

```bash
python scripts/check_api.py --config configs/pipeline.gpu.yaml
```

## Running the pipeline

```bash
python scripts/run_pipeline.py \
  --topics trec_rag_2026_test_query_set.tsv \
  --config configs/pipeline.gpu.yaml \
  --team-id my-team --run-id my-rag-run \
  --out-dir output/
```

Writes `output/r_output_trec_rag_2026.tsv` and `output/rag_output_trec_rag_2026.jsonl`
(filenames come from the config). Use `--task retrieval|rag|both`,
`--limit N` to smoke-test on the first N topics, and `--workers N` for
concurrent (network-bound) topic processing. Use `--workers 1` with GPU
backends: each neural stage holds one model instance, and concurrent forward
passes across threads are not safe.

### Resuming

The run is resumable: every finished topic is checkpointed to
`<out-dir>/checkpoints/` (`--work-dir` to override) the moment it's produced,
so a crash or a rate-limit failure costs only that topic, not the run.
Re-invoking the same command skips topics already checkpointed; pass
`--no-resume` to force recomputation. A topic that raises is logged and
skipped rather than aborting the run — re-run the same command to retry just
the failed ones. A run that doesn't cover every topic in the input file
(including a `--limit` smoke test) writes `<name>.partial` instead of the
official filenames, so an in-progress or partial run can never overwrite a
complete submission. Logs are teed to `<out-dir>/run_<run-id>.log` by default
(`--log-file` to override).

## Validating a submission

```bash
python scripts/validate_submission.py \
  --retrieval output/r_output_trec_rag_2026.tsv \
  --rag output/rag_output_trec_rag_2026.jsonl \
  --topics trec_rag_2026_test_query_set.tsv
```

Checks: 6-column run format, ranks restarting at 1 and non-increasing scores
per topic, full topic coverage (plus a soft check flagging any topic whose
retrieval count is far below the run's own median), RAG metadata shape
(extra participant-defined keys are allowed), citation bounds (≤3/sentence,
valid indices), an explicit empty-answer violation, ≤1024-word answers, and
a best-effort scan for token-like secrets in the output files. Uncited
references are deliberately **not** flagged — the track spec explicitly
permits them.

## Training the LTR stage

The default LTR backend is a dependency-free linear fallback. `scripts/train_ltr.py`
trains a real LightGBM LambdaMART booster, on the same real TREC RAG 2026 dev
data used to train QDER (`data/dev/` — 22 topics with UMBRELA qrels, the only
labeled RAG26 data that exists anywhere; see "Training a QDER checkpoint"
below for why). For each topic it rebuilds the pool exactly as the live
pipeline would (BM25 → biencoder → RRF), keeps only candidates the qrels file
actually judged, and trains against those — no separate held-out dev split,
unlike QDER's trainer, since LightGBM doesn't need one for a plain
`num_boost_round` fit:

```bash
pip install -e ".[ltr]"
python scripts/train_ltr.py \
  --topics data/dev/topics/rag25-topics-dev.tsv \
  --qrels data/dev/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \
  --output-model models/ltr_lambdamart.txt
```

A topic with fewer than 2 judged candidates in its pool is skipped (logged as
a warning) — with only 22 topics total, check the log for how many actually
contributed training rows. `--feature-names` (default: `bm25_score`,
`bm25_rank`, `bien_score`, `bien_rank`, `rrf_score`) and `--num-boost-round`
(default 100) are the two knobs worth tuning first.

Then set `stages.ltr.model_path` in `configs/pipeline.yaml` to the saved model.

> **Caveat**: `train_ltr.py`'s `label_gain` is computed as `list(range(max(labels) + 1))`,
> which assumes qrels grades are non-negative. A qrels file using negative
> grades (e.g. a "spam" judgment below 0) would silently mis-index via
> Python's negative-index wraparound rather than erroring — none of the
> `data/dev/` UMBRELA qrels currently do this, but it's untested against one
> that does.

## The QDER stage

`src/trecrag/rerank/qder/` implements QDER (see `../plans/qder-approach.md`):
query-guided attention over a text channel (and an optional entity channel),
Addition + Multiplication interaction patterns, mean-pooled and scaled by an
upstream retrieval score, combined via a bilinear scorer — then optionally
interpolated with the upstream BM25-like score (`stages.qder.hybrid_lambda`,
a fixed stand-in for the paper's coordinate-ascent-tuned λ).

- `_vendor/` is vendored **unmodified** from `SIGIR2025-QDER` (`src/doc_ranking/base_model.py`,
  `text_embedding.py`, `qder_model.py`, main branch) — only the two local
  imports in `qder_model.py` were adjusted to this package's layout. This is
  the real BERT-based `QDERModel`, used by `TransformerQDERScorer`.
- `entities.py` defines `EntityEmbeddingProvider`. Real entity linking (WAT)
  and Wikipedia2Vec embeddings for ClimbMix aren't available here, so
  `use_entities` defaults to `false`. `HeuristicEntityProvider` (capitalized-
  phrase spotting + hashed vectors) is a dependency-free stand-in that
  exercises the entity channel's code path when `use_entities: true`; swap in
  a real linker + embedding table for production use.
- `reranker.py` has `HashingQDERScorer` (default fallback — same architecture,
  hashed vectors, untrained/identity bilinear matrix) and `TransformerQDERScorer`
  (wraps the vendored `QDERModel`; pass `model_path` to load a trained checkpoint).

### Training a QDER checkpoint

The trainer is vendored in-repo at `third_party/qder_training/` (a straight,
unmodified copy of SIGIR2025-QDER's `src/doc_ranking/`, MIT-licensed — see
`third_party/qder_training/README.md`), so no sibling-repo checkout is needed.
`scripts/train_qder.py` chains the whole thing — data prep, training, and
checkpoint-unwrapping — into one command, defaulting to the real TREC RAG
2026 dev data already in this repo (`data/dev/`, 22 topics with UMBRELA
qrels — the only labeled RAG26 data that exists anywhere; the 119-topic test
query set has no qrels and can't be used for training):

```bash
# Needs torch/transformers/tqdm/pytrec_eval, not part of this project's own
# dependency groups. CPU by default; add --use-cuda on a GPU box.
python scripts/train_qder.py --out-dir models/qder --use-cuda
```

This prints a ready-to-paste config block once it finishes. Re-running after
a training crash without re-paying the retrieval cascade's API cost:

```bash
python scripts/train_qder.py --out-dir models/qder --skip-data-prep --use-cuda
```

Under the hood, that's three steps you can also run individually — see each
script's own `--help` (`scripts/prepare_qder_data.py`,
`third_party/qder_training/train.py`) to inspect or reuse the intermediate
JSONL, or try a different qrels judge:

1. **Build data**: `prepare_qder_data.py` rebuilds the live candidate pool
   (BM25 → biencoder → RRF → LTR → ColBERT) and labels it from a qrels file.
2. **Train**: `third_party/qder_training/train.py` trains the checkpoint.
3. **Unwrap**: the trainer saves optimizer/scheduler state alongside the
   weights; `TransformerQDERScorer` expects a raw state dict, so
   `train_qder.py` unwraps `model_state_dict` into `model_weights_only.bin`.

Everything lands in `--out-dir` and is usable directly from there — no step
requires moving or copying files into any other directory (`--out-dir` can be
anywhere; there's nothing special about `models/qder` beyond it being the
example above). Two safety checks guard the two ways this can go wrong
silently: before doing any work, `train_qder.py` verifies `--topics`/`--qrels`
actually exist and warns (rather than failing deep inside a subprocess) if
`data/dev/` has been cleared or rebuilt with different filenames and the
defaults no longer resolve; after training, it verifies the trainer actually
wrote a checkpoint before attempting to unwrap it, and warns instead of
crashing on a confusing `torch.load` error if it didn't.

Then set in `configs/pipeline.yaml`:

```yaml
stages:
  qder:
    backend: transformer
    model_name: "bert-base-uncased"   # must match the checkpoint's --text-enc
    model_path: "models/qder/model_weights_only.bin"
    use_entities: false               # must match what the checkpoint was trained with
```

`use_entities`, `score_method`, and the text encoder must match exactly between
training and inference — they determine the model's parameter shapes, so a
mismatch fails to load outright. This repo has no real entity linker for
ClimbMix, so `third_party/qder_training/train.py`'s `--use-entities` flag
defaults to `false` (a deliberate deviation from upstream, which hardcoded
`True`) — leave it off to match this pipeline's default config; passing
`--use-entities` trains a checkpoint that can only load if
`stages.qder.use_entities: true` is also set on the inference side.

## Tests

```bash
python -m unittest discover -s tests -t .
```

(`-t .` matters: it makes `tests` import as a package so `tests/__init__.py`
puts `src/` on `sys.path` before any test module needs `trecrag`.) Tests run
fully offline against fakes and the hashing/lexical fallback backends — no
network, GPU, or API keys required. If `pytest` is installed (`pip install
-e ".[dev]"`), `pytest` from the repo root also works.

## Known gaps / next steps

- `stages.ltr.model_path` ships unset (linear fallback); train one with
  `scripts/train_ltr.py` against dev qrels for a real learned ranker.
- `stages.colbert.backend` / `stages.qder.backend` / `stages.cross_encoder.backend` /
  `stages.llm_listwise.backend` default to the dependency-free fallbacks;
  flip to `transformer` / `sentence_transformer` / `openai`|`anthropic`
  once compute/API budget is available, per `configs/pipeline.yaml`.
- `stages.qder.use_entities` ships `false` (no WAT linker / Wikipedia2Vec
  table for ClimbMix); `HeuristicEntityProvider` is a rough proxy if you want
  to exercise the entity channel before wiring in a real one.
- `stages.qder.hybrid_lambda` is a fixed config value, not the paper's
  coordinate-ascent-tuned λ — no training/tuning infrastructure for that is
  wired up here.
- No QDER checkpoint has been trained yet (`stages.qder.model_path: null` in
  both configs). `scripts/train_qder.py` (see "Training a QDER checkpoint"
  above) builds the data and trains it in one command, but nobody has run
  that end-to-end yet.
- `data/dev/` holds three separate RAG25 dev UMBRELA qrels files, graded by
  different judge models — they don't necessarily agree, and nothing here
  currently picks one over another for training/tuning; that's a judgment
  call left to whoever runs it.
- Submission logistics (upload portal, exact deadline) are still unconfirmed
  upstream — see `../plans/trec-rag-2026-participation-guide.md` §6–7 before
  the real submission.
