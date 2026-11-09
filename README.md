# trec-rag-2026

TREC RAG 2026 submission pipeline for the RAG (`RAG`) task, implementing:

```
Query --> (BM25 + Biencoder) --> LTR --> ColBERT --> CE --> LLM Listwise --> Final Ranked-List --> Generator
```

Retrieves from NVIDIA's ClimbMix-400b corpus via the official Pyserini REST
API, cascades through five reranking stages, and generates a sentence-cited,
word/citation-bounded answer per the TREC RAG 2026 track guidelines. See
`../plans/trec-rag-2026-participation-guide.md` for the full task/format
writeup this implementation targets.

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
| Cross-encoder | lexical token-overlap scorer | `sentence_transformers.CrossEncoder` (`neural` extra) |
| LLM listwise | identity (keeps input order) | OpenAI-compatible or Anthropic client (`llm` extra) |
| Generator | extractive (best query-overlap sentence per doc) | LLM JSON-cited generator with extractive fallback on parse failure |

## Project layout

```
configs/pipeline.yaml         Stage depths, backend switches, env var names
src/trecrag/
  schema.py                   Topic / Candidate / RagAnswer dataclasses
  config.py                   YAML config + .env.local loader (never logs secrets)
  text_utils.py                tokenize() / split_sentences() shared by fallback backends
  clients/
    pyserini_client.py         ClimbMix search + doc fetch (Pyserini REST API)
    llm_client.py              LLMClient interface + Identity/OpenAI-compatible/Anthropic backends
  retrieval/
    bm25.py                     First-stage sparse retrieval
    biencoder.py                 Dense re-scoring of the BM25 pool (hashing | sentence-transformer)
    hybrid.py                   Reciprocal Rank Fusion
  rerank/
    ltr.py                       Learning-to-rank fusion of upstream features
    colbert.py                   Late-interaction MaxSim reranking
    cross_encoder.py            Joint (query, passage) scoring
    llm_listwise.py             RankGPT-style sliding-window listwise reranking
  generation/
    generator.py                 Extractive / LLM answer generation
    postprocess.py               Enforces word/citation/reference submission rules
  output/writers.py             Writes the two official submission files
  validate.py                  Offline checks against the track's validation rules
  pipeline.py                  RagPipeline orchestrator (stage order + depths)
  factory.py                   Builds a RagPipeline from configs/pipeline.yaml
scripts/
  run_pipeline.py              CLI: topics -> submission files
  validate_submission.py       CLI: validate submission files
  train_ltr.py                 CLI: train the LTR LightGBM booster from a qrels file
tests/                         unittest suite (network-free; fakes/hashing backends only)
```

## Setup

```bash
conda create -y -n trec-rag-2026 python=3.11 && conda activate trec-rag-2026
pip install -e ".[neural,ltr]"
# torch must match the local driver; on CUDA 12.9 boxes:
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

`pip install -e .` alone (requests + PyYAML + numpy) is enough to run the
zero-setup fallback config.

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
| `configs/pipeline.gpu.yaml` | The submission config. Real e5 biencoder + cross-encoder on GPU, Ollama-backed listwise reranking and answer generation, and on-disk caching of Pyserini responses. |

## Running the pipeline

```bash
python scripts/run_pipeline.py \
  --topics trec_rag_2026_test_query_set.tsv \
  --config configs/pipeline.gpu.yaml \
  --team-id iris --run-id bruteforce-cascade \
  --out-dir output/
```

Run from the repo root — `.env.local` is resolved relative to the working
directory. Use `--workers 1` with GPU backends: each neural stage holds one
model instance and concurrent forward passes across threads are not safe.

For the full 119 topics use a detached session (the run takes roughly 1.5–2
hours, dominated by listwise reranking):

```bash
byobu new-session -d -s trecrag 'bash -lc "
  source ~/anaconda3/etc/profile.d/conda.sh && conda activate trec-rag-2026 && \
  cd /media/12TB/shared/shared_projects/trec-rag-2026 && \
  python scripts/run_pipeline.py \
    --topics trec_rag_2026_test_query_set.tsv \
    --config configs/pipeline.gpu.yaml \
    --team-id iris --run-id bruteforce-cascade \
    --run-desc \"BM25+e5 biencoder RRF -> LTR -> cross-encoder -> LLM listwise -> LLM generator\" \
    --out-dir output/ --workers 1"'
byobu attach -t trecrag        # detach with F6
```

### Resuming

The run is resumable at two independent layers, so an interruption costs
minutes rather than the whole run:

- **Per-topic checkpoints** in `<out-dir>/checkpoints/` (one atomic JSON per
  topic). Re-running the exact same command skips every finished topic and
  reassembles the submission files from the checkpoints — no models loaded, no
  LLM calls. A topic that raises is logged and skipped rather than aborting the
  run, and the script exits non-zero listing any topic missing from the output.
- **Pyserini response cache** under `pyserini.cache_dir`, keyed by
  (index, query, hits), so retrieval never re-hits the rate-limited shared API.

Pass `--no-resume` to force recomputation, or delete the relevant directory.
Logs are teed to `<out-dir>/run_<run-id>.log` by default (`--log-file` to
override).

Writes `output/r_output_trec_rag_2026.tsv` and `output/rag_output_trec_rag_2026.jsonl`
(filenames come from `configs/pipeline.yaml`). Use `--task retrieval|rag|both`,
`--limit N` to smoke-test on the first N topics, and `--workers N` for
concurrent (network-bound) topic processing.

## Validating a submission

```bash
python scripts/validate_submission.py \
  --retrieval output/r_output_trec_rag_2026.tsv \
  --rag output/rag_output_trec_rag_2026.jsonl \
  --topics trec_rag_2026_test_query_set.tsv
```

Checks: 6-column run format, ranks restarting at 1 and non-increasing scores
per topic, full topic coverage, RAG metadata shape, citation bounds
(≤3/sentence, valid indices), no orphan references, ≤1024-word answers, and
a best-effort scan for token-like secrets in the output files.

## Training the LTR stage

The default LTR backend is a dependency-free linear fallback. To train a real
LightGBM LambdaMART booster against a qrels file (e.g. one of the RAG25 dev
UMBRELA qrels in the sibling `trec-rag-data` repo):

```bash
pip install -e ".[ltr]"
python scripts/train_ltr.py \
  --topics <dev-topics.tsv> --qrels <dev.qrels> \
  --output-model models/ltr_lambdamart.txt
```

Then set `stages.ltr.model_path` in `configs/pipeline.yaml` to the saved model.

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

- `stages.ltr.model_path` ships unset (linear fallback) in both configs; train
  one with `scripts/train_ltr.py` against the RAG25 dev UMBRELA qrels for a real
  learned ranker.
- `stages.colbert.backend` stays on the lexical fallback even in the GPU config.
  No ColBERT checkpoint is cached locally, and loading `colbert-ir/colbertv2.0`
  through `AutoModel` would skip its projection head — i.e. score raw BERT
  hidden states and call it ColBERT. The stage only prunes 200 -> 100 ahead of
  the real cross-encoder, so the fallback is the honest choice until a proper
  late-interaction backend is wired.
- **Retrieval depth is effectively 20, not `retrieval_output.depth`.** The
  cascade ends at `stages.llm_listwise.top_k`, so the `R` run gets whatever that
  is; `depth: 100` only ever truncates. The track wants a variable per-narrative
  `k` of "all and only" useful documents, so decide that number deliberately
  rather than inheriting it from the last rerank stage.
- Submission logistics (upload portal) are still unconfirmed upstream; the
  deadline is Aug 8 2026 per the track guidelines skill.
