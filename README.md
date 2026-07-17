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

## Running the pipeline

```bash
python scripts/run_pipeline.py \
  --topics trec_rag_2026_test_query_set.tsv \
  --team-id my-team --run-id my-rag-run \
  --out-dir output/
```

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

- `stages.ltr.model_path` ships unset (linear fallback); train one with
  `scripts/train_ltr.py` against dev qrels for a real learned ranker.
- `stages.colbert.backend` / `stages.cross_encoder.backend` /
  `stages.llm_listwise.backend` default to the dependency-free fallbacks;
  flip to `transformer` / `sentence_transformer` / `openai`|`anthropic`
  once compute/API budget is available, per `configs/pipeline.yaml`.
- Submission logistics (upload portal, exact deadline) are still unconfirmed
  upstream — see `../plans/trec-rag-2026-participation-guide.md` §6–7 before
  the real submission.
