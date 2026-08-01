# QDER training driver (vendored)

Everything in this directory is copied **unmodified** from the `src/doc_ranking/`
subtree of [SIGIR2025-QDER](https://github.com/shubham526/SIGIR2025-QDER)
(Chatterjee & Dalton, SIGIR 2025) — see `LICENSE` (MIT, Copyright Shubham
Chatterjee) for the required notice.

`src/trecrag/rerank/qder/_vendor/` also vendors `base_model.py`/`qder_model.py`/
`text_embedding.py` from the same upstream source, adjusted to import as part
of the `trecrag` package (proper package-relative imports) for use at
inference time. The copies in **this** directory are deliberately left with
their original bare imports (`import dataset`, `from qder_model import
QDERModel`, etc.) unchanged, because that's what `train.py` needs: run it
directly and Python puts this directory on `sys.path[0]` automatically, so
the bare imports resolve exactly as they do upstream. The duplication is
intentional — trying to share one copy across both a package-relative import
context (trecrag itself) and a bare-script context (standalone training)
would be fragile for no real benefit.

## Usage

Build training/dev data first with `scripts/prepare_qder_data.py` from the
repo root (see its `--help`), then:

```bash
pip install torch transformers tqdm pytrec_eval   # not part of trecrag's own dependency groups

python third_party/qder_training/train.py \
  --train data/qder_train.jsonl --dev data/qder_dev.jsonl \
  --qrels data/dev/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \
  --save-dir output/qder --text-enc bert --score-method bilinear --use-cuda
```

This saves a **wrapped** checkpoint (`model_state_dict` + optimizer/scheduler
state + training metadata) to `<save-dir>/model.bin`. `trecrag`'s own
`TransformerQDERScorer` (`src/trecrag/rerank/qder/reranker.py`) expects a raw
state dict when loading `stages.qder.model_path`, so unwrap it first:

```python
import torch
ckpt = torch.load("output/qder/model.bin", map_location="cpu")
torch.save(ckpt["model_state_dict"], "output/qder/model_weights_only.bin")
```

Then set `stages.qder.model_path: output/qder/model_weights_only.bin` in your
pipeline config. `use_entities`, `score_method`, and the text encoder
(`--text-enc`) must match exactly between training and inference — they
determine the model's actual parameter shapes, so a mismatch fails to load
outright rather than degrading quietly.

This repo has no real WAT entity linker or Wikipedia2Vec table for ClimbMix
(see `src/trecrag/rerank/qder/entities.py`), so `train.py`'s `--use-entities`
flag defaults to **off** (a deliberate deviation from upstream, where
`QDERModel(use_entities=...)` was hardcoded `True` in `main()`) — leave it off
to pair with trecrag's own `use_entities: false` default. Passing
`--use-entities` anyway trains a checkpoint whose entity-channel weights are
meaningless to trecrag's `NullEntityProvider`, and worse, loading it will
fail outright on a shape mismatch since inference builds the model without an
entity channel at all unless `stages.qder.use_entities: true` is also set.

`scripts/train_qder.py` (repo root) wraps the whole build-data →
train → unwrap-checkpoint flow above into one command — see its `--help`.
