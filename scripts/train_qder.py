#!/usr/bin/env python3
"""Train a QDER checkpoint end to end: build training data from this
pipeline's own retrieval cascade, train via the vendored SIGIR2025-QDER
trainer at `third_party/qder_training/train.py`, and unwrap the resulting
checkpoint so it's ready to drop into `stages.qder.model_path`.

Defaults to the real TREC RAG 2026 dev data already in this repo
(`data/dev/`) -- the official 22-topic dev set with UMBRELA qrels, copied
verbatim from the upstream `TREC-RAG/trec-rag-data` GitHub repo. That is the
only labeled RAG26 data that exists anywhere right now: the 119-topic test
query set has no qrels at all and cannot be used for training, and there is
no larger official labeled set to substitute. Because 22 topics is small,
this script splits them into an internal train/dev split for the trainer's
own early stopping (`--dev-fraction`, default 0.2) rather than expecting a
separate held-out file.

This is a thin orchestrator over two already-standalone pieces -- run them
separately (see their own --help) if you want to inspect the intermediate
JSONL, try a different qrels judge, or rerun training without re-paying the
retrieval cascade's API cost (--skip-data-prep):
  1. scripts/prepare_qder_data.py       -- rebuilds the live candidate pool
     (BM25 -> biencoder -> RRF -> LTR -> ColBERT) and labels it from qrels.
  2. third_party/qder_training/train.py -- the vendored SIGIR2025-QDER trainer.

Example (all defaults -- CPU, hashing-fallback pool reconstruction):
    python scripts/train_qder.py --out-dir models/qder

Example (GPU pipeline config, GPU training, more epochs):
    python scripts/train_qder.py --out-dir models/qder \\
        --config configs/pipeline.gpu.yaml --use-cuda --epoch 40

Example (skip re-preparing data after a training crash):
    python scripts/train_qder.py --out-dir models/qder --skip-data-prep --use-cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOPICS = REPO_ROOT / "data/dev/topics/rag25-topics-dev.tsv"
DEFAULT_QRELS = (
    REPO_ROOT
    / "data/dev/rag25-dev-umbrela-qrels"
    / "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)
PREPARE_SCRIPT = REPO_ROOT / "scripts/prepare_qder_data.py"
TRAINER_SCRIPT = REPO_ROOT / "third_party/qder_training/train.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics", default=str(DEFAULT_TOPICS), help="Dev topics TSV (default: the real TREC RAG 2026 dev set, 22 topics).")
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS), help="Qrels for --topics (default: the codex-gpt5.5 UMBRELA judgments).")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline config used to reconstruct the candidate pool.")
    parser.add_argument("--out-dir", required=True, help="Where the prepared JSONL, checkpoint, and unwrapped weights are written.")
    parser.add_argument("--dev-fraction", type=float, default=0.2, help="Internal train/dev split of --topics (there is no separate held-out dev file).")
    parser.add_argument("--label-mode", choices=["qrels-only", "llm-judge"], default="qrels-only")
    parser.add_argument("--skip-data-prep", action="store_true", help="Reuse an existing <out-dir>/qder_{train,dev}.jsonl instead of rebuilding the pool.")
    parser.add_argument("--text-enc", default="bert", choices=["bert", "roberta", "deberta"])
    parser.add_argument("--score-method", default="bilinear", choices=["linear", "bilinear"])
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument(
        "--use-entities",
        action="store_true",
        help=(
            "Train WITH the entity channel. Off by default to match "
            "stages.qder.use_entities: false -- there is no real "
            "Wikipedia2Vec/WAT entity linker for ClimbMix here, so a "
            "checkpoint trained with entities on cannot be served correctly "
            "by this pipeline's HeuristicEntityProvider."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _require_input_file(path_str: str, default_path: Path, label: str, logger) -> None:
    """Warn and stop if an input file is missing, instead of failing deep inside a subprocess."""
    path = Path(path_str)
    if path.is_file():
        return
    if str(path) == str(default_path):
        logger.warning(
            "no %s found at the default path %s -- data/dev/ looks like it's been "
            "cleared or rebuilt with different filenames. Pass --topics/--qrels "
            "explicitly to point at the new files.",
            label, path,
        )
    else:
        logger.warning("no %s found at %s", label, path)
    raise SystemExit(1)


def main() -> None:
    args = parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("train_qder.py requires torch: pip install torch") from exc

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from trecrag.logging_utils import configure_logging, get_logger

    configure_logging(args.log_level)
    logger = get_logger("train_qder")

    # qrels is used both for data-prep labeling and the trainer's own --qrels
    # (validation metric), so it's required in both modes; topics is only
    # consumed by data-prep.
    _require_input_file(args.qrels, DEFAULT_QRELS, "qrels file", logger)
    if not args.skip_data_prep:
        _require_input_file(args.topics, DEFAULT_TOPICS, "topics file", logger)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "qder_train.jsonl"
    dev_jsonl = out_dir / "qder_dev.jsonl"

    if args.skip_data_prep:
        if not train_jsonl.is_file() or not dev_jsonl.is_file():
            raise SystemExit(f"--skip-data-prep set but {train_jsonl} and/or {dev_jsonl} don't exist yet")
        logger.info("Step 1/3: skipped, reusing %s / %s", train_jsonl, dev_jsonl)
    else:
        logger.info("Step 1/3: building training data from %s (%s)", args.topics, args.label_mode)
        subprocess.run(
            [
                sys.executable, str(PREPARE_SCRIPT),
                "--topics", args.topics,
                "--qrels", args.qrels,
                "--config", args.config,
                "--dev-fraction", str(args.dev_fraction),
                "--label-mode", args.label_mode,
                "--output-train", str(train_jsonl),
                "--output-dev", str(dev_jsonl),
                "--log-level", args.log_level,
            ],
            check=True,
            cwd=REPO_ROOT,
        )

    if args.use_entities:
        logger.warning(
            "training WITH entities: this pipeline has no real entity linker "
            "for ClimbMix, so HeuristicEntityProvider's hashed vectors at "
            "inference will not match whatever embeddings this run trains "
            "against, even though tensor shapes will line up."
        )

    logger.info("Step 2/3: training via %s", TRAINER_SCRIPT)
    train_cmd = [
        sys.executable, str(TRAINER_SCRIPT),
        "--train", str(train_jsonl),
        "--dev", str(dev_jsonl),
        "--qrels", args.qrels,
        "--save-dir", str(out_dir),
        "--save", "model.bin",
        "--text-enc", args.text_enc,
        "--score-method", args.score_method,
        "--epoch", str(args.epoch),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--patience", str(args.patience),
    ]
    if args.use_cuda:
        train_cmd.append("--use-cuda")
    subprocess.run(train_cmd, check=True, cwd=REPO_ROOT)

    logger.info("Step 3/3: unwrapping checkpoint")
    checkpoint_path = out_dir / "model.bin"
    if not checkpoint_path.is_file():
        logger.warning(
            "training finished but no checkpoint was written to %s -- "
            "nothing to unwrap. Check the trainer's own output above for "
            "what went wrong (e.g. an empty dev/train split).",
            checkpoint_path,
        )
        raise SystemExit(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    weights_path = out_dir / "model_weights_only.bin"
    torch.save(state_dict, weights_path)
    logger.info("done -- unwrapped QDER weights written to %s", weights_path)

    print(
        f"""
Trained QDER checkpoint ready in {out_dir} -- use it directly from there, no
need to move or copy anything anywhere else. Point a config at it:

  stages:
    qder:
      enabled: true
      backend: transformer
      model_name: "bert-base-uncased"   # must match --text-enc {args.text_enc}
      model_path: "{weights_path}"
      score_method: {args.score_method}
      use_entities: {"true" if args.use_entities else "false"}
"""
    )


if __name__ == "__main__":
    main()
