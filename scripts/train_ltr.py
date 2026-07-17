#!/usr/bin/env python3
"""Train the LTR stage's LightGBM LambdaMART booster from a topics file and a
standard TREC qrels file (e.g. one of the RAG25 development UMBRELA qrels in
the sibling `trec-rag-data` repo).

This only trains on the BM25 + biencoder + hybrid-fusion features (the
signals available before the LTR stage runs); it does not require ColBERT,
the cross-encoder, or an LLM. Requires `pip install trecrag[ltr]`.

Example:
    python scripts/train_ltr.py \\
        --topics ../Official/trec-rag-data/trec-rag-2026/development-data/topics/rag25-topics-dev.tsv \\
        --qrels ../Official/trec-rag-data/trec-rag-2026/development-data/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \\
        --output-model models/ltr_lambdamart.txt
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from trecrag.config import load_dotenv_local, load_pipeline_config
from trecrag.data.topics import load_topics
from trecrag.factory import build_biencoder, build_pyserini_client
from trecrag.logging_utils import configure_logging, get_logger
from trecrag.retrieval.bm25 import BM25Retriever
from trecrag.retrieval.hybrid import reciprocal_rank_fusion

logger = get_logger("train_ltr")

DEFAULT_FEATURE_NAMES = ["bm25_score", "bm25_rank", "bien_score", "bien_rank", "rrf_score"]


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            topic_id, _iteration, docid, grade = parts
            qrels[topic_id][docid] = int(grade)
    return qrels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--feature-names", nargs="+", default=DEFAULT_FEATURE_NAMES)
    parser.add_argument("--num-boost-round", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit("train_ltr.py requires lightgbm: pip install trecrag[ltr]") from exc

    configure_logging(args.log_level)
    load_dotenv_local()

    config = load_pipeline_config(args.config)
    topics = load_topics(args.topics)
    qrels = load_qrels(args.qrels)

    client = build_pyserini_client(config)
    bm25 = BM25Retriever(client)
    biencoder = build_biencoder(config["stages"].get("biencoder", {}))
    hybrid_cfg = config["stages"].get("hybrid", {})

    rows: list[list[float]] = []
    labels: list[int] = []
    group_sizes: list[int] = []

    for topic in topics:
        topic_qrels = qrels.get(topic.topic_id)
        if not topic_qrels:
            logger.warning("no qrels for topic %s, skipping", topic.topic_id)
            continue

        bm25_candidates = bm25.retrieve(topic, int(config["stages"]["bm25"]["hits"]))
        ranked_lists = [bm25_candidates]
        if biencoder is not None:
            biencoder_candidates = biencoder.rescore(
                topic, bm25_candidates, int(config["stages"]["bm25"]["hits"])
            )
            if biencoder_candidates:
                ranked_lists.append(biencoder_candidates)

        fused = reciprocal_rank_fusion(
            ranked_lists,
            k_const=int(hybrid_cfg.get("rrf_k", 60)),
            top_k=int(hybrid_cfg.get("top_k", 500)),
        )
        judged = [c for c in fused if c.docid in topic_qrels]
        if len(judged) < 2:
            logger.warning("topic %s: fewer than 2 judged candidates in the pool, skipping", topic.topic_id)
            continue

        for candidate in judged:
            rows.append([candidate.features.get(name, 0.0) for name in args.feature_names])
            labels.append(topic_qrels[candidate.docid])
        group_sizes.append(len(judged))
        logger.info("topic %s: %d judged candidates in pool", topic.topic_id, len(judged))

    if not rows:
        raise SystemExit("no training rows collected; check --topics/--qrels overlap")

    train_set = lgb.Dataset(np.asarray(rows, dtype=np.float32), label=np.asarray(labels), group=group_sizes)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "label_gain": list(range(max(labels) + 1)),
        "verbosity": -1,
    }
    booster = lgb.train(params, train_set, num_boost_round=args.num_boost_round)

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))
    logger.info("saved LTR model to %s (%d rows, %d groups)", output_path, len(rows), len(group_sizes))


if __name__ == "__main__":
    main()
