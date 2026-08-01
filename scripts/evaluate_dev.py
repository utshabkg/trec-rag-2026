#!/usr/bin/env python3
"""Score the retrieval cascade on the RAG25 dev topics against UMBRELA qrels.

This is the missing closed loop. Every quality judgement in this project so far
has been inspection, not measurement, which is why config values that need
evidence -- the fusion weight for an untrained QDER, BM25 k1/b, stage depths --
have had to ship at their safe defaults.

    python scripts/evaluate_dev.py --config configs/pipeline.gpu.yaml
    python scripts/evaluate_dev.py --ablate stages.fusion.feature_weights.qder_score=0,0.5,1.0

Rankings are cached per (config-signature, topic), so the first run pays the
Pyserini calls and every later comparison is free. That matters here: none of
the 22 dev topics are in the response cache, and uncached hits=1000 searches
over them are exactly what triggered the 429 storm that killed train_ltr.py.
An ablation therefore issues its API calls once, on the first variant, and
reuses them for the rest.

READ `judged@k` FIRST. These qrels were pooled from other systems' runs, so a
document we retrieve may be unjudged rather than irrelevant, and unjudged scores
as non-relevant. Comparisons between our own configurations remain meaningful;
absolute numbers are a lower bound.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trecrag.config import load_dotenv_local, load_pipeline_config
from trecrag.data.topics import load_topics
from trecrag.eval import evaluate_run, load_qrels
from trecrag.factory import build_pipeline_from_config
from trecrag.logging_utils import configure_logging, get_logger

logger = get_logger("evaluate_dev")

DEV_TOPICS = "data/dev/topics/rag25-topics-dev.tsv"
DEV_QRELS = (
    "data/dev/rag25-dev-umbrela-qrels/"
    "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels"
)


def set_by_path(config: dict, dotted: str, value):
    """Set `stages.fusion.feature_weights.qder_score` style paths for ablations."""
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(f"{dotted}: no such config key {key!r}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"{dotted}: no such config key {keys[-1]!r}")
    node[keys[-1]] = value


def _coerce(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def config_signature(config: dict) -> str:
    """Hash of everything that can change a ranking.

    Deliberately excludes the generation section: this evaluates retrieval, so
    two configs differing only in answer generation should share a cache entry.
    """
    relevant = {"pyserini": config.get("pyserini"), "stages": config.get("stages")}
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_retrieval(config: dict, topics, cache_dir: Path) -> dict[str, list[str]]:
    signature = config_signature(config)
    cache_path = cache_dir / f"ranking_{signature}.json"
    if cache_path.is_file():
        logger.info("reusing cached rankings (%s)", cache_path.name)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline_from_config(config)
    run: dict[str, list[str]] = {}
    started = time.time()
    for i, topic in enumerate(topics, start=1):
        try:
            run[topic.topic_id] = [c.docid for c in pipeline.retrieve(topic)]
        except Exception as exc:  # one topic must not lose the whole sweep
            logger.error("topic %s FAILED (%s: %s)", topic.topic_id, type(exc).__name__, exc)
            continue
        # Persist as we go: these calls are expensive and rate-limited.
        cache_path.write_text(json.dumps(run), encoding="utf-8")
        logger.info("%s (%d/%d, %.0fs elapsed)", topic.topic_id, i, len(topics), time.time() - started)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pipeline.gpu.yaml")
    parser.add_argument("--topics", default=DEV_TOPICS)
    parser.add_argument("--qrels", default=DEV_QRELS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", default="logs/dev_eval")
    parser.add_argument(
        "--set", action="append", default=[],
        help="override a config value, e.g. --set stages.fusion.top_k=100",
    )
    parser.add_argument(
        "--ablate", default=None,
        help="sweep one key: stages.fusion.feature_weights.qder_score=0,0.5,1.0",
    )
    # Default 3, not 1. These qrels are graded 0-4 over a large pool: at grade>=1
    # the median topic has 975 relevant documents, so recall@50 is capped at 5.1%
    # and cannot discriminate between configurations. Measured ceilings:
    #   grade>=1  975 rel/topic   recall@50 <= 5.1%
    #   grade>=2  598             <= 8.4%
    #   grade>=3  256             <= 19.6%
    #   grade>=4   92             <= 54.1%, but 2 topics have no relevant doc at all
    # grade>=3 keeps every topic scoreable while leaving recall room to move.
    # nDCG is unaffected by this flag -- it weights by grade directly.
    parser.add_argument("--relevant-grade", type=int, default=3,
                        help="minimum UMBRELA grade counted relevant for recall/MAP (default 3)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    load_dotenv_local()

    base = load_pipeline_config(args.config)
    for override in args.set:
        key, _, raw = override.partition("=")
        set_by_path(base, key, _coerce(raw))

    topics = load_topics(args.topics)
    if args.limit:
        topics = topics[: args.limit]
    qrels = load_qrels(args.qrels)
    logger.info("%d dev topics | %d topics in qrels", len(topics), len(qrels))

    variants: list[tuple[str, dict]] = []
    if args.ablate:
        key, _, values = args.ablate.partition("=")
        for raw in values.split(","):
            config = copy.deepcopy(base)
            set_by_path(config, key, _coerce(raw))
            variants.append((f"{key.split('.')[-1]}={raw}", config))
    else:
        variants.append(("as-configured", base))

    rows = []
    for label, config in variants:
        logger.info("--- %s ---", label)
        run = run_retrieval(config, topics, Path(args.cache_dir))
        metrics = evaluate_run(run, qrels, relevant_grade=args.relevant_grade)
        rows.append((label, metrics))

    columns = ["ndcg@10", "ndcg@50", "recall@10", "recall@50", "map", "judged@10", "judged@50"]
    width = max(len(label) for label, _ in rows) + 2
    print()
    print(f"{'variant':{width}s}" + "".join(f"{c:>11s}" for c in columns) + f"{'topics':>8s}")
    print("-" * (width + 11 * len(columns) + 8))
    for label, m in rows:
        print(
            f"{label:{width}s}"
            + "".join(f"{m.get(c, float('nan')):>11.4f}" for c in columns)
            + f"{int(m.get('topics', 0)):>8d}"
        )
    print()
    worst_judged = min((m.get("judged@50", 0.0) for _, m in rows), default=0.0)
    if worst_judged < 0.5:
        print(
            f"NOTE: judged@50 is {worst_judged:.2f} -- most retrieved documents were never\n"
            f"      assessed, so these scores are a LOWER BOUND, not a measurement.\n"
            f"      Comparisons between variants remain meaningful; absolute values do not."
        )


if __name__ == "__main__":
    main()
