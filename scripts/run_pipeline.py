#!/usr/bin/env python3
"""Run the full TREC RAG 2026 pipeline over a topics file and write the
official submission files (`r_output_trec_rag_2026.tsv` / `rag_output_trec_rag_2026.jsonl`).

The run is resumable: every finished topic is checkpointed to `--work-dir`, and
re-invoking the same command skips the topics already done. A topic that raises
is logged and skipped rather than killing the run, so one bad topic never costs
the other 118; re-run to retry just those.

Example:
    python scripts/run_pipeline.py \\
        --topics trec_rag_2026_test_query_set.tsv \\
        --config configs/pipeline.gpu.yaml \\
        --team-id iris-mst --run-id bruteforce-cascade \\
        --out-dir output/
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trecrag.config import load_dotenv_local, load_pipeline_config
from trecrag.data.topics import load_topics
from trecrag.diagnostics import StageDiagnostics
from trecrag.factory import build_pipeline_from_config
from trecrag.logging_utils import configure_logging, get_logger
from trecrag.output.checkpoint import TopicCheckpointStore
from trecrag.output.writers import build_rag_record, write_rag_jsonl, write_retrieval_run
from trecrag.schema import Topic

logger = get_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", required=True, help="Path to a topics TSV file")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config YAML")
    parser.add_argument("--out-dir", default="output", help="Directory to write submission files into")
    parser.add_argument("--team-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-desc", default="BM25+Biencoder -> LTR -> ColBERT -> CE -> LLM listwise -> Generator")
    parser.add_argument("--task", choices=["retrieval", "rag", "both"], default="both")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N topics")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent topics (network-bound stages)")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Per-topic checkpoint directory (default: <out-dir>/checkpoints)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute every topic even if a checkpoint exists",
    )
    parser.add_argument("--log-file", default=None, help="Tee logs to this file "
                        "(default: logs/<run-id>_<timestamp>.log)")
    parser.add_argument("--log-dir", default="logs", help="Directory for run + stage logs")
    parser.add_argument("--no-diagnostics", action="store_true",
                        help="Skip per-stage diagnostics JSONL")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logs live in the repo (logs/), not beside the submission artifacts, so a
    # run's history survives deleting or re-creating an output directory.
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = args.log_file or (log_dir / f"{args.run_id}_{stamp}.log")
    configure_logging(args.log_level, log_file=log_file)
    load_dotenv_local()

    config = load_pipeline_config(args.config)
    topics = load_topics(args.topics)
    if args.limit is not None:
        topics = topics[: args.limit]

    diagnostics = StageDiagnostics(
        None if args.no_diagnostics else log_dir / f"stages_{args.run_id}_{stamp}.jsonl"
    )
    store = TopicCheckpointStore(args.work_dir or (out_dir / "checkpoints"))
    pending = [t for t in topics if args.no_resume or not store.has(t.topic_id)]
    logger.info(
        "Loaded %d topics from %s | %d already checkpointed | %d to run | config=%s",
        len(topics), args.topics, len(topics) - len(pending), len(pending), args.config,
    )

    failures: list[str] = []
    started = time.time()
    done = 0

    def process(topic: Topic) -> None:
        nonlocal done
        topic_started = time.time()
        try:
            ranked, rag_answer = pipeline.run_topic(topic)
        except Exception as exc:  # one bad topic must not end the run
            failures.append(topic.topic_id)
            logger.error("Topic %s FAILED (%s: %s)", topic.topic_id, type(exc).__name__, exc)
            return
        store.save(topic.topic_id, ranked, rag_answer)
        diagnostics.flush()  # per topic, so a killed run keeps what it measured
        done += 1
        elapsed = time.time() - started
        rate = elapsed / max(done, 1)
        logger.info(
            "Topic %s done in %.1fs (%d/%d, ~%.0f min left)",
            topic.topic_id, time.time() - topic_started, done, len(pending),
            max(0.0, (len(pending) - done) * rate / 60.0),
        )

    if pending:
        pipeline = build_pipeline_from_config(config, diagnostics=diagnostics)
        if args.workers > 1:
            logger.warning(
                "--workers %d: neural stages share one model instance per stage; "
                "concurrent forward passes are not thread-safe. Use --workers 1 with "
                "GPU backends.", args.workers,
            )
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(process, pending))

    # Assemble submissions from checkpoints, in canonical topic order regardless
    # of the order topics actually completed in across runs.
    retrieval_results: dict[str, list] = {}
    rag_records: list[dict] = []
    missing: list[str] = []
    retrieval_depth = int(config["retrieval_output"]["depth"])

    for topic in topics:
        loaded = store.load(topic.topic_id)
        if loaded is None:
            missing.append(topic.topic_id)
            continue
        ranked, rag_answer = loaded
        if args.task in ("retrieval", "both"):
            retrieval_results[topic.topic_id] = ranked[:retrieval_depth]
        if args.task in ("rag", "both"):
            rag_records.append(
                build_rag_record(args.team_id, topic, args.run_id, args.run_desc, rag_answer)
            )

    # A partial run must never overwrite a complete submission. --limit slices
    # `topics`, so `missing` is computed over the slice and comes back empty --
    # a 5-topic smoke test would otherwise truncate a finished 119-topic
    # submission in place and exit 0 reporting success.
    all_topic_ids = {t.topic_id for t in load_topics(args.topics)}
    is_complete = not missing and {t.topic_id for t in topics} == all_topic_ids
    suffix = "" if is_complete else ".partial"
    if not is_complete:
        logger.warning(
            "Partial run (%d of %d narratives): writing '*%s' instead of the official "
            "filenames so a complete submission is never overwritten.",
            len(topics), len(all_topic_ids), suffix,
        )

    if args.task in ("retrieval", "both"):
        retrieval_path = out_dir / (config["retrieval_output"]["filename"] + suffix)
        write_retrieval_run(retrieval_path, args.run_id, retrieval_results)
        logger.info("Wrote retrieval run: %s (%d topics)", retrieval_path, len(retrieval_results))

    if args.task in ("rag", "both"):
        rag_path = out_dir / (config["generation"]["filename"] + suffix)
        write_rag_jsonl(rag_path, rag_records)
        logger.info("Wrote RAG output: %s (%d topics)", rag_path, len(rag_records))

    if failures:
        logger.error("%d topic(s) failed this run: %s", len(failures), ", ".join(failures))
    if missing:
        logger.error(
            "%d of %d topic(s) have no result and are ABSENT from the submission: %s. "
            "Re-run the same command to retry them.",
            len(missing), len(topics), ", ".join(missing[:10]),
        )
        raise SystemExit(1)

    logger.info("All %d topics present. Elapsed %.1f min.", len(topics), (time.time() - started) / 60.0)


if __name__ == "__main__":
    main()
