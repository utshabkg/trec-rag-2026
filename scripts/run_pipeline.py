#!/usr/bin/env python3
"""Run the full TREC RAG 2026 pipeline over a topics file and write the
official submission files (`r_output_trec_rag_2026.tsv` / `rag_output_trec_rag_2026.jsonl`).

Example:
    python scripts/run_pipeline.py \\
        --topics trec_rag_2026_test_query_set.tsv \\
        --team-id my-team --run-id my-rag-run \\
        --out-dir output/
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trecrag.config import load_dotenv_local, load_pipeline_config
from trecrag.data.topics import load_topics
from trecrag.factory import build_pipeline_from_config
from trecrag.logging_utils import configure_logging, get_logger
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    load_dotenv_local()

    config = load_pipeline_config(args.config)
    topics = load_topics(args.topics)
    if args.limit is not None:
        topics = topics[: args.limit]
    logger.info("Loaded %d topics from %s", len(topics), args.topics)

    pipeline = build_pipeline_from_config(config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_results: dict[str, list] = {}
    rag_records: list[dict] = []
    retrieval_depth = int(config["retrieval_output"]["depth"])

    def process(topic: Topic):
        logger.info("Running topic %s", topic.topic_id)
        return topic, pipeline.run_topic(topic)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for topic, (ranked, rag_answer) in pool.map(process, topics):
            if args.task in ("retrieval", "both"):
                retrieval_results[topic.topic_id] = ranked[:retrieval_depth]
            if args.task in ("rag", "both"):
                rag_records.append(
                    build_rag_record(args.team_id, topic, args.run_id, args.run_desc, rag_answer)
                )

    if args.task in ("retrieval", "both"):
        retrieval_path = out_dir / config["retrieval_output"]["filename"]
        write_retrieval_run(retrieval_path, args.run_id, retrieval_results)
        logger.info("Wrote retrieval run: %s (%d topics)", retrieval_path, len(retrieval_results))

    if args.task in ("rag", "both"):
        rag_path = out_dir / config["generation"]["filename"]
        write_rag_jsonl(rag_path, rag_records)
        logger.info("Wrote RAG output: %s (%d topics)", rag_path, len(rag_records))


if __name__ == "__main__":
    main()
