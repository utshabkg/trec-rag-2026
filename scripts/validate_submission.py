#!/usr/bin/env python3
"""Validate submission files against the TREC RAG 2026 track guidelines'
validation rules. Exits non-zero if any violation is found.

Example:
    python scripts/validate_submission.py \\
        --retrieval output/r_output_trec_rag_2026.tsv \\
        --rag output/rag_output_trec_rag_2026.jsonl \\
        --topics trec_rag_2026_test_query_set.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trecrag.data.topics import load_topics
from trecrag.logging_utils import configure_logging, get_logger
from trecrag.validate import validate_rag_jsonl, validate_retrieval_run

logger = get_logger("validate_submission")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", help="Path to r_output_trec_rag_2026.tsv")
    parser.add_argument("--rag", help="Path to rag_output_trec_rag_2026.jsonl")
    parser.add_argument("--topics", help="Topics TSV used to check full topic coverage")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.retrieval and not args.rag:
        print("Nothing to validate: pass --retrieval and/or --rag", file=sys.stderr)
        raise SystemExit(2)

    topics = load_topics(args.topics) if args.topics else None
    topic_ids = {t.topic_id for t in topics} if topics else None

    violations: list[str] = []

    if args.retrieval:
        retrieval_violations = validate_retrieval_run(args.retrieval, expected_topic_ids=topic_ids)
        logger.info("%s: %d violation(s)", args.retrieval, len(retrieval_violations))
        violations.extend(f"[retrieval] {v}" for v in retrieval_violations)

    if args.rag:
        rag_violations = validate_rag_jsonl(args.rag, topics=topics)
        logger.info("%s: %d violation(s)", args.rag, len(rag_violations))
        violations.extend(f"[rag] {v}" for v in rag_violations)

    if violations:
        print(f"\n{len(violations)} violation(s) found:\n")
        for v in violations:
            print(f"  - {v}")
        raise SystemExit(1)

    print("All checks passed.")


if __name__ == "__main__":
    main()
