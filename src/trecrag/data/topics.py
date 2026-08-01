"""Loader for `trec_rag_2026_queries.tsv`-style topic files.

Format (per the track guidelines): two tab-separated columns per line,
`topic_id \t topic_text`. Topic IDs and text must be preserved exactly.
"""

from __future__ import annotations

from pathlib import Path

from trecrag.schema import Topic


class TopicFormatError(ValueError):
    pass


def load_topics(path: str | Path) -> list[Topic]:
    topics: list[Topic] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise TopicFormatError(
                    f"{path}:{line_no}: expected 2 tab-separated columns, got {len(parts)}"
                )
            topic_id, text = parts
            topic_id = topic_id.strip()
            if not topic_id:
                raise TopicFormatError(f"{path}:{line_no}: empty topic_id")
            if topic_id in seen:
                raise TopicFormatError(f"{path}:{line_no}: duplicate topic_id {topic_id!r}")
            seen.add(topic_id)
            topics.append(Topic(topic_id=topic_id, text=text))
    return topics
