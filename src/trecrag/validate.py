"""Validation checks mirroring the TREC RAG 2026 track guidelines' validation
rules for `r_output_trec_rag_2026.tsv` and `rag_output_trec_rag_2026.jsonl`.

Each function returns a list of human-readable violation strings; an empty
list means the file passed every check that can be verified offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from trecrag.logging_utils import get_logger
from trecrag.schema import Topic

_logger = get_logger("validate")

_REQUIRED_METADATA_KEYS = {"team_id", "narrative_id", "narrative", "run_id", "run_desc"}
_SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9\-_.]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"PYSERINI_API_TOKEN\s*="),
]


def _scan_for_secrets(raw_text: str) -> list[str]:
    violations = []
    for pattern in _SECRET_PATTERNS:
        if pattern.search(raw_text):
            violations.append(f"possible secret/token-like content matching {pattern.pattern!r}")
    return violations


def validate_retrieval_run(
    path: str | Path,
    expected_topic_ids: set[str] | None = None,
) -> list[str]:
    violations: list[str] = []
    raw_text = Path(path).read_text(encoding="utf-8")
    violations.extend(_scan_for_secrets(raw_text))

    seen_topic_ids: set[str] = set()
    last_rank: dict[str, int] = {}
    last_score: dict[str, float] = {}

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            violations.append(f"line {line_no}: expected 6 whitespace-separated columns, got {len(fields)}")
            continue
        topic_id, q0, docid, rank_str, score_str, _run_id = fields
        if q0 != "Q0":
            violations.append(f"line {line_no}: second column must be 'Q0', got {q0!r}")
        if not docid:
            violations.append(f"line {line_no}: empty docid")

        try:
            rank = int(rank_str)
        except ValueError:
            violations.append(f"line {line_no}: rank {rank_str!r} is not an integer")
            continue
        try:
            score = float(score_str)
        except ValueError:
            violations.append(f"line {line_no}: score {score_str!r} is not numeric")
            continue

        seen_topic_ids.add(topic_id)
        expected_rank = last_rank.get(topic_id, 0) + 1
        if rank != expected_rank:
            violations.append(
                f"line {line_no}: topic {topic_id} rank {rank} out of sequence "
                f"(expected {expected_rank}; ranks must restart at 1 and increase by 1 per topic)"
            )
        last_rank[topic_id] = rank

        if topic_id in last_score and score > last_score[topic_id]:
            violations.append(
                f"line {line_no}: topic {topic_id} score {score} increased from "
                f"previous {last_score[topic_id]} (scores must be non-increasing within a topic)"
            )
        last_score[topic_id] = score

    if expected_topic_ids is not None:
        missing = expected_topic_ids - seen_topic_ids
        if missing:
            violations.append(f"missing output for {len(missing)} topic(s): {sorted(missing)[:10]}")

    return violations


def validate_rag_jsonl(
    path: str | Path,
    topics: list[Topic] | None = None,
    max_words: int = 1024,
    max_citations_per_sentence: int = 3,
) -> list[str]:
    violations: list[str] = []
    raw_text = Path(path).read_text(encoding="utf-8")
    violations.extend(_scan_for_secrets(raw_text))

    topics_by_id = {t.topic_id: t for t in topics} if topics else None
    seen_narrative_ids: set[str] = set()

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            violations.append(f"line {line_no}: invalid JSON ({exc})")
            continue

        prefix = f"line {line_no}"
        if not isinstance(record, dict):
            violations.append(f"{prefix}: top-level JSON value must be an object")
            continue

        for required_key in ("metadata", "references", "answer"):
            if required_key not in record:
                violations.append(f"{prefix}: missing required key {required_key!r}")
        metadata = record.get("metadata", {})
        references = record.get("references", [])
        answer = record.get("answer", [])

        if isinstance(metadata, dict):
            # Extra metadata keys are explicitly permitted: "metadata: may also
            # contain any additional participant-defined fields" (rag-task.md).
            missing_keys = _REQUIRED_METADATA_KEYS - set(metadata)
            if missing_keys:
                violations.append(f"{prefix}: metadata missing keys {sorted(missing_keys)}")

            narrative_id = metadata.get("narrative_id")
            if narrative_id in seen_narrative_ids:
                violations.append(f"{prefix}: duplicate narrative_id {narrative_id!r}")
            if narrative_id is not None:
                seen_narrative_ids.add(narrative_id)

            if topics_by_id is not None:
                if narrative_id not in topics_by_id:
                    violations.append(f"{prefix}: narrative_id {narrative_id!r} not found in topic set")
                elif metadata.get("narrative") != topics_by_id[narrative_id].text:
                    violations.append(f"{prefix}: narrative text does not match topic {narrative_id!r} exactly")
        else:
            violations.append(f"{prefix}: 'metadata' must be an object")

        if not isinstance(references, list) or not all(isinstance(r, str) and r for r in references):
            violations.append(f"{prefix}: 'references' must be a list of non-empty strings")
            references = []

        if not isinstance(answer, list):
            violations.append(f"{prefix}: 'answer' must be a list")
            answer = []

        # An empty answer is syntactically valid but scores zero, and every other
        # check below passes vacuously on one -- so without this the validator
        # cheerfully reports "All checks passed" on a dead narrative.
        if isinstance(answer, list) and not answer:
            violations.append(f"{prefix}: 'answer' is empty (this narrative would score zero)")
        if isinstance(references, list) and not references:
            violations.append(f"{prefix}: 'references' is empty (nothing is cited)")

        cited_reference_indices: set[int] = set()
        total_words = 0
        for sentence_idx, sentence in enumerate(answer):
            sentence_prefix = f"{prefix} answer[{sentence_idx}]"
            if not isinstance(sentence, dict) or "text" not in sentence or "citations" not in sentence:
                violations.append(f"{sentence_prefix}: must be an object with 'text' and 'citations'")
                continue
            text = sentence["text"]
            citations = sentence["citations"]
            total_words += len(str(text).split())

            if not isinstance(citations, list):
                violations.append(f"{sentence_prefix}: 'citations' must be a list")
                continue
            if len(citations) > max_citations_per_sentence:
                violations.append(
                    f"{sentence_prefix}: {len(citations)} citations exceeds max "
                    f"{max_citations_per_sentence} per sentence"
                )
            for citation in citations:
                if not isinstance(citation, int) or not (0 <= citation < len(references)):
                    violations.append(f"{sentence_prefix}: citation index {citation!r} is invalid")
                else:
                    cited_reference_indices.add(citation)

        if total_words > max_words:
            violations.append(f"{prefix}: answer is {total_words} words, exceeds max {max_words}")

        # Deliberately not a violation: rag-task.md says "Do not reject or penalize
        # a RAG object because `references` contains uncited document IDs."
        uncited = set(range(len(references))) - cited_reference_indices
        if uncited:
            _logger.info(
                "%s: %d reference(s) uncited (allowed by the spec, not a violation)",
                prefix, len(uncited),
            )

    if topics_by_id is not None:
        missing = set(topics_by_id) - seen_narrative_ids
        if missing:
            violations.append(f"missing RAG output for {len(missing)} topic(s): {sorted(missing)[:10]}")

    return violations
