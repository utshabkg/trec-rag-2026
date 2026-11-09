"""Load `preprocess_step3_decompose.py` output into `DecomposedTopic` objects.

The `ain` block (step 2) and the `expanded_queries` block (step 3) are produced
by *separate runs of separate scripts*. Joining them wrongly does not fail
loudly: retrieval simply runs a well-formed query for the wrong information
need, every downstream stage behaves normally, and the run answers a different
question than the one it reports. So the join is keyed explicitly on
`(sub_question_id, ain_id)` and any disagreement raises.

That mirrors the all-or-nothing gate discipline the decompose scripts already
apply to their own output, for the same reason: a partial structure here looks
exactly like a complete one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trecrag.decomposed.schema import AtomicNeed, DecomposedTopic, SubQuestion
from trecrag.schema import Topic

# `step3c_expanded_query` is populated even when PRF did not run: with an empty
# `step3c_prf_terms` the term-append produces the 3B query verbatim. Trusting the
# string alone would report a 3C run that never happened, so the status field is
# the authority on whether PRF terms actually exist.
_PRF_RAN_STATUSES = frozenset({"ok", "success", "enabled", "complete", "completed"})


class DecompositionFormatError(ValueError):
    """Raised when a decomposition record cannot be trusted to be complete."""


def load_decomposed_topics(path: str | Path) -> dict[str, DecomposedTopic]:
    """Parse a step-3 JSONL file into `{query_id: DecomposedTopic}`."""
    decomposed: dict[str, DecomposedTopic] = {}
    with open(path, encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DecompositionFormatError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise DecompositionFormatError(f"{path}:{line_no}: expected a JSON object")
            try:
                topic = _parse_record(record)
            except DecompositionFormatError as exc:
                raise DecompositionFormatError(f"{path}:{line_no}: {exc}") from exc
            if topic.topic_id in decomposed:
                raise DecompositionFormatError(
                    f"{path}:{line_no}: duplicate query_id {topic.topic_id!r}"
                )
            decomposed[topic.topic_id] = topic
    if not decomposed:
        raise DecompositionFormatError(f"{path}: no records")
    return decomposed


def _parse_record(record: dict[str, Any]) -> DecomposedTopic:
    query_id = str(record.get("query_id") or "").strip()
    if not query_id:
        raise DecompositionFormatError("missing query_id")
    composite = str(record.get("comp_query") or "").strip()
    if not composite:
        raise DecompositionFormatError(f"{query_id}: missing comp_query")

    sub_questions = [
        SubQuestion(
            sub_question_id=str(entry.get("sub_question_id", "")).strip(),
            text=str(entry.get("sub_question_text", "")).strip(),
        )
        for entry in record.get("sub_questions", [])
        if isinstance(entry, dict)
    ]

    dependencies = [
        (str(entry["depends_on"]), str(entry["dependent"]))
        for entry in record.get("dependencies", [])
        if isinstance(entry, dict) and "depends_on" in entry and "dependent" in entry
    ]

    needs = _join_needs(query_id, record)
    if not needs:
        raise DecompositionFormatError(f"{query_id}: no atomic information needs")

    return DecomposedTopic(
        topic_id=query_id,
        composite_text=composite,
        metadata=str(record.get("query_metadata") or "").strip(),
        sub_questions=sub_questions,
        dependencies=dependencies,
        needs=needs,
    )


def _join_needs(query_id: str, record: dict[str, Any]) -> list[AtomicNeed]:
    """Join step 2's `ain` block onto step 3's `expanded_queries` block.

    Both are keyed by `(sub_question_id, ain_id)`. The two key sets must match
    exactly — a need with no reformulation would be retrieved on raw text while
    its siblings use conditioned queries, and a reformulation with no need is a
    query for something step 2 never identified.
    """
    ain_texts: dict[tuple[str, int], str] = {}
    for group in record.get("ain", []):
        if not isinstance(group, dict):
            continue
        sub_question_id = str(group.get("sub_question_id", "")).strip()
        for entry in group.get("ain_array", []):
            if not isinstance(entry, dict):
                continue
            key = (sub_question_id, _as_int(entry.get("ain_id"), query_id, "ain_id"))
            text = str(entry.get("ain_text") or "").strip()
            if not text:
                raise DecompositionFormatError(f"{query_id}: empty ain_text for {key}")
            ain_texts[key] = text

    expansions: dict[tuple[str, int], dict[str, Any]] = {}
    for group in record.get("expanded_queries", []):
        if not isinstance(group, dict):
            continue
        sub_question_id = str(group.get("sub_question_id", "")).strip()
        for entry in group.get("eq_array", []):
            if not isinstance(entry, dict):
                continue
            key = (sub_question_id, _as_int(entry.get("ain_id"), query_id, "ain_id"))
            expansions[key] = entry

    if ain_texts.keys() != expansions.keys():
        missing = sorted(ain_texts.keys() - expansions.keys())
        extra = sorted(expansions.keys() - ain_texts.keys())
        raise DecompositionFormatError(
            f"{query_id}: ain/expanded_queries misalignment "
            f"(no reformulation for {missing}; no need for {extra})"
        )

    needs: list[AtomicNeed] = []
    for key in sorted(ain_texts, key=lambda k: (k[0], k[1])):
        sub_question_id, ain_id = key
        entry = expansions[key]
        needs.append(
            AtomicNeed(
                ain_id=ain_id,
                sub_question_id=sub_question_id,
                ain_text=ain_texts[key],
                conditioned_query=_clean(entry.get("step3a_conditioned_query")),
                expanded_query=_clean(entry.get("step3b_expanded_query")),
                prf_query=_prf_query(entry),
            )
        )
    return needs


def _prf_query(entry: dict[str, Any]) -> str | None:
    """The 3C query, but only when PRF actually ran (see `_PRF_RAN_STATUSES`)."""
    status = str(entry.get("step3c_status") or "").strip().lower()
    if status not in _PRF_RAN_STATUSES:
        return None
    return _clean(entry.get("step3c_expanded_query"))


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _as_int(value: Any, query_id: str, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DecompositionFormatError(f"{query_id}: non-integer {field} {value!r}") from exc


def attach(
    topics: list[Topic], decomposed: dict[str, DecomposedTopic]
) -> list[DecomposedTopic]:
    """Return one `DecomposedTopic` per input topic, in the input's order.

    Raises if any topic has no decomposition. Silently falling back to composite
    retrieval for the unmatched ones would produce a run that is part decomposed
    and part baseline, with no record of which topic took which path — the result
    would be uninterpretable and would look like a successful run.
    """
    missing = [topic.topic_id for topic in topics if topic.topic_id not in decomposed]
    if missing:
        shown = ", ".join(missing[:10])
        suffix = f" (and {len(missing) - 10} more)" if len(missing) > 10 else ""
        raise DecompositionFormatError(
            f"{len(missing)} of {len(topics)} topic(s) have no decomposition: {shown}{suffix}. "
            "Run the step 1-3 preprocessors over this topics file first."
        )
    return [decomposed[topic.topic_id] for topic in topics]
