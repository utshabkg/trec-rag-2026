"""Writers for the two official TREC RAG 2026 submission files:
`r_output_trec_rag_2026.tsv` (Retrieval) and `rag_output_trec_rag_2026.jsonl` (RAG).
"""

from __future__ import annotations

import json
from pathlib import Path

from trecrag.schema import Candidate, RagAnswer, Topic


def write_retrieval_run(
    path: str | Path,
    run_id: str,
    results: dict[str, list[Candidate]],
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for topic_id, candidates in results.items():
            for candidate in candidates:
                fh.write(
                    f"{topic_id} Q0 {candidate.docid} {candidate.rank} "
                    f"{candidate.score:.6f} {run_id}\n"
                )


def build_rag_record(
    team_id: str,
    topic: Topic,
    run_id: str,
    run_desc: str,
    rag_answer: RagAnswer,
) -> dict:
    return {
        "metadata": {
            "team_id": team_id,
            "narrative_id": topic.topic_id,
            "narrative": topic.text,
            "run_id": run_id,
            "run_desc": run_desc,
        },
        "references": list(rag_answer.references),
        "answer": [
            {"text": sentence.text, "citations": list(sentence.citations)}
            for sentence in rag_answer.sentences
        ],
    }


def write_rag_jsonl(path: str | Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
