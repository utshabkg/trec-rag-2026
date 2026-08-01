"""Per-topic checkpointing so a long run is resumable and crash-safe.

A full 119-topic run takes over an hour and spends most of it on Pyserini calls
and LLM generation. Accumulating everything in memory and writing the two
submission files at the end means any failure — an API 429 storm, an OOM, a
dropped SSH session — costs the entire run. So each topic's finished result is
persisted the moment it is produced, and the submission files are assembled from
those records afterwards.

One JSON file per topic (rather than appending to the real submission files)
keeps the writes atomic and order-independent, lets a resumed run reassemble
output in canonical topic order regardless of completion order, and makes
re-assembly possible without re-running anything.
"""

from __future__ import annotations

import json
from pathlib import Path

from trecrag.schema import AnswerSentence, Candidate, RagAnswer


class TopicCheckpointStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, topic_id: str) -> Path:
        # Topic IDs are `rag2026-<n>`, already filesystem-safe; guard anyway.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic_id)
        return self.directory / f"{safe}.json"

    def has(self, topic_id: str) -> bool:
        return self._path(topic_id).is_file()

    def save(self, topic_id: str, ranked: list[Candidate], answer: RagAnswer) -> None:
        # Scores arriving from the numpy/torch stages are np.float32, which the
        # json module refuses; coerce the whole numeric surface to Python floats.
        payload = {
            "topic_id": topic_id,
            "ranked": [
                {
                    "docid": c.docid,
                    "score": float(c.score),
                    "rank": int(c.rank),
                    "features": {k: float(v) for k, v in c.features.items()},
                }
                for c in ranked
            ],
            "answer": {
                "references": answer.references,
                "sentences": [
                    {"text": s.text, "citations": s.citations} for s in answer.sentences
                ],
            },
        }
        path = self._path(topic_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: a partial write can never look like a done topic

    def load(self, topic_id: str) -> tuple[list[Candidate], RagAnswer] | None:
        path = self._path(topic_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        ranked = [
            Candidate(
                docid=entry["docid"],
                score=float(entry["score"]),
                rank=int(entry["rank"]),
                features=entry.get("features", {}),
            )
            for entry in payload.get("ranked", [])
        ]
        raw_answer = payload.get("answer", {})
        answer = RagAnswer(
            references=list(raw_answer.get("references", [])),
            sentences=[
                AnswerSentence(text=s["text"], citations=list(s.get("citations", [])))
                for s in raw_answer.get("sentences", [])
            ],
        )
        return ranked, answer
