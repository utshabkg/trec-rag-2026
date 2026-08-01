"""Per-stage instrumentation for the rerank cascade.

The cascade is a sequence of rerankers, each of which can only reorder or discard
what the previous one passed it. When answer quality drops there is no way, from
the submission files alone, to tell which stage caused it: every stage produces a
plausible ranked list, and a stage that reorders randomly looks exactly like a
stage that reorders well.

This records, per topic per stage: how many candidates went in and out, the score
distribution, and how much the stage actually moved things relative to its input
(top-k overlap and displacement). Those three numbers separate the cases that
matter:

  - overlap@10 ~= 1.0 and no displacement -> the stage is a no-op; it costs time
    and buys nothing.
  - overlap@10 near 0 -> the stage is overriding everything upstream. That is
    either the point (a strong cross-encoder) or a bug (a parser reading noise),
    and which one it is shows up in the score spread.
  - a near-flat score distribution -> the model cannot discriminate between these
    candidates, so its ordering is close to arbitrary regardless of overlap.

Written as JSONL so a run can be analysed after the fact without re-running it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from trecrag.schema import Candidate


def _score_stats(candidates: list[Candidate]) -> dict[str, float | None]:
    scores = [float(c.score) for c in candidates]
    if not scores:
        return {"min": None, "max": None, "mean": None, "stdev": None}
    return {
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "mean": round(statistics.mean(scores), 6),
        # Spread is the discrimination signal: a stage whose scores are all
        # within noise of each other is not really ranking.
        "stdev": round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0,
    }


def _overlap_at_k(before: list[Candidate], after: list[Candidate], k: int) -> float | None:
    """Fraction of the input top-k that survives in the output top-k."""
    if not before or not after:
        return None
    a = {c.docid for c in before[:k]}
    b = {c.docid for c in after[:k]}
    if not a:
        return None
    return round(len(a & b) / len(a), 4)


def _mean_displacement(before: list[Candidate], after: list[Candidate], k: int) -> float | None:
    """Mean |rank change| for documents present in both, over the input top-k.

    Distinguishes a stage that reshuffles a few positions from one that rewrites
    the ordering wholesale -- overlap alone cannot tell those apart.
    """
    if not before or not after:
        return None
    pos_after = {c.docid: i for i, c in enumerate(after)}
    deltas = [
        abs(i - pos_after[c.docid]) for i, c in enumerate(before[:k]) if c.docid in pos_after
    ]
    return round(statistics.mean(deltas), 2) if deltas else None


class StageDiagnostics:
    """Collects one record per (topic, stage). Disabled instances are no-ops."""

    def __init__(self, path: str | Path | None = None, top_k: int = 10):
        self.path = Path(path) if path else None
        self.top_k = top_k
        self._records: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(
        self,
        topic_id: str,
        stage: str,
        before: list[Candidate],
        after: list[Candidate],
    ) -> None:
        if not self.enabled:
            return
        self._records.append(
            {
                "topic_id": topic_id,
                "stage": stage,
                "n_in": len(before),
                "n_out": len(after),
                "with_text_in": sum(1 for c in before if c.text),
                "with_text_out": sum(1 for c in after if c.text),
                "scores": _score_stats(after),
                f"overlap_at_{self.top_k}": _overlap_at_k(before, after, self.top_k),
                f"mean_displacement_at_{self.top_k}": _mean_displacement(
                    before, after, self.top_k
                ),
            }
        )

    def flush(self) -> None:
        """Append and clear. Called per topic so a killed run keeps what it had."""
        if not self.enabled or not self._records:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._records.clear()
