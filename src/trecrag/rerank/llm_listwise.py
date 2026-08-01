"""LLM listwise reranking stage (RankGPT-style sliding window).

Processes the candidate list in overlapping windows from the bottom up,
asking the LLM to output a relevance-ordered permutation of each window and
splicing the reordered window back in, then sliding toward the top. With
`IdentityLLMClient` (empty completions) every window's parsed order degrades
to the original order, so this stage is a safe no-op by default.
"""

from __future__ import annotations

import re

from trecrag.clients.llm_client import LLMClient
from trecrag.logging_utils import get_logger
from trecrag.rerank.base import Reranker
from trecrag.schema import Candidate, Topic

logger = get_logger("llm_listwise")

_BRACKETED_RE = re.compile(r"\[(\d+)\]")
_INT_RE = re.compile(r"\d+")


class LLMListwiseReranker(Reranker):
    name = "llm_listwise"

    def __init__(
        self,
        llm: LLMClient,
        window_size: int = 20,
        step: int = 10,
        max_passage_chars: int = 300,
        rerank_depth: int | None = None,
    ):
        self.llm = llm
        self.window_size = window_size
        self.step = step
        self.max_passage_chars = max_passage_chars
        self.rerank_depth = rerank_depth

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []

        # This stage dominates runtime -- one LLM call per sliding window, and the
        # windows scale with how deep we rerank. `rerank_depth` limits it to the
        # prefix that actually matters: the head becomes the generator's evidence,
        # while the tail only pads the retrieval run, where the cross-encoder's
        # ordering is already a reasonable answer. The tail is appended unchanged.
        tail: list[Candidate] = []
        if self.rerank_depth is not None and len(candidates) > self.rerank_depth:
            candidates, tail = candidates[: self.rerank_depth], candidates[self.rerank_depth :]

        working = list(candidates)
        n = len(working)
        window = min(self.window_size, n)
        step = min(self.step, window) or window

        end = n
        start = max(0, n - window)
        while True:
            working[start:end] = self._rank_window(topic.text, working[start:end])
            if start == 0:
                break
            end = max(window, end - step)
            start = max(0, end - window)

        top = (working + tail)[:k]
        for new_rank, candidate in enumerate(top, start=1):
            candidate.rank = new_rank
            # Synthetic descending score: the run file requires non-increasing
            # scores within a topic, and splicing a CE-ordered tail (whose scores
            # are on a different scale) onto a listwise-ordered head would
            # otherwise break monotonicity.
            candidate.score = float(len(top) - new_rank + 1)
            candidate.features["llm_listwise_rank"] = float(new_rank)
        return top

    def _rank_window(self, query: str, window: list[Candidate]) -> list[Candidate]:
        if len(window) <= 1:
            return window
        # 256 tokens truncates a 20-item permutation for chattier models, and a
        # truncated reasoning block leaves its digits in the string to be parsed.
        response = self.llm.complete(self._build_prompt(query, window), max_tokens=512)
        order = self._parse_order(response, len(window))
        if order is None:
            logger.warning(
                "listwise: unparsable ordering for a %d-passage window; keeping input order",
                len(window),
            )
            return window
        return [window[i] for i in order]

    def _build_prompt(self, query: str, window: list[Candidate]) -> str:
        lines = [
            f"Query: {query}",
            "",
            "Rank the following passages from most to least relevant to the query.",
            'Respond with one line listing passage numbers in order, most relevant '
            'first, e.g. "[3] > [1] > [2]".',
            "",
        ]
        for i, candidate in enumerate(window, start=1):
            text = (candidate.text or "")[: self.max_passage_chars]
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _parse_order(response: str, n: int) -> list[int] | None:
        """Parse the model's permutation, or None if the reply isn't one.

        The prompt asks for "[3] > [1] > [2]", so bracketed indices are the
        signal. Reading every bare integer instead silently corrupts the ranking
        whenever the model numbers its list ("1. [7]" -> 1 outranks 7) or explains
        itself in prose -- and since missing indices are back-filled, the result
        is always a *valid* permutation, so nothing ever errors or logs.
        """
        matches = _BRACKETED_RE.findall(response)
        bracketed = bool(matches)
        if not bracketed:
            # No brackets at all: fall back to bare integers, which is still right
            # for a terse reply like "3 > 1 > 2".
            matches = _INT_RE.findall(response)

        order: list[int] = []
        seen: set[int] = set()
        for match in matches:
            idx = int(match) - 1
            if 0 <= idx < n and idx not in seen:
                order.append(idx)
                seen.add(idx)

        if not order:
            return None
        # Bracketed indices mean the model answered in the requested format, so
        # even a partial ranking is signal and the rest is back-filled. Bare
        # integers are only trustworthy when they cover most of the window --
        # otherwise we are reading digits out of prose, and back-filling would
        # dress that noise up as a confident reordering.
        if not bracketed and len(order) * 2 < n:
            return None

        order.extend(idx for idx in range(n) if idx not in seen)
        return order
