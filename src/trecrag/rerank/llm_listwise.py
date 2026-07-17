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
from trecrag.rerank.base import Reranker
from trecrag.schema import Candidate, Topic

_INT_RE = re.compile(r"\d+")


class LLMListwiseReranker(Reranker):
    name = "llm_listwise"

    def __init__(
        self,
        llm: LLMClient,
        window_size: int = 20,
        step: int = 10,
        max_passage_chars: int = 300,
    ):
        self.llm = llm
        self.window_size = window_size
        self.step = step
        self.max_passage_chars = max_passage_chars

    def rerank(self, topic: Topic, candidates: list[Candidate], k: int) -> list[Candidate]:
        if not candidates:
            return []
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

        top = working[:k]
        for new_rank, candidate in enumerate(top, start=1):
            candidate.rank = new_rank
            candidate.score = float(len(top) - new_rank + 1)
            candidate.features["llm_listwise_rank"] = float(new_rank)
        return top

    def _rank_window(self, query: str, window: list[Candidate]) -> list[Candidate]:
        if len(window) <= 1:
            return window
        response = self.llm.complete(self._build_prompt(query, window), max_tokens=256)
        order = self._parse_order(response, len(window))
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
    def _parse_order(response: str, n: int) -> list[int]:
        found = [int(match) - 1 for match in _INT_RE.findall(response)]
        order: list[int] = []
        seen: set[int] = set()
        for idx in found:
            if 0 <= idx < n and idx not in seen:
                order.append(idx)
                seen.add(idx)
        for idx in range(n):
            if idx not in seen:
                order.append(idx)
        return order
