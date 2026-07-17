"""Shared network-free test doubles."""

from __future__ import annotations

from trecrag.clients.llm_client import LLMClient
from trecrag.retrieval.base import Retriever
from trecrag.schema import Candidate, Topic


class FakeRetriever(Retriever):
    name = "fake"

    def __init__(self, candidates_by_topic: dict[str, list[Candidate]]):
        self._by_topic = candidates_by_topic

    def retrieve(self, topic: Topic, k: int) -> list[Candidate]:
        candidates = self._by_topic.get(topic.topic_id, [])
        return [
            Candidate(docid=c.docid, score=c.score, rank=c.rank, text=c.text, features=dict(c.features))
            for c in candidates[:k]
        ]


class ReversingLLMClient(LLMClient):
    """Always answers with passages in fully reversed order, to exercise the
    listwise sliding window without needing a real model."""

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        lines = [line for line in prompt.splitlines() if line.startswith("[")]
        indices = list(range(1, len(lines) + 1))
        return " > ".join(f"[{i}]" for i in reversed(indices))
