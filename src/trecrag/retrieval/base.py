from __future__ import annotations

from abc import ABC, abstractmethod

from trecrag.schema import Candidate, Topic


class Retriever(ABC):
    name: str = "retriever"

    @abstractmethod
    def retrieve(self, topic: Topic, k: int) -> list[Candidate]: ...
