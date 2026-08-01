"""Pluggable entity-embedding sources for QDER's entity channel.

The paper's entity channel needs (a) an entity linker to find entity mentions
in text and (b) pretrained knowledge-graph embeddings (Wikipedia2Vec) for the
linked entities — see plans/qder-approach.md §2.1. Neither a WAT linker nor a
Wikipedia2Vec table for ClimbMix is available here, so `use_entities` defaults
to off (`NullEntityProvider`). `HeuristicEntityProvider` is a dependency-free
stand-in (capitalized-phrase spotting + hashed vectors) that exercises the
entity channel's code path end to end without either dependency; swap in a
real linker + embedding table (e.g. the `entity_ranking/` pipeline in
SIGIR2025-QDER) by implementing `EntityEmbeddingProvider`.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

_ENTITY_MENTION_RE = re.compile(r"\b[A-Z][a-zA-Z0-9']*(?:\s+[A-Z][a-zA-Z0-9']*)*\b")
_LEADING_STOPWORDS = {"The", "A", "An", "This", "That", "These", "Those", "It", "I"}


class EntityEmbeddingProvider(ABC):
    dims: int = 300

    @abstractmethod
    def get_entity_vectors(self, text: str, max_entities: int) -> np.ndarray:
        """Return an [num_entities, dims] matrix of entity embeddings for `text`."""


class NullEntityProvider(EntityEmbeddingProvider):
    """Entity channel disabled: always returns zero entities."""

    def get_entity_vectors(self, text: str, max_entities: int) -> np.ndarray:
        return np.zeros((0, self.dims), dtype=np.float32)


class HeuristicEntityProvider(EntityEmbeddingProvider):
    """Regex capitalized-phrase spotting + hashed embeddings.

    Not a real entity linker — a legitimate, dependency-free stand-in so the
    entity channel can run without a WAT linker or a Wikipedia2Vec table.
    Mentions are deduplicated per call; each is hashed into a deterministic
    unit vector in a namespace distinct from the text channel's token hashes.
    """

    def __init__(self, dims: int = 300):
        self.dims = dims

    def _mentions(self, text: str, max_entities: int) -> list[str]:
        seen: list[str] = []
        for match in _ENTITY_MENTION_RE.finditer(text):
            words = match.group(0).split()
            if words and words[0] in _LEADING_STOPWORDS:
                # Strip only the leading stopword ("The Eiffel Tower" -> "Eiffel
                # Tower") rather than discarding the whole match, which used to
                # throw away the real entity along with the stopword.
                words = words[1:]
            if not words:
                continue
            phrase = " ".join(words)

            # A single Title-Case word is usually just sentence-initial
            # capitalization (English capitalizes sentence starts), not a real
            # entity -- require 2+ consecutive capitalized words, with a
            # dedicated exception for short real acronyms ("US", "AI", "NASA").
            is_multi_word = len(words) >= 2
            is_acronym = len(words) == 1 and words[0].isupper() and 2 <= len(words[0]) <= 6
            if not (is_multi_word or is_acronym):
                continue

            if phrase not in seen:
                seen.append(phrase)
            if len(seen) >= max_entities:
                break
        return seen

    def get_entity_vectors(self, text: str, max_entities: int) -> np.ndarray:
        mentions = self._mentions(text, max_entities)
        if not mentions:
            return np.zeros((0, self.dims), dtype=np.float32)
        vectors = np.zeros((len(mentions), self.dims), dtype=np.float32)
        for i, mention in enumerate(mentions):
            digest = hashlib.blake2b(
                f"entity::{mention.lower()}".encode("utf-8"), digest_size=8
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            vector = rng.normal(size=self.dims).astype(np.float32)
            vectors[i] = vector / (float(np.linalg.norm(vector)) + 1e-9)
        return vectors
