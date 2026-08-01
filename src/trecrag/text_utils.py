"""Shared lightweight text helpers used by the dependency-free fallback
backends (hashing encoders, lexical scorers, extractive generator)."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
