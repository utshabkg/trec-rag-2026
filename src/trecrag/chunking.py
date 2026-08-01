"""Query-focused passage selection inside long ClimbMix documents.

ClimbMix documents are whole web pages and books, not passages. Measured over a
real 1000-candidate search response: mean 66,685 chars, median 42,948, max
834,116 — and *not one document* was under 2000 chars.

Every stage after BM25 truncates its input to a prefix (the biencoder and
cross-encoder to ~2000 chars, the listwise reranker to a few hundred, the
generator to ~1500). So without passage selection the whole cascade reranks and
cites each document by its title page. Measured over 200 real candidates for
`rag2026-0`: the leading window was the most query-relevant window in only
10/200 documents, the median best window began at char offset 15,900, and
choosing it raised query-overlap by 32%. The symptom downstream was that all 50
surviving cross-encoder scores were negative — the model was being asked to
judge front matter against the narrative.

`rag-task.md` explicitly permits this: "Apply its own chunking or passage
selection internally." Selection here is lexical and CPU-only (~1.7s per
1000-candidate topic), so it costs nothing next to the neural stages.
"""

from __future__ import annotations

from trecrag.schema import Candidate
from trecrag.text_utils import tokenize


def iter_windows(text: str, window_chars: int, stride_chars: int) -> list[tuple[int, str]]:
    """Overlapping (offset, window) pairs. Overlap keeps a relevant span from
    being split across a boundary and scoring poorly in both halves."""
    if not text:
        return []
    if len(text) <= window_chars:
        return [(0, text)]
    stride = max(1, stride_chars)
    offsets = range(0, len(text) - window_chars + stride, stride)
    return [(off, text[off : off + window_chars]) for off in offsets if off < len(text)]


def _score(window: str, query_tokens: set[str]) -> float:
    tokens = tokenize(window)
    if not tokens:
        return 0.0
    # Length-normalised overlap: without the sqrt, long windows win on raw count.
    return sum(1 for t in tokens if t in query_tokens) / (len(tokens) ** 0.5)


def select_passage(
    text: str,
    query_tokens: set[str],
    window_chars: int = 2000,
    stride_chars: int = 1000,
    windows_per_doc: int = 2,
    joiner: str = " ... ",
) -> str:
    """Return the most query-relevant span(s) of `text`, in document order.

    Taking more than one window matters here because the narratives are
    multi-faceted: a single span often answers one sub-question while another
    sits elsewhere in the same document. Selected windows are non-overlapping,
    and are re-joined in document order so the passage still reads forwards.
    """
    windows = iter_windows(text, window_chars, stride_chars)
    if not windows:
        return text
    if len(windows) == 1:
        return windows[0][1]

    ranked = sorted(windows, key=lambda ow: _score(ow[1], query_tokens), reverse=True)

    chosen: list[tuple[int, str]] = []
    for offset, window in ranked:
        if len(chosen) >= max(1, windows_per_doc):
            break
        if any(abs(offset - other) < window_chars for other, _ in chosen):
            continue  # overlaps something already taken; skip to add new material
        chosen.append((offset, window))

    chosen.sort(key=lambda ow: ow[0])
    return joiner.join(window for _, window in chosen)


class PassageSelector:
    """Rewrites `Candidate.text` to a query-focused span, in place.

    Runs once per topic immediately after BM25, so every downstream stage —
    biencoder, cross-encoder, listwise reranker, generator — sees the same
    relevant span rather than each independently truncating the raw document.
    """

    def __init__(
        self,
        window_chars: int = 2000,
        stride_chars: int = 1000,
        windows_per_doc: int = 2,
        min_doc_chars: int = 2000,
    ):
        self.window_chars = window_chars
        self.stride_chars = stride_chars
        self.windows_per_doc = windows_per_doc
        self.min_doc_chars = min_doc_chars

    def apply(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return candidates
        for candidate in candidates:
            text = candidate.text
            if not text or len(text) <= self.min_doc_chars:
                continue  # already passage-sized; selection would only lose context
            candidate.text = select_passage(
                text,
                query_tokens,
                window_chars=self.window_chars,
                stride_chars=self.stride_chars,
                windows_per_doc=self.windows_per_doc,
            )
        return candidates
