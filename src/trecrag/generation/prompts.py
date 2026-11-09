"""Jinja rendering for the generator prompt.

The prompt lives in `templates/rag_answer.jinja` rather than in a Python string
because it is the most-edited artefact in the system and the least testable:
every line of it was added in response to a measured failure over 119
narratives, and reading it as a template makes the instruction set legible as a
whole instead of as concatenated list items.

`StrictUndefined` is deliberate. A mistyped variable would otherwise render as
an empty string, producing a prompt that is subtly wrong -- missing its word
target, or its citation cap -- and still returns a plausible answer. There is no
downstream check that would catch it, so the template fails loudly instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_ANSWER_TEMPLATE = "rag_answer.jinja"
_VERIFICATION_TEMPLATE = "answer_verification.jinja"

# The verifier shows passages more tersely than the generator: it is judging
# attribution, not writing from them, and its own reply is only a few tokens.
VERIFIER_PASSAGE_CHAR_CAP = 1200

# Fractions of the word budget used as the soft target. "At most N words" alone
# reliably produced ~250-word answers against a 1024-word budget (WORK_LOG 5.1),
# so the prompt states an explicit range instead.
_TARGET_LOW = 0.6
_TARGET_HIGH = 0.9

# The generator's own passage cap. Far more generous than the rerankers' windows
# because by this point the text is already the query-focused span.
PASSAGE_CHAR_CAP = 2500


@dataclass(frozen=True)
class Revision:
    """Verifier feedback threaded into a retry prompt."""

    issues: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.issues or self.instructions)


@lru_cache(maxsize=1)
def _environment():
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "The LLM generator prompt is a Jinja template; install 'jinja2' "
            "(it ships in the base dependencies of this package)."
        ) from exc
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,  # a prompt is plain text; escaping would corrupt it
    )


def render_answer_prompt(
    question: str,
    passage_texts: list[str],
    max_words: int,
    max_citations: int,
    thinking: bool = False,
    revision: Revision | None = None,
    passage_char_cap: int = PASSAGE_CHAR_CAP,
) -> str:
    """Render the generator prompt.

    `passage_texts` is in evidence order; indices handed to the model are
    zero-based positions into that same list, which is what the response parser
    and `RagAnswer.references` both assume.
    """
    passages = [
        {"index": i, "text": (text or "")[:passage_char_cap]}
        for i, text in enumerate(passage_texts)
    ]
    return _environment().get_template(_ANSWER_TEMPLATE).render(
        question=question,
        passages=passages,
        max_words=max_words,
        max_citations=max_citations,
        target_low=int(max_words * _TARGET_LOW),
        target_high=int(max_words * _TARGET_HIGH),
        thinking=thinking,
        revision=revision if (revision and revision) else None,
    )


def render_verification_prompt(
    question: str,
    passage_texts: list[str],
    sentences: list[tuple[str, list[int]]],
    thinking: bool = False,
    passage_char_cap: int = VERIFIER_PASSAGE_CHAR_CAP,
) -> str:
    """Render the answer-verification prompt.

    `sentences` is `(text, citations)` in answer order. Sentence numbers handed
    to the model are **1-based** — `LLMVerifier._parse` range-checks the numbers
    it reads back against `1..len(sentences)`, so numbering from zero here would
    silently drop every reported sentence and report a clean answer.

    With `thinking`, raise the caller's token budget too: the judge's reply is
    only two lines, so a default-sized budget spent on the think block leaves it
    unterminated, `strip_reasoning` drops the lot, and an empty judgment is
    treated as "passed" — verification would switch itself off rather than fail.
    """
    passages = [
        {"index": i, "text": (text or "")[:passage_char_cap]}
        for i, text in enumerate(passage_texts)
    ]
    numbered = [
        {"number": i, "text": text, "citations": citations}
        for i, (text, citations) in enumerate(sentences, start=1)
    ]
    return _environment().get_template(_VERIFICATION_TEMPLATE).render(
        question=question,
        passages=passages,
        sentences=numbered,
        thinking=thinking,
    )
