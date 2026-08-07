"""Typed carriers for one decomposed narrative.

Deliberately separate from `trecrag.schema.Topic`, which stays `topic_id` +
`text`. The cascade, the run writers and the checkpoint store all read `Topic`,
so widening it would ripple through code paths that produced the banked
submission. Instead a `DecomposedTopic` *emits* plain `Topic` objects — one per
atomic information need — and the cascade runs unmodified against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trecrag.schema import Topic

# Ordered best-to-worst. Asking for a variant that a record does not carry falls
# through to the next one rather than retrieving on an empty string: every record
# produced so far has `step3c_status: "disabled"`, so a config naming 3C is a
# live case, not a hypothetical.
QUERY_VARIANTS = (
    "step3c_expanded_query",
    "step3b_expanded_query",
    "step3a_conditioned_query",
    "ain_text",
)


@dataclass(frozen=True)
class AtomicNeed:
    """One single-intent retrieval unit, with its reformulations."""

    ain_id: int
    sub_question_id: str
    ain_text: str
    conditioned_query: str | None = None      # step 3A
    expanded_query: str | None = None         # step 3B
    prf_query: str | None = None              # step 3C (unpopulated while disabled)

    def query_text(self, variant: str = "step3b_expanded_query") -> str:
        """The query to retrieve with, degrading through `QUERY_VARIANTS`.

        Falling back rather than raising is deliberate: a missing 3C query means
        that stage has not been run, which is the current state of every record
        on disk. Retrieving with the 3B query is the correct behaviour there, and
        failing the whole run instead would make the config's default unusable.
        """
        if variant not in QUERY_VARIANTS:
            raise ValueError(
                f"unknown query variant {variant!r}; expected one of {list(QUERY_VARIANTS)}"
            )
        by_variant = {
            "step3c_expanded_query": self.prf_query,
            "step3b_expanded_query": self.expanded_query,
            "step3a_conditioned_query": self.conditioned_query,
            "ain_text": self.ain_text,
        }
        for candidate_variant in QUERY_VARIANTS[QUERY_VARIANTS.index(variant) :]:
            text = by_variant.get(candidate_variant)
            if text and text.strip():
                return text.strip()
        # `ain_text` is gated non-empty by the loader, so this is unreachable in
        # practice; returning it keeps the signature total rather than raising
        # deep inside a fan-out loop.
        return self.ain_text


@dataclass(frozen=True)
class SubQuestion:
    sub_question_id: str
    text: str


@dataclass(frozen=True)
class DecomposedTopic:
    """One narrative, decomposed. `needs` is flattened across sub-questions."""

    topic_id: str
    composite_text: str
    metadata: str = ""
    sub_questions: list[SubQuestion] = field(default_factory=list)
    # (depends_on, dependent) sub-question ids. Carried for later use; nothing
    # in the retrieval path acts on it yet.
    dependencies: list[tuple[str, str]] = field(default_factory=list)
    needs: list[AtomicNeed] = field(default_factory=list)

    def as_topic(self) -> Topic:
        """The composite narrative, unmodified — the baseline retrieval arm and
        the question the generator is ultimately answering."""
        return Topic(topic_id=self.topic_id, text=self.composite_text)

    def as_generation_topic(self, include_sub_questions: bool = True) -> Topic:
        """The composite narrative with its sub-questions spelled out.

        The generator prompt already demands "identify every distinct
        sub-question the request asks and address each one explicitly", but until
        now the model had to re-derive that list from a dense paragraph at
        generation time — the exact task step 1 already did carefully, offline,
        with a validating parser. Handing over the extracted list makes the
        existing instruction actionable instead of aspirational.

        Safe with respect to the submission: `scripts/run_pipeline.py` builds the
        RAG record from the topic it loaded from the TSV, not from this one, so
        `metadata.narrative` still matches the official text exactly — which
        `validate_rag_jsonl` checks byte-for-byte.
        """
        if not include_sub_questions or not self.sub_questions:
            return self.as_topic()
        lines = [self.composite_text, "", "This request asks the following distinct questions:"]
        lines += [
            f"{i}. {sub_question.text}"
            for i, sub_question in enumerate(self.sub_questions, start=1)
            if sub_question.text
        ]
        return Topic(topic_id=self.topic_id, text="\n".join(lines))

    def ain_topics(self, variant: str = "step3b_expanded_query") -> list[Topic]:
        """One synthetic `Topic` per need.

        The synthetic id (`rag2026-0#sq1-ain3`) keeps per-need diagnostics
        attributable to the need that produced them, and it is never written to a
        submission file — the fused result is stored under the real `topic_id`.
        """
        return [
            Topic(
                topic_id=f"{self.topic_id}#sq{need.sub_question_id}-ain{need.ain_id}",
                text=need.query_text(variant),
            )
            for need in self.needs
        ]
