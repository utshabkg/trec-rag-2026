"""Fan the cascade out over atomic information needs, then pool the results.

Exposes the same `run_topic(topic) -> (ranked, RagAnswer)` contract as
`RagPipeline`, so `scripts/run_pipeline.py` drives it with its existing worker
loop, per-topic checkpointing, canonical-order assembly and `.partial` guard —
none of that had to be duplicated or changed.

Two structural choices are worth stating up front:

* **Listwise reranking runs once, on the pooled list — never per need.** It costs
  ~9.9s per call against ~0.5s for the cross-encoder, and a narrative averages
  ~10.7 needs. Per-need listwise would put a single stage at roughly 80% of a
  multi-hour run to order candidate sets that are about to be merged anyway.
* **Generation is unchanged.** The generator receives the *composite* narrative
  and the pooled evidence, exactly as it does today; only the evidence pool is
  built differently. That is the whole of "Step 6" in `Full-pipeline.md`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from trecrag.decomposed.evidence import effective_evidence_k, select_evidence
from trecrag.decomposed.fusion import AinRun, fuse_ain_runs
from trecrag.decomposed.schema import AtomicNeed, DecomposedTopic
from trecrag.diagnostics import StageDiagnostics
from trecrag.generation.generator import Generator
from trecrag.logging_utils import get_logger
from trecrag.pipeline import GenerationSettings, RagPipeline
from trecrag.rerank.base import Reranker
from trecrag.schema import Candidate, RagAnswer, Topic

logger = get_logger("decomposed")


@dataclass
class FanoutSettings:
    query_variant: str = "step3b_expanded_query"
    per_ain_top_k: int = 5
    fusion_rrf_k: int = 60
    fusion_top_k: int = 100
    composite_weight: float = 1.0
    include_composite_run: bool = True
    # Runaway guard, not a style rule. Step 2 measured one sub-question yielding
    # 24 needs; a narrative like that would otherwise buy 24 cascades. `None`
    # means no cap, which is the intended default.
    max_ains: int | None = None
    final_top_k: int = 50
    # Evidence selection spans sub-questions instead of taking the pool's head.
    # Off would reproduce the baseline's coverage failure on a decomposed pool.
    coverage_aware_evidence: bool = True
    # Floor on `evidence_k`, as a multiple of the narrative's sub-question count.
    # 1 is inert on today's data (max 8 sub-questions vs evidence_k 10).
    evidence_per_sub_question: int = 1
    # Spell the extracted sub-questions out in the generation prompt.
    include_sub_questions_in_prompt: bool = True


class DecomposedPipeline:
    def __init__(
        self,
        ain_cascade: RagPipeline,
        generator: Generator,
        decomposed: dict[str, DecomposedTopic],
        settings: FanoutSettings,
        generation: GenerationSettings,
        composite_cascade: RagPipeline | None = None,
        llm_listwise: Reranker | None = None,
        diagnostics: StageDiagnostics | None = None,
    ):
        self.ain_cascade = ain_cascade
        self.composite_cascade = composite_cascade
        self.llm_listwise = llm_listwise
        self.generator = generator
        self.decomposed = decomposed
        self.settings = settings
        self.generation = generation
        self.diagnostics = diagnostics or StageDiagnostics(None)

    def retrieve(self, topic: Topic) -> list[Candidate]:
        decomposed_topic = self.decomposed.get(topic.topic_id)
        if decomposed_topic is None:
            # `loader.attach` should have caught this before the run started;
            # failing here too keeps a hand-built pipeline from quietly
            # degrading to composite-only retrieval for some topics.
            raise KeyError(f"no decomposition for topic {topic.topic_id!r}")

        needs = self._select_needs(decomposed_topic)
        started = time.perf_counter()
        runs: list[AinRun] = []
        for need, ain_topic in needs:
            candidates = self._retrieve_ain(ain_topic)
            if candidates:
                runs.append(
                    AinRun(
                        sub_question_id=need.sub_question_id,
                        ain_id=need.ain_id,
                        candidates=candidates,
                    )
                )

        composite_run: list[Candidate] | None = None
        if self.settings.include_composite_run and self.composite_cascade is not None:
            composite_run = self.composite_cascade.retrieve(decomposed_topic.as_topic())

        fused = fuse_ain_runs(
            runs,
            composite_run=composite_run,
            k_const=self.settings.fusion_rrf_k,
            top_k=self.settings.fusion_top_k,
            composite_weight=self.settings.composite_weight,
        )
        # `before` is the composite run on purpose: overlap@10 between it and the
        # pool then reads as "how much did decomposition actually change the
        # evidence?", which is the question this whole arm exists to answer.
        self.diagnostics.record(
            topic.topic_id, "ain_fanout", composite_run or [], fused,
            elapsed_s=time.perf_counter() - started,
        )
        logger.info(
            "Topic %s: %d/%d needs retrieved, pool=%d (composite=%d)",
            topic.topic_id, len(runs), len(needs), len(fused),
            len(composite_run) if composite_run else 0,
        )

        started = time.perf_counter()
        before = fused
        if self.llm_listwise is not None:
            # Ordered against the composite narrative, not any single need: this
            # list is the answer to the whole question, and the needs it came
            # from have already done their job.
            ranked = self.llm_listwise.rerank(
                decomposed_topic.as_topic(), fused, self.settings.final_top_k
            )
        else:
            ranked = fused[: self.settings.final_top_k]
        self.diagnostics.record(
            topic.topic_id, "pooled_listwise", before, ranked,
            elapsed_s=time.perf_counter() - started,
        )
        return ranked

    def run_topic(self, topic: Topic) -> tuple[list[Candidate], RagAnswer]:
        final_ranked = self.retrieve(topic)
        decomposed_topic = self.decomposed[topic.topic_id]
        evidence = self._select_evidence(decomposed_topic, final_ranked)
        started = time.perf_counter()
        rag_answer = self.generator.generate(
            # The composite narrative, not a need: the answer is judged against
            # the question the user actually asked. Sub-questions are appended so
            # the prompt's existing "address each one explicitly" instruction has
            # the extracted list to work from.
            decomposed_topic.as_generation_topic(
                include_sub_questions=self.settings.include_sub_questions_in_prompt
            ),
            evidence,
            max_words=self.generation.max_words,
            max_citations=self.generation.max_citations_per_sentence,
        )
        self.diagnostics.record(
            topic.topic_id, "generation", evidence, evidence,
            elapsed_s=time.perf_counter() - started,
        )
        return final_ranked, rag_answer

    def _select_evidence(
        self, decomposed_topic: DecomposedTopic, ranked: list[Candidate]
    ) -> list[Candidate]:
        """The passages the generator may cite, chosen to span the narrative.

        Pool rank alone concentrates evidence: fusion rewards agreement, needs
        cluster inside sub-questions, so the head of the pool skews toward
        whichever sub-question spawned the most needs. See `evidence.py`.
        """
        sub_question_ids = [sq.sub_question_id for sq in decomposed_topic.sub_questions]
        evidence_k = effective_evidence_k(
            self.generation.evidence_k,
            len(sub_question_ids),
            self.settings.evidence_per_sub_question,
        )
        if not self.settings.coverage_aware_evidence or not sub_question_ids:
            return ranked[:evidence_k]

        evidence = select_evidence(ranked, sub_question_ids, evidence_k)
        covered = {
            sq
            for sq in sub_question_ids
            if any(c.features.get(f"ain_hit_sq{sq}") for c in evidence)
        }
        logger.info(
            "Topic %s: evidence %d docs covering %d/%d sub-questions",
            decomposed_topic.topic_id, len(evidence), len(covered), len(sub_question_ids),
        )
        return evidence

    def _select_needs(
        self, decomposed_topic: DecomposedTopic
    ) -> list[tuple[AtomicNeed, Topic]]:
        """Needs paired with their synthetic topics, capped by `max_ains`.

        Capping round-robins across sub-questions rather than truncating the
        flat list. The list is ordered by sub-question, so truncation would drop
        a narrative's last sub-questions entirely — precisely the failure that
        decomposition exists to prevent.
        """
        ain_topics = decomposed_topic.ain_topics(self.settings.query_variant)
        paired = list(zip(decomposed_topic.needs, ain_topics))
        cap = self.settings.max_ains
        if cap is None or len(paired) <= cap:
            return paired

        by_sub_question: dict[str, list[tuple[AtomicNeed, Topic]]] = {}
        for need, ain_topic in paired:
            by_sub_question.setdefault(need.sub_question_id, []).append((need, ain_topic))

        selected: list[tuple[AtomicNeed, Topic]] = []
        queues = list(by_sub_question.values())
        while len(selected) < cap and any(queues):
            for queue in queues:
                if not queue:
                    continue
                selected.append(queue.pop(0))
                if len(selected) >= cap:
                    break
        logger.warning(
            "Topic %s: capped %d needs to %d (round-robin across %d sub-questions)",
            decomposed_topic.topic_id, len(paired), len(selected), len(queues),
        )
        return selected

    def _retrieve_ain(self, ain_topic: Topic) -> list[Candidate]:
        """One need through the cascade. A failure costs that need, not the topic.

        Same posture as `run_pipeline.py`'s per-topic guard: with ~10 needs per
        narrative and a search API that intermittently drops responses, letting
        one need abort the narrative would trade a slightly thinner pool for no
        submission row at all.
        """
        try:
            return self.ain_cascade.retrieve(ain_topic)[: self.settings.per_ain_top_k]
        except Exception as exc:  # noqa: BLE001 - one need must not end the topic
            logger.error(
                "AIN %s FAILED (%s: %s); continuing without it",
                ain_topic.topic_id, type(exc).__name__, exc,
            )
            return []
