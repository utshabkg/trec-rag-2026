import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeRetriever
from trecrag.decomposed.orchestrator import DecomposedPipeline, FanoutSettings
from trecrag.decomposed.schema import AtomicNeed, DecomposedTopic, SubQuestion
from trecrag.generation.generator import ExtractiveGenerator
from trecrag.output.writers import build_rag_record, write_rag_jsonl, write_retrieval_run
from trecrag.pipeline import GenerationSettings, RagPipeline, StageDepths
from trecrag.rerank.base import Reranker
from trecrag.rerank.cross_encoder import CrossEncoderReranker, LexicalOverlapScorer
from trecrag.schema import Candidate, RagAnswer, Topic
from trecrag.validate import validate_rag_jsonl, validate_retrieval_run

_TOPIC_ID = "rag2026-0"


def _decomposed_topic(n_sub_questions=2, ains_per_sub_question=2) -> DecomposedTopic:
    needs, sub_questions = [], []
    for sq in range(1, n_sub_questions + 1):
        sub_questions.append(SubQuestion(str(sq), f"sub question {sq}"))
        for ain in range(1, ains_per_sub_question + 1):
            needs.append(
                AtomicNeed(
                    ain_id=ain,
                    sub_question_id=str(sq),
                    ain_text=f"need {sq}.{ain}",
                    expanded_query=f"expanded query {sq}.{ain}",
                )
            )
    return DecomposedTopic(
        topic_id=_TOPIC_ID,
        composite_text="the whole composite narrative",
        metadata="context about the asker",
        sub_questions=sub_questions,
        needs=needs,
    )


def _candidates(*docids):
    return [
        Candidate(docid=d, score=float(10 - i), rank=i + 1, text=f"text of {d} about nursing")
        for i, d in enumerate(docids)
    ]


def _cascade(candidates_by_topic) -> RagPipeline:
    return RagPipeline(
        bm25=FakeRetriever(candidates_by_topic),
        depths=StageDepths(
            bm25_hits=10, hybrid_top_k=10, ltr_top_k=10,
            colbert_top_k=10, ce_top_k=10, llm_listwise_top_k=10,
        ),
        generation=GenerationSettings(),
        generator=ExtractiveGenerator(),
        biencoder=None,
        cross_encoder=CrossEncoderReranker(LexicalOverlapScorer()),
        llm_listwise=None,
    )


class CountingListwise(Reranker):
    """Stands in for the LLM listwise stage and records how it was called."""

    name = "counting_listwise"

    def __init__(self):
        self.calls = 0
        self.seen_queries = []
        self.seen_sizes = []

    def rerank(self, topic, candidates, k):
        self.calls += 1
        self.seen_queries.append(topic.text)
        self.seen_sizes.append(len(candidates))
        top = candidates[:k]
        for rank, candidate in enumerate(top, start=1):
            candidate.rank = rank
            candidate.score = float(len(top) - rank + 1)
        return top


class RecordingGenerator(ExtractiveGenerator):
    def __init__(self):
        self.seen_topic = None
        self.seen_evidence = None

    def generate(self, topic, evidence, max_words, max_citations):
        self.seen_topic = topic
        self.seen_evidence = list(evidence)
        return super().generate(topic, evidence, max_words, max_citations)


def _build(
    decomposed_topic=None,
    settings=None,
    listwise=None,
    generator=None,
    include_composite=True,
    ain_docs=None,
    composite_docs=("z1", "z2"),
):
    decomposed_topic = decomposed_topic or _decomposed_topic()
    ain_topics = decomposed_topic.ain_topics()
    if ain_docs is None:
        # Each need finds one private document plus a shared one, so fusion has
        # both agreement and disagreement to resolve.
        ain_docs = {
            t.topic_id: _candidates(f"d{i}", "shared")
            for i, t in enumerate(ain_topics)
        }
    cascade_docs = dict(ain_docs)
    cascade_docs[_TOPIC_ID] = _candidates(*composite_docs)

    cascade = _cascade(cascade_docs)
    return DecomposedPipeline(
        ain_cascade=cascade,
        composite_cascade=cascade if include_composite else None,
        llm_listwise=listwise,
        generator=generator or ExtractiveGenerator(),
        decomposed={_TOPIC_ID: decomposed_topic},
        settings=settings
        or FanoutSettings(include_composite_run=include_composite, final_top_k=50),
        generation=GenerationSettings(evidence_k=3),
    )


class TestFanout(unittest.TestCase):
    def test_pool_contains_documents_from_every_need(self):
        pipeline = _build()
        ranked = pipeline.retrieve(Topic(_TOPIC_ID, "composite"))
        docids = {c.docid for c in ranked}
        for i in range(4):  # 2 sub-questions x 2 needs
            self.assertIn(f"d{i}", docids)

    def test_composite_documents_are_included_when_enabled(self):
        ranked = _build().retrieve(Topic(_TOPIC_ID, "composite"))
        self.assertTrue({"z1", "z2"} <= {c.docid for c in ranked})

    def test_composite_documents_are_absent_when_disabled(self):
        pipeline = _build(include_composite=False)
        ranked = pipeline.retrieve(Topic(_TOPIC_ID, "composite"))
        self.assertFalse({"z1", "z2"} & {c.docid for c in ranked})

    def test_document_found_by_every_need_is_marked_and_ranked_first(self):
        ranked = _build(include_composite=False).retrieve(Topic(_TOPIC_ID, "composite"))
        self.assertEqual(ranked[0].docid, "shared")
        self.assertEqual(ranked[0].features["n_ain_hits"], 4.0)

    def test_unknown_topic_raises(self):
        with self.assertRaises(KeyError):
            _build().retrieve(Topic("rag2026-999", "not decomposed"))


class TestListwiseRunsOnceOnThePool(unittest.TestCase):
    """The entire runtime argument for this design."""

    def test_listwise_is_called_once_not_per_need(self):
        listwise = CountingListwise()
        pipeline = _build(listwise=listwise)
        pipeline.retrieve(Topic(_TOPIC_ID, "composite"))
        self.assertEqual(listwise.calls, 1)

    def test_listwise_scales_with_needs_not_at_all(self):
        for n_needs in (2, 8):
            listwise = CountingListwise()
            topic = _decomposed_topic(n_sub_questions=1, ains_per_sub_question=n_needs)
            _build(decomposed_topic=topic, listwise=listwise).retrieve(
                Topic(_TOPIC_ID, "composite")
            )
            self.assertEqual(listwise.calls, 1, f"{n_needs} needs")

    def test_listwise_orders_against_the_composite_narrative(self):
        # Not against any single need: the pooled list answers the whole question.
        listwise = CountingListwise()
        _build(listwise=listwise).retrieve(Topic(_TOPIC_ID, "composite"))
        self.assertEqual(listwise.seen_queries, ["the whole composite narrative"])

    def test_listwise_sees_the_pool_not_a_single_run(self):
        listwise = CountingListwise()
        _build(listwise=listwise).retrieve(Topic(_TOPIC_ID, "composite"))
        self.assertGreater(listwise.seen_sizes[0], 4)


class TestResilience(unittest.TestCase):
    def test_one_failing_need_does_not_cost_the_topic(self):
        class ExplodingOnOneTopic(FakeRetriever):
            def retrieve(self, topic, k):
                if topic.topic_id.endswith("#sq1-ain1"):
                    raise RuntimeError("simulated API failure")
                return super().retrieve(topic, k)

        decomposed_topic = _decomposed_topic()
        docs = {
            t.topic_id: _candidates(f"d{i}")
            for i, t in enumerate(decomposed_topic.ain_topics())
        }
        docs[_TOPIC_ID] = _candidates("z1")
        cascade = _cascade(docs)
        cascade.bm25 = ExplodingOnOneTopic(docs)

        pipeline = DecomposedPipeline(
            ain_cascade=cascade,
            composite_cascade=cascade,
            generator=ExtractiveGenerator(),
            decomposed={_TOPIC_ID: decomposed_topic},
            settings=FanoutSettings(),
            generation=GenerationSettings(evidence_k=3),
        )
        ranked = pipeline.retrieve(Topic(_TOPIC_ID, "composite"))
        docids = {c.docid for c in ranked}
        self.assertNotIn("d0", docids)          # the failed need contributed nothing
        self.assertTrue({"d1", "d2", "d3"} <= docids)  # its siblings still did


class TestMaxAinsCap(unittest.TestCase):
    def test_cap_round_robins_across_sub_questions(self):
        listwise = CountingListwise()
        topic = _decomposed_topic(n_sub_questions=3, ains_per_sub_question=4)
        docs = {t.topic_id: _candidates(f"d{i}") for i, t in enumerate(topic.ain_topics())}
        docs[_TOPIC_ID] = _candidates("z1")
        cascade = _cascade(docs)
        pipeline = DecomposedPipeline(
            ain_cascade=cascade,
            composite_cascade=None,
            llm_listwise=listwise,
            generator=ExtractiveGenerator(),
            decomposed={_TOPIC_ID: topic},
            settings=FanoutSettings(max_ains=3, include_composite_run=False),
            generation=GenerationSettings(evidence_k=3),
        )
        ranked = pipeline.retrieve(Topic(_TOPIC_ID, "composite"))
        # d0/d4/d8 are the first need of sub-questions 1/2/3. Truncating the flat
        # list would have taken d0/d1/d2 and dropped sub-questions 2 and 3.
        self.assertEqual({c.docid for c in ranked}, {"d0", "d4", "d8"})

    def test_no_cap_by_default(self):
        topic = _decomposed_topic(n_sub_questions=3, ains_per_sub_question=4)
        ranked = _build(decomposed_topic=topic, include_composite=False).retrieve(
            Topic(_TOPIC_ID, "composite")
        )
        self.assertEqual(len({c.docid for c in ranked}), 13)  # 12 private + shared


class TestGenerationContract(unittest.TestCase):
    def test_generator_answers_the_composite_narrative(self):
        generator = RecordingGenerator()
        _build(generator=generator).run_topic(Topic(_TOPIC_ID, "whatever the runner passes"))
        self.assertTrue(generator.seen_topic.text.startswith("the whole composite narrative"))
        self.assertEqual(generator.seen_topic.topic_id, _TOPIC_ID)

    def test_prompt_carries_the_extracted_sub_questions(self):
        generator = RecordingGenerator()
        _build(generator=generator).run_topic(Topic(_TOPIC_ID, "composite"))
        self.assertIn("sub question 1", generator.seen_topic.text)
        self.assertIn("sub question 2", generator.seen_topic.text)

    def test_sub_questions_can_be_left_out(self):
        generator = RecordingGenerator()
        _build(
            generator=generator,
            settings=FanoutSettings(include_sub_questions_in_prompt=False),
        ).run_topic(Topic(_TOPIC_ID, "composite"))
        self.assertEqual(generator.seen_topic.text, "the whole composite narrative")

    def test_generator_receives_pooled_evidence_capped_at_evidence_k(self):
        generator = RecordingGenerator()
        _build(generator=generator).run_topic(Topic(_TOPIC_ID, "composite"))
        self.assertEqual(len(generator.seen_evidence), 3)

    def test_evidence_spans_sub_questions_rather_than_taking_the_pool_head(self):
        # Sub-question 2's documents are deliberately weak: only its own need
        # finds them, so fusion ranks them below sq1's agreed-upon documents.
        topic = _decomposed_topic(n_sub_questions=2, ains_per_sub_question=2)
        ain_topics = topic.ain_topics()
        docs = {
            ain_topics[0].topic_id: _candidates("sq1_a", "sq1_b", "sq1_c"),
            ain_topics[1].topic_id: _candidates("sq1_a", "sq1_b", "sq1_c"),
            ain_topics[2].topic_id: _candidates("sq2_only"),
            ain_topics[3].topic_id: _candidates("sq2_only"),
        }
        docs[_TOPIC_ID] = _candidates("sq1_a", "sq1_b", "sq1_c")
        cascade = _cascade(docs)

        generator = RecordingGenerator()
        pipeline = DecomposedPipeline(
            ain_cascade=cascade,
            composite_cascade=cascade,
            generator=generator,
            decomposed={_TOPIC_ID: topic},
            settings=FanoutSettings(),
            generation=GenerationSettings(evidence_k=2),
        )
        pipeline.run_topic(Topic(_TOPIC_ID, "composite"))
        cited = {c.docid for c in generator.seen_evidence}
        self.assertIn("sq2_only", cited, "sub-question 2 has no evidence at all")

    def test_run_topic_returns_a_ranked_list_and_an_answer(self):
        ranked, answer = _build().run_topic(Topic(_TOPIC_ID, "composite"))
        self.assertIsInstance(answer, RagAnswer)
        self.assertTrue(ranked)


class TestSubmissionInvariants(unittest.TestCase):
    def test_output_passes_the_run_file_validator(self):
        ranked, _ = _build(listwise=CountingListwise()).run_topic(
            Topic(_TOPIC_ID, "composite")
        )
        out = Path(tempfile.mkdtemp()) / "run.tsv"
        write_retrieval_run(out, "decomposed-test", {_TOPIC_ID: ranked})
        self.assertEqual(validate_retrieval_run(out, {_TOPIC_ID}), [])

    def test_ranks_are_contiguous_and_scores_non_increasing(self):
        ranked, _ = _build(listwise=CountingListwise()).run_topic(
            Topic(_TOPIC_ID, "composite")
        )
        self.assertEqual([c.rank for c in ranked], list(range(1, len(ranked) + 1)))
        scores = [c.score for c in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_duplicate_docids(self):
        ranked, _ = _build().run_topic(Topic(_TOPIC_ID, "composite"))
        docids = [c.docid for c in ranked]
        self.assertEqual(len(docids), len(set(docids)))

    def test_final_top_k_is_respected(self):
        topic = _decomposed_topic(n_sub_questions=4, ains_per_sub_question=5)
        pipeline = _build(
            decomposed_topic=topic,
            settings=FanoutSettings(final_top_k=5, include_composite_run=False),
            include_composite=False,
        )
        self.assertEqual(len(pipeline.retrieve(Topic(_TOPIC_ID, "composite"))), 5)

    def test_rag_output_passes_the_submission_validator(self):
        # The official topic text -- what the runner writes into the record --
        # is the TSV's, NOT the sub-question-augmented prompt text, and
        # `validate_rag_jsonl` compares it byte-for-byte.
        official = Topic(_TOPIC_ID, "the official narrative text from the TSV")
        _, answer = _build(listwise=CountingListwise()).run_topic(official)

        out = Path(tempfile.mkdtemp()) / "rag.jsonl"
        write_rag_jsonl(
            out,
            [build_rag_record("iris-mst", official, "decomposed-test", "desc", answer)],
        )
        self.assertEqual(validate_rag_jsonl(out, [official]), [])

    def test_augmented_prompt_never_leaks_into_the_submission(self):
        official = Topic(_TOPIC_ID, "the official narrative text from the TSV")
        _, answer = _build().run_topic(official)
        record = build_rag_record("iris-mst", official, "run", "desc", answer)
        self.assertEqual(record["metadata"]["narrative"], official.text)
        self.assertNotIn("This request asks", record["metadata"]["narrative"])


if __name__ == "__main__":
    unittest.main()
