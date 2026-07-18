import tempfile
import unittest
from pathlib import Path

from tests.fakes import FakeRetriever
from trecrag.clients.llm_client import IdentityLLMClient
from trecrag.generation.generator import ExtractiveGenerator
from trecrag.output.writers import build_rag_record, write_rag_jsonl, write_retrieval_run
from trecrag.pipeline import GenerationSettings, RagPipeline, StageDepths
from trecrag.rerank.colbert import ColBERTReranker, HashingTokenEncoder
from trecrag.rerank.cross_encoder import CrossEncoderReranker, LexicalOverlapScorer
from trecrag.rerank.llm_listwise import LLMListwiseReranker
from trecrag.rerank.ltr import LTRReranker
from trecrag.rerank.qder import HashingQDERScorer, QDERReranker
from trecrag.retrieval.biencoder import BiencoderRetriever, HashingEncoder
from trecrag.schema import Candidate, Topic
from trecrag.validate import validate_rag_jsonl, validate_retrieval_run

_DOC_TEXTS = {
    "d1": "Congestion pricing can provide dedicated, reliable revenue for transit improvements.",
    "d2": "The MTA and Albany must be held accountable for how congestion pricing revenue is spent.",
    "d3": "Environmental tradeoffs of congestion pricing include reduced emissions near the tolling zone.",
    "d4": "Small businesses worry congestion pricing increases delivery costs in city districts.",
    "d5": "A recipe for banana bread requires flour, sugar, and ripe bananas.",
    "d6": "The history of jazz music spans New Orleans, Chicago, and New York.",
    "d7": "Transit riders in outer boroughs depend on reliable bus and subway service.",
    "d8": "Congestion pricing programs abroad, such as London and Stockholm, cut traffic significantly.",
}


def _build_pipeline() -> RagPipeline:
    candidates = [
        Candidate(docid=docid, score=float(10 - i), rank=i + 1, text=text)
        for i, (docid, text) in enumerate(_DOC_TEXTS.items())
    ]
    bm25 = FakeRetriever({"rag2026-0": candidates})

    return RagPipeline(
        bm25=bm25,
        biencoder=BiencoderRetriever(HashingEncoder(dims=128)),
        ltr=LTRReranker(
            feature_names=["bm25_score", "bm25_rank", "bien_score", "rrf_score"],
            weights={"bm25_score": 0.3, "bien_score": 0.3, "rrf_score": 0.4},
        ),
        colbert=ColBERTReranker(HashingTokenEncoder(dims=128)),
        qder=QDERReranker(HashingQDERScorer(), hybrid_lambda=0.5),
        cross_encoder=CrossEncoderReranker(LexicalOverlapScorer()),
        llm_listwise=LLMListwiseReranker(IdentityLLMClient(), window_size=5, step=3),
        generator=ExtractiveGenerator(),
        depths=StageDepths(
            bm25_hits=8,
            hybrid_top_k=8,
            ltr_top_k=6,
            colbert_top_k=5,
            qder_top_k=4,
            ce_top_k=4,
            llm_listwise_top_k=3,
        ),
        generation=GenerationSettings(evidence_k=3, max_words=1024, max_citations_per_sentence=3),
    )


class TestRagPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.pipeline = _build_pipeline()
        self.topic = Topic(
            topic_id="rag2026-0",
            text="How should congestion pricing fund transit while addressing accountability and small business costs?",
        )

    def test_retrieve_cascade_shrinks_to_final_depth_with_valid_ranks(self):
        ranked = self.pipeline.retrieve(self.topic)
        self.assertEqual(len(ranked), 3)
        self.assertEqual([c.rank for c in ranked], [1, 2, 3])
        scores = [c.score for c in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_run_topic_produces_grounded_answer(self):
        ranked, rag_answer = self.pipeline.run_topic(self.topic)
        self.assertLessEqual(len(rag_answer.references), len(ranked))
        for sentence in rag_answer.sentences:
            for citation in sentence.citations:
                self.assertTrue(0 <= citation < len(rag_answer.references))
        total_words = sum(len(s.text.split()) for s in rag_answer.sentences)
        self.assertLessEqual(total_words, 1024)

    def test_full_submission_round_trip_passes_validation(self):
        ranked, rag_answer = self.pipeline.run_topic(self.topic)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            retrieval_path = tmp_path / "r_output_trec_rag_2026.tsv"
            rag_path = tmp_path / "rag_output_trec_rag_2026.jsonl"

            write_retrieval_run(retrieval_path, "test-run", {self.topic.topic_id: ranked})
            record = build_rag_record("test-team", self.topic, "test-run", "unit test run", rag_answer)
            write_rag_jsonl(rag_path, [record])

            retrieval_violations = validate_retrieval_run(retrieval_path, expected_topic_ids={self.topic.topic_id})
            rag_violations = validate_rag_jsonl(rag_path, topics=[self.topic])

        self.assertEqual(retrieval_violations, [])
        self.assertEqual(rag_violations, [])


if __name__ == "__main__":
    unittest.main()
