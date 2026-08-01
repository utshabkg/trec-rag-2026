import unittest

from trecrag.rerank.ltr import LTRReranker
from trecrag.schema import Candidate, Topic


class TestLTRReranker(unittest.TestCase):
    def test_linear_fallback_orders_by_weighted_features(self):
        topic = Topic(topic_id="t1", text="q")
        candidates = [
            Candidate(docid="low", score=0.0, rank=1, features={"bm25_score": 1.0, "rrf_score": 0.1}),
            Candidate(docid="high", score=0.0, rank=2, features={"bm25_score": 10.0, "rrf_score": 0.9}),
        ]
        ltr = LTRReranker(
            feature_names=["bm25_score", "rrf_score"],
            weights={"bm25_score": 0.5, "rrf_score": 0.5},
        )
        reranked = ltr.rerank(topic, candidates, k=2)
        self.assertEqual([c.docid for c in reranked], ["high", "low"])
        self.assertEqual(reranked[0].rank, 1)
        self.assertIn("ltr_score", reranked[0].features)

    def test_missing_feature_defaults_to_zero(self):
        topic = Topic(topic_id="t1", text="q")
        candidates = [
            Candidate(docid="a", score=0.0, rank=1, features={"bm25_score": 5.0}),
            Candidate(docid="b", score=0.0, rank=2, features={}),
        ]
        ltr = LTRReranker(feature_names=["bm25_score"], weights={"bm25_score": 1.0})
        reranked = ltr.rerank(topic, candidates, k=2)
        self.assertEqual(reranked[0].docid, "a")

    def test_truncates_to_k(self):
        topic = Topic(topic_id="t1", text="q")
        candidates = [
            Candidate(docid=str(i), score=0.0, rank=i, features={"bm25_score": float(i)})
            for i in range(1, 11)
        ]
        ltr = LTRReranker(feature_names=["bm25_score"], weights={"bm25_score": 1.0})
        reranked = ltr.rerank(topic, candidates, k=3)
        self.assertEqual(len(reranked), 3)
        self.assertEqual([c.docid for c in reranked], ["10", "9", "8"])

    def test_empty_input(self):
        ltr = LTRReranker(feature_names=["bm25_score"])
        self.assertEqual(ltr.rerank(Topic("t1", "q"), [], k=10), [])


if __name__ == "__main__":
    unittest.main()
