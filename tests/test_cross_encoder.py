import unittest

from trecrag.rerank.cross_encoder import CrossEncoderReranker, LexicalOverlapScorer
from trecrag.schema import Candidate, Topic


class TestLexicalOverlapScorer(unittest.TestCase):
    def test_no_overlap_scores_zero(self):
        scorer = LexicalOverlapScorer()
        self.assertEqual(scorer.score("congestion pricing", "unrelated cooking recipe"), 0.0)

    def test_full_overlap_scores_higher_than_partial(self):
        scorer = LexicalOverlapScorer()
        full = scorer.score("congestion pricing funding", "congestion pricing funding")
        partial = scorer.score("congestion pricing funding", "congestion only")
        self.assertGreater(full, partial)


class TestCrossEncoderReranker(unittest.TestCase):
    def test_reranks_by_pair_score(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding")
        candidates = [
            Candidate(docid="off", score=0.0, rank=1, text="unrelated cooking content"),
            Candidate(docid="on", score=0.0, rank=2, text="congestion pricing transit funding"),
        ]
        reranker = CrossEncoderReranker(LexicalOverlapScorer())
        reranked = reranker.rerank(topic, candidates, k=2)
        self.assertEqual(reranked[0].docid, "on")
        self.assertIn("ce_score", reranked[0].features)

    def test_empty_input(self):
        reranker = CrossEncoderReranker(LexicalOverlapScorer())
        self.assertEqual(reranker.rerank(Topic("t1", "q"), [], k=5), [])


if __name__ == "__main__":
    unittest.main()
