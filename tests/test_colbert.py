import unittest

from trecrag.rerank.colbert import ColBERTReranker, HashingTokenEncoder, max_sim
from trecrag.schema import Candidate, Topic


class TestMaxSim(unittest.TestCase):
    def test_empty_inputs_score_zero(self):
        encoder = HashingTokenEncoder(dims=32)
        empty = encoder.encode_tokens("", 10)
        non_empty = encoder.encode_tokens("some text", 10)
        self.assertEqual(max_sim(empty, non_empty), 0.0)
        self.assertEqual(max_sim(non_empty, empty), 0.0)

    def test_identical_text_scores_highest(self):
        encoder = HashingTokenEncoder(dims=64)
        query = encoder.encode_tokens("congestion pricing funding", 10)
        same = max_sim(query, encoder.encode_tokens("congestion pricing funding", 10))
        different = max_sim(query, encoder.encode_tokens("unrelated cooking recipe", 10))
        self.assertGreater(same, different)


class TestColBERTReranker(unittest.TestCase):
    def test_reranks_by_token_overlap(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding")
        candidates = [
            Candidate(docid="off", score=0.0, rank=1, text="unrelated cooking recipe content here"),
            Candidate(docid="on", score=0.0, rank=2, text="congestion pricing transit funding plan"),
        ]
        reranker = ColBERTReranker(HashingTokenEncoder(dims=256))
        reranked = reranker.rerank(topic, candidates, k=2)
        self.assertEqual(reranked[0].docid, "on")
        self.assertIn("colbert_score", reranked[0].features)

    def test_candidates_without_text_sort_last(self):
        topic = Topic(topic_id="t1", text="anything")
        candidates = [
            Candidate(docid="no_text", score=0.0, rank=1, text=None),
            Candidate(docid="has_text", score=0.0, rank=2, text="anything at all"),
        ]
        reranker = ColBERTReranker(HashingTokenEncoder(dims=32))
        reranked = reranker.rerank(topic, candidates, k=2)
        self.assertEqual(reranked[-1].docid, "no_text")

    def test_empty_input(self):
        reranker = ColBERTReranker(HashingTokenEncoder(dims=32))
        self.assertEqual(reranker.rerank(Topic("t1", "q"), [], k=5), [])


if __name__ == "__main__":
    unittest.main()
