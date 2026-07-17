import unittest

from tests.fakes import ReversingLLMClient
from trecrag.clients.llm_client import IdentityLLMClient
from trecrag.rerank.llm_listwise import LLMListwiseReranker
from trecrag.schema import Candidate, Topic


def _candidates(n):
    return [Candidate(docid=f"d{i}", score=0.0, rank=i, text=f"passage {i}") for i in range(1, n + 1)]


class TestLLMListwiseReranker(unittest.TestCase):
    def test_identity_client_preserves_order(self):
        candidates = _candidates(6)
        reranker = LLMListwiseReranker(IdentityLLMClient(), window_size=3, step=2)
        reranked = reranker.rerank(Topic("t1", "q"), candidates, k=6)
        self.assertEqual([c.docid for c in reranked], [f"d{i}" for i in range(1, 7)])

    def test_single_window_reversal(self):
        candidates = _candidates(5)
        reranker = LLMListwiseReranker(ReversingLLMClient(), window_size=5, step=5)
        reranked = reranker.rerank(Topic("t1", "q"), candidates, k=5)
        self.assertEqual([c.docid for c in reranked], ["d5", "d4", "d3", "d2", "d1"])

    def test_truncates_to_k_and_assigns_rank(self):
        candidates = _candidates(10)
        reranker = LLMListwiseReranker(IdentityLLMClient(), window_size=4, step=2)
        reranked = reranker.rerank(Topic("t1", "q"), candidates, k=3)
        self.assertEqual(len(reranked), 3)
        self.assertEqual([c.rank for c in reranked], [1, 2, 3])

    def test_empty_input(self):
        reranker = LLMListwiseReranker(IdentityLLMClient())
        self.assertEqual(reranker.rerank(Topic("t1", "q"), [], k=5), [])

    def test_parse_order_ignores_malformed_response(self):
        order = LLMListwiseReranker._parse_order("not a real ranking", n=3)
        self.assertEqual(order, [0, 1, 2])

    def test_parse_order_appends_missing_indices(self):
        order = LLMListwiseReranker._parse_order("[2] > [1]", n=3)
        self.assertEqual(order, [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
