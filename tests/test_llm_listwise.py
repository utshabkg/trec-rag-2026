import unittest

from tests.fakes import ReversingLLMClient
from trecrag.clients.llm_client import IdentityLLMClient, LLMClient
from trecrag.rerank.llm_listwise import LLMListwiseReranker
from trecrag.schema import Candidate, Topic


class _FixedResponseLLMClient(LLMClient):
    def __init__(self, response: str):
        self._response = response

    def complete(self, prompt: str, *, max_tokens: int = 1024) -> str:
        return self._response


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
        self.assertIsNone(order)

    def test_parse_order_appends_missing_indices(self):
        order = LLMListwiseReranker._parse_order("[2] > [1]", n=3)
        self.assertEqual(order, [1, 0, 2])

    def test_parse_order_prefers_bracketed_indices_over_incidental_prose_numbers(self):
        order = LLMListwiseReranker._parse_order("I found 5 relevant results. [2] > [1] > [3]", n=3)
        self.assertEqual(order, [1, 0, 2])

    def test_parse_order_accepts_bare_integers_covering_the_whole_window(self):
        order = LLMListwiseReranker._parse_order("3 > 1 > 2", n=3)
        self.assertEqual(order, [2, 0, 1])

    def test_parse_order_rejects_sparse_bare_integers(self):
        order = LLMListwiseReranker._parse_order("I think option number 4 might work.", n=10)
        self.assertIsNone(order)

    def test_rerank_falls_back_to_input_order_and_logs_warning_on_unparsable_response(self):
        candidates = _candidates(5)
        llm = _FixedResponseLLMClient("I read passage 2 and thought it was fine.")
        reranker = LLMListwiseReranker(llm, window_size=5, step=5)
        with self.assertLogs("llm_listwise", level="WARNING"):
            reranked = reranker.rerank(Topic("t1", "q"), candidates, k=5)
        self.assertEqual([c.docid for c in reranked], [f"d{i}" for i in range(1, 6)])


if __name__ == "__main__":
    unittest.main()
