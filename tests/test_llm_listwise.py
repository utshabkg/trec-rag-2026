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

    def test_rerank_depth_limits_llm_calls_but_keeps_every_candidate(self):
        """This stage is ~85% of runtime, one call per sliding window. Limiting it
        to the head must not drop candidates or break the run file's ordering
        invariants, since the tail is spliced back on a different score scale."""

        class CountingLLM(ReversingLLMClient):
            calls = 0

            def complete(self, prompt, *, max_tokens=1024):
                type(self).calls += 1
                return super().complete(prompt, max_tokens=max_tokens)

        candidates = [
            Candidate(docid=f"d{i}", score=100.0 - i, rank=i + 1, text=f"t{i}") for i in range(50)
        ]
        CountingLLM.calls = 0
        out = LLMListwiseReranker(
            CountingLLM(), window_size=20, step=10, rerank_depth=20
        ).rerank(Topic("t1", "q"), candidates, k=50)

        self.assertEqual(CountingLLM.calls, 1)  # 4 windows without rerank_depth
        self.assertEqual(len(out), 50)
        self.assertEqual(len({c.docid for c in out}), 50)
        self.assertEqual([c.rank for c in out], list(range(1, 51)))
        scores = [c.score for c in out]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
        # the untouched tail keeps its upstream (cross-encoder) ordering
        self.assertEqual([c.docid for c in out[20:]], [f"d{i}" for i in range(20, 50)])

    def test_parse_order_rejects_a_response_that_is_not_a_ranking(self):
        """None means 'keep the upstream order', not 'here is a ranking'.

        Back-filling a prose reply into a full permutation would present noise as
        a confident reordering of the last stage before the run file."""
        self.assertIsNone(LLMListwiseReranker._parse_order("not a real ranking", n=3))

    def test_parse_order_ignores_list_numbering(self):
        """"1. [7]" must rank passage 7 first -- not passage 1."""
        order = LLMListwiseReranker._parse_order("1. [7]\n2. [3]\n3. [5]", n=8)
        self.assertEqual(order[:3], [6, 2, 4])

    def test_parse_order_ignores_digits_in_prose(self):
        order = LLMListwiseReranker._parse_order(
            "[7] > [3] > [5] because 2 of 3 criteria are met", n=8
        )
        self.assertEqual(order[:3], [6, 2, 4])

    def test_parse_order_falls_back_to_bare_integers(self):
        order = LLMListwiseReranker._parse_order("3 > 1 > 2", n=3)
        self.assertEqual(order, [2, 0, 1])

    def test_parse_order_appends_missing_indices(self):
        order = LLMListwiseReranker._parse_order("[2] > [1]", n=3)
        self.assertEqual(order, [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
