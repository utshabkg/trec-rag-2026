import unittest

from trecrag.retrieval.hybrid import reciprocal_rank_fusion
from trecrag.schema import Candidate


def _cand(docid, rank, text=None, **features):
    return Candidate(docid=docid, score=0.0, rank=rank, text=text, features=features)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_single_list_passthrough_order(self):
        bm25 = [_cand("a", 1), _cand("b", 2), _cand("c", 3)]
        fused = reciprocal_rank_fusion([bm25])
        self.assertEqual([c.docid for c in fused], ["a", "b", "c"])
        self.assertEqual([c.rank for c in fused], [1, 2, 3])

    def test_document_present_in_both_lists_outranks_single_list_hits(self):
        bm25 = [_cand("a", 1), _cand("b", 2)]
        dense = [_cand("b", 1), _cand("c", 2)]
        fused = reciprocal_rank_fusion([bm25, dense], k_const=60)
        # b is the only doc retrieved by both lists, so its RRF score (sum of
        # two reciprocal ranks) beats a/c, which each only score from one list.
        self.assertEqual(fused[0].docid, "b")

    def test_top_k_truncates(self):
        bm25 = [_cand(str(i), i) for i in range(1, 11)]
        fused = reciprocal_rank_fusion([bm25], top_k=3)
        self.assertEqual(len(fused), 3)

    def test_merges_text_and_features_across_lists(self):
        bm25 = [_cand("a", 1, text="doc-a", bm25_score=5.0)]
        dense = [_cand("a", 1, bien_score=0.9)]
        fused = reciprocal_rank_fusion([bm25, dense])
        merged = fused[0]
        self.assertEqual(merged.text, "doc-a")
        self.assertEqual(merged.features["bm25_score"], 5.0)
        self.assertEqual(merged.features["bien_score"], 0.9)
        self.assertIn("rrf_score", merged.features)

    def test_scores_are_non_increasing(self):
        bm25 = [_cand(str(i), i) for i in range(1, 21)]
        dense = [_cand(str(i), 21 - i) for i in range(1, 21)]
        fused = reciprocal_rank_fusion([bm25, dense])
        scores = [c.score for c in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
