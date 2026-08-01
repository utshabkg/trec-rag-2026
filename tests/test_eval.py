"""Retrieval-metric tests, pinned to hand-computed values.

A metric that is subtly wrong is worse than no metric: it produces confident
numbers that silently rank configurations backwards, and nothing downstream can
detect it. Each expected value here is derived by hand in the test rather than
recorded from a previous run of this same code.
"""

import math
import tempfile
import unittest
from pathlib import Path

from trecrag.eval import (
    average_precision,
    evaluate_run,
    judged_at_k,
    load_qrels,
    ndcg_at_k,
    recall_at_k,
)


class TestNDCG(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        judgements = {"a": 3, "b": 2, "c": 1}
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], judgements, 3), 1.0)

    def test_reversed_ranking_scores_below_one(self):
        judgements = {"a": 3, "b": 2, "c": 1}
        self.assertLess(ndcg_at_k(["c", "b", "a"], judgements, 3), 1.0)

    def test_matches_hand_computation(self):
        """gain = 2^rel - 1, discount = log2(rank+1)."""
        judgements = {"a": 3, "b": 0, "c": 1}
        ranking = ["a", "b", "c"]
        dcg = (2**3 - 1) / math.log2(2) + 0.0 / math.log2(3) + (2**1 - 1) / math.log2(4)
        idcg = (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3) + 0.0 / math.log2(4)
        self.assertAlmostEqual(ndcg_at_k(ranking, judgements, 3), dcg / idcg)

    def test_unjudged_documents_contribute_nothing(self):
        judgements = {"a": 3}
        self.assertEqual(
            ndcg_at_k(["a", "unjudged"], judgements, 2), ndcg_at_k(["a"], judgements, 2)
        )

    def test_k_truncates(self):
        judgements = {"a": 0, "b": 3}
        self.assertEqual(ndcg_at_k(["a", "b"], judgements, 1), 0.0)

    def test_empty_judgements_is_zero_not_a_crash(self):
        self.assertEqual(ndcg_at_k(["a"], {}, 10), 0.0)

    def test_linear_gain_option(self):
        judgements = {"a": 3, "b": 1}
        linear = ndcg_at_k(["b", "a"], judgements, 2, exponential=False)
        expo = ndcg_at_k(["b", "a"], judgements, 2, exponential=True)
        self.assertNotAlmostEqual(linear, expo)


class TestRecall(unittest.TestCase):
    def test_counts_only_documents_at_or_above_the_grade_threshold(self):
        judgements = {"a": 0, "b": 1, "c": 3}
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], judgements, 3, relevant_grade=1), 1.0)
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], judgements, 3, relevant_grade=2), 1.0)
        self.assertAlmostEqual(recall_at_k(["a", "b"], judgements, 2, relevant_grade=2), 0.0)

    def test_partial_recall(self):
        judgements = {"a": 1, "b": 1, "c": 1, "d": 1}
        self.assertAlmostEqual(recall_at_k(["a", "b"], judgements, 2), 0.5)

    def test_no_relevant_documents_is_zero(self):
        self.assertEqual(recall_at_k(["a"], {"a": 0}, 10, relevant_grade=1), 0.0)


class TestAveragePrecision(unittest.TestCase):
    def test_matches_hand_computation(self):
        judgements = {"a": 1, "c": 1}          # 2 relevant
        ranking = ["a", "b", "c"]              # hits at ranks 1 and 3
        self.assertAlmostEqual(average_precision(ranking, judgements), (1 / 1 + 2 / 3) / 2)

    def test_perfect_ranking_scores_one(self):
        self.assertAlmostEqual(average_precision(["a", "b"], {"a": 1, "b": 1}), 1.0)


class TestJudgedAtK(unittest.TestCase):
    def test_reports_pool_coverage(self):
        """The number to read first: low coverage means the other metrics are a
        lower bound, because unjudged is scored as non-relevant."""
        self.assertAlmostEqual(judged_at_k(["a", "b", "x", "y"], {"a": 1, "b": 0}, 4), 0.5)
        self.assertAlmostEqual(judged_at_k(["x", "y"], {"a": 1}, 2), 0.0)

    def test_counts_grade_zero_as_judged(self):
        """Grade 0 means 'assessed, not relevant' -- that IS coverage."""
        self.assertAlmostEqual(judged_at_k(["a"], {"a": 0}, 1), 1.0)


class TestLoadQrels(unittest.TestCase):
    def test_parses_trec_format_and_skips_junk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.qrels"
            path.write_text("14 0 shard_1 4\n14 0 shard_2 0\n\nbadline\n31 0 shard_3 2\n")
            qrels = load_qrels(path)
        self.assertEqual(qrels["14"], {"shard_1": 4, "shard_2": 0})
        self.assertEqual(qrels["31"], {"shard_3": 2})


class TestEvaluateRun(unittest.TestCase):
    QRELS = {"t1": {"a": 3, "b": 1}, "t2": {"c": 2}}

    def test_macro_averages_over_shared_topics_only(self):
        run = {"t1": ["a", "b"], "t2": ["c"], "t3": ["z"]}   # t3 absent from qrels
        results = evaluate_run(run, self.QRELS)
        self.assertEqual(results["topics"], 2)
        self.assertAlmostEqual(results["ndcg@10"], 1.0)

    def test_partial_run_reports_a_smaller_topic_count(self):
        """A run covering fewer topics must be visibly partial, not silently
        flattered by averaging over its own subset."""
        results = evaluate_run({"t1": ["a", "b"]}, self.QRELS)
        self.assertEqual(results["topics"], 1)

    def test_no_overlap_returns_zero_topics(self):
        self.assertEqual(evaluate_run({"zz": ["a"]}, self.QRELS), {"topics": 0})

    def test_reports_every_expected_metric(self):
        results = evaluate_run({"t1": ["a", "b"]}, self.QRELS)
        for name in ("ndcg@10", "ndcg@50", "recall@10", "recall@50", "map", "judged@10"):
            self.assertIn(name, results)


if __name__ == "__main__":
    unittest.main()
