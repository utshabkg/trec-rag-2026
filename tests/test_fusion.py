"""Tests for feature-level rank fusion.

This stage exists because a cascade keeps only its last scorer's opinion: every
earlier stage re-sorted a pool that the next stage re-sorted from scratch, which
made both QDER and LTR byte-for-byte no-ops. The failure was silent -- each
stage ran, wrote a score, and changed nothing -- so the tests here pin the two
properties that make it detectable: a zero weight must be exactly inert, and a
non-zero weight must actually move the ranking.
"""

import unittest

from trecrag.rerank.fusion import FeatureFusionReranker
from trecrag.schema import Candidate, Topic

TOPIC = Topic(topic_id="t1", text="q")


def _candidates(spec):
    """spec: list of (docid, {feature: score})."""
    return [
        Candidate(docid=d, score=0.0, rank=i + 1, text="x", features=dict(f))
        for i, (d, f) in enumerate(spec)
    ]


class TestZeroWeightIsInert(unittest.TestCase):
    def test_zero_weight_signal_cannot_change_the_order(self):
        """The safety property: a signal we do not trust yet (an untrained model)
        must be exactly disable-able, not merely down-weighted."""
        spec = [
            ("a", {"ce_score": 3.0, "qder_score": 0.0}),
            ("b", {"ce_score": 2.0, "qder_score": 99.0}),
            ("c", {"ce_score": 1.0, "qder_score": 50.0}),
        ]
        ce_only = FeatureFusionReranker({"ce_score": 1.0, "qder_score": 0.0})
        order = [c.docid for c in ce_only.rerank(TOPIC, _candidates(spec), k=3)]
        self.assertEqual(order, ["a", "b", "c"])

    def test_single_signal_reproduces_that_signals_ranking(self):
        spec = [("a", {"ce_score": 1.0}), ("b", {"ce_score": 5.0}), ("c", {"ce_score": 3.0})]
        fusion = FeatureFusionReranker({"ce_score": 1.0})
        self.assertEqual(
            [c.docid for c in fusion.rerank(TOPIC, _candidates(spec), k=3)], ["b", "c", "a"]
        )


class TestNonZeroWeightIsLive(unittest.TestCase):
    def test_a_weighted_second_signal_changes_the_ranking(self):
        spec = [
            ("a", {"ce_score": 3.0, "qder_score": 0.0}),
            ("b", {"ce_score": 2.9, "qder_score": 99.0}),
        ]
        ce_only = [c.docid for c in FeatureFusionReranker({"ce_score": 1.0}).rerank(
            TOPIC, _candidates(spec), k=2)]
        fused = [c.docid for c in FeatureFusionReranker(
            {"ce_score": 1.0, "qder_score": 5.0}).rerank(TOPIC, _candidates(spec), k=2)]
        self.assertEqual(ce_only, ["a", "b"])
        self.assertEqual(fused, ["b", "a"])

    def test_fusion_is_by_rank_not_score_magnitude(self):
        """Cross-encoder logits ran all-negative in a real run while QDER's scale
        is arbitrary. Fusing raw scores would let one stage's spread dominate for
        reasons unrelated to accuracy, so only ordering may matter."""
        small = [("a", {"s": -5.44, "t": 1.0}), ("b", {"s": -1.84, "t": 2.0})]
        huge = [("a", {"s": -5440.0, "t": 1000.0}), ("b", {"s": -1840.0, "t": 2000.0})]
        fusion = FeatureFusionReranker({"s": 1.0, "t": 1.0})
        self.assertEqual(
            [c.docid for c in fusion.rerank(TOPIC, _candidates(small), k=2)],
            [c.docid for c in fusion.rerank(TOPIC, _candidates(huge), k=2)],
        )


class TestMissingSignals(unittest.TestCase):
    def test_a_candidate_missing_a_feature_is_not_scored_worst_for_it(self):
        """QDER skips text-less candidates. 'Not judged' must not be silently
        recorded as 'judged worst'."""
        spec = [
            ("judged_low", {"ce_score": 1.0, "qder_score": 0.1}),
            ("unjudged", {"ce_score": 1.0}),
        ]
        fusion = FeatureFusionReranker({"ce_score": 1.0, "qder_score": 1.0})
        out = fusion.rerank(TOPIC, _candidates(spec), k=2)
        self.assertEqual({c.docid for c in out}, {"judged_low", "unjudged"})

    def test_strict_mode_rejects_a_weighted_feature_nobody_has(self):
        """Weighting a feature no stage produces would fuse nothing while looking
        exactly like a working ensemble -- the same silent-no-op class of bug this
        module exists to fix."""
        spec = [("a", {"ce_score": 1.0})]
        with self.assertRaises(ValueError):
            FeatureFusionReranker({"ce_score": 1.0, "typo_score": 1.0}).rerank(
                TOPIC, _candidates(spec), k=1
            )

    def test_non_strict_mode_warns_and_continues(self):
        spec = [("a", {"ce_score": 1.0}), ("b", {"ce_score": 2.0})]
        fusion = FeatureFusionReranker({"ce_score": 1.0, "absent": 1.0}, strict=False)
        self.assertEqual([c.docid for c in fusion.rerank(TOPIC, _candidates(spec), k=2)], ["b", "a"])


class TestOutputInvariants(unittest.TestCase):
    def test_ranks_and_scores_satisfy_the_run_file_rules(self):
        spec = [(f"d{i}", {"ce_score": float(i)}) for i in range(10)]
        out = FeatureFusionReranker({"ce_score": 1.0}).rerank(TOPIC, _candidates(spec), k=5)
        self.assertEqual([c.rank for c in out], [1, 2, 3, 4, 5])
        scores = [c.score for c in out]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
        self.assertIn("fusion_score", out[0].features)

    def test_empty_input_and_empty_weights(self):
        self.assertEqual(FeatureFusionReranker({"ce_score": 1.0}).rerank(TOPIC, [], k=5), [])
        spec = [("a", {"ce_score": 1.0}), ("b", {"ce_score": 2.0})]
        passthrough = FeatureFusionReranker({}).rerank(TOPIC, _candidates(spec), k=1)
        self.assertEqual([c.docid for c in passthrough], ["a"])


if __name__ == "__main__":
    unittest.main()
