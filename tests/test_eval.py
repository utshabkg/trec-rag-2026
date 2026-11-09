"""Tests for the development-set evaluators.

An evaluator's failure mode is the worst kind: it does not crash, it just reports
a number. Every wrong-but-plausible score here would be acted on -- a config
change kept or reverted on the strength of it -- so the parsing and the
relevance-threshold logic are pinned rather than trusted.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "eval"))

from evaluate_nuggets import parse_judgments, score_topic  # noqa: E402
from evaluate_retrieval import load_qrels, load_run  # noqa: E402


class TestLoadQrels(unittest.TestCase):
    def _write(self, tmp, text):
        path = Path(tmp) / "x.qrels"
        path.write_text(text, encoding="utf-8")
        return path

    def test_splits_graded_and_binary_at_the_threshold(self):
        """Graded labels drive nDCG; binary drives recall/MAP. Feeding graded
        labels to recall would count grade-1 ('related but unhelpful') as a hit."""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "14 0 d0 0\n14 0 d1 1\n14 0 d2 2\n14 0 d3 3\n")
            graded, binary = load_qrels(path, rel_threshold=2)
        self.assertEqual(graded["14"], {"d0": 0, "d1": 1, "d2": 2, "d3": 3})
        self.assertEqual(binary["14"], {"d0": 0, "d1": 0, "d2": 1, "d3": 1})

    def test_threshold_is_configurable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "14 0 d1 1\n14 0 d2 2\n")
            _, binary = load_qrels(path, rel_threshold=1)
        self.assertEqual(binary["14"], {"d1": 1, "d2": 1})

    def test_malformed_lines_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "14 0 d1 2\ngarbage\n\n14 0 d2 x\n14 0 d3 3\n")
            graded, _ = load_qrels(path, rel_threshold=2)
        self.assertEqual(sorted(graded["14"]), ["d1", "d3"])


class TestLoadRun(unittest.TestCase):
    def test_reads_six_column_trec_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.trec"
            path.write_text("14 Q0 d1 1 9.5 run\n14 Q0 d2 2 8.5 run\n", encoding="utf-8")
            run = load_run(path)
        self.assertEqual(run["14"], {"d1": 9.5, "d2": 8.5})

    def test_wrong_column_count_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.trec"
            path.write_text("14 Q0 d1 1 9.5 run\nbad line here\n", encoding="utf-8")
            self.assertEqual(len(load_run(path)["14"]), 1)


class TestParseJudgments(unittest.TestCase):
    def test_accepts_common_label_formats(self):
        for raw in ("1: support\n2: no\n3: partial", "1. support\n2. no\n3. partial"):
            self.assertEqual(len(parse_judgments(raw, 3)), 3)

    def test_normalises_not_support_to_no(self):
        self.assertEqual(parse_judgments("1: not_support\n2: no\n3: no", 3)[1], "no")

    def test_tolerates_a_preamble(self):
        self.assertEqual(len(parse_judgments("Here you go:\n1: support\n2: no\n3: no", 3)), 3)

    def test_rejects_a_truncated_judgment_rather_than_backfilling(self):
        """Back-filling the missing nuggets as 'no' would invent misses and
        silently understate coverage."""
        self.assertIsNone(parse_judgments("1: support\n2: no", 10))

    def test_rejects_garbage_and_empty(self):
        self.assertIsNone(parse_judgments("I cannot judge this.", 5))
        self.assertIsNone(parse_judgments("", 3))
        self.assertIsNone(parse_judgments(None, 3))


class TestScoreTopic(unittest.TestCase):
    NUGGETS = [
        {"importance": "vital", "text": "a"},
        {"importance": "vital", "text": "b"},
        {"importance": "okay", "text": "c"},
    ]

    def test_partial_counts_toward_coverage_but_not_strict(self):
        """Folding 'partial' into support is exactly how a coverage number gets
        quietly inflated, so the two are reported separately."""
        scores = score_topic({1: "support", 2: "partial", 3: "no"}, self.NUGGETS)
        self.assertAlmostEqual(scores["vital_coverage"], 1.0)
        self.assertAlmostEqual(scores["vital_strict"], 0.5)
        self.assertAlmostEqual(scores["all_coverage"], 2 / 3)

    def test_vital_and_all_are_computed_over_different_sets(self):
        scores = score_topic({1: "no", 2: "no", 3: "support"}, self.NUGGETS)
        self.assertAlmostEqual(scores["vital_coverage"], 0.0)
        self.assertAlmostEqual(scores["all_coverage"], 1 / 3)
        self.assertEqual(scores["n_vital"], 2)
        self.assertEqual(scores["n_nuggets"], 3)

    def test_missing_labels_count_as_not_covered(self):
        self.assertAlmostEqual(score_topic({}, self.NUGGETS)["all_coverage"], 0.0)

    def test_topic_with_no_vital_nuggets_reports_none_not_zero(self):
        """None means 'not applicable'; zero would drag the mean down."""
        scores = score_topic({1: "support"}, [{"importance": "okay", "text": "a"}])
        self.assertIsNone(scores["vital_coverage"])
        self.assertAlmostEqual(scores["all_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
