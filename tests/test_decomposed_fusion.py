import unittest

from trecrag.decomposed.fusion import AinRun, fuse_ain_runs, sub_question_feature
from trecrag.retrieval.hybrid import reciprocal_rank_fusion
from trecrag.schema import Candidate


def _cands(*docids, text_for=None):
    return [
        Candidate(
            docid=docid,
            score=float(100 - i),
            rank=i + 1,
            text=(text_for or {}).get(docid),
        )
        for i, docid in enumerate(docids)
    ]


_NEXT_AIN = [0]


def _run(*docids, text_for=None, sq="1"):
    """One need's run, with a distinct ain_id per call."""
    _NEXT_AIN[0] += 1
    return AinRun(sq, _NEXT_AIN[0], _cands(*docids, text_for=text_for))


class TestNeedProvenance(unittest.TestCase):
    def test_n_ain_hits_counts_distinct_needs(self):
        fused = fuse_ain_runs([_run("a", "b"), _run("a", "c"), _run("a")])
        by_docid = {c.docid: c for c in fused}
        self.assertEqual(by_docid["a"].features["n_ain_hits"], 3.0)
        self.assertEqual(by_docid["b"].features["n_ain_hits"], 1.0)
        self.assertEqual(by_docid["c"].features["n_ain_hits"], 1.0)

    def test_repeats_within_one_need_count_once(self):
        # A single run listing the same docid twice is one need's opinion, not two.
        run = _run("a", "b")
        run.candidates.append(Candidate(docid="a", score=1.0, rank=3))
        fused = fuse_ain_runs([run])
        self.assertEqual({c.docid: c.features["n_ain_hits"] for c in fused}["a"], 1.0)

    def test_composite_run_does_not_inflate_n_ain_hits(self):
        fused = fuse_ain_runs([_run("a")], composite_run=_cands("a", "z"))
        by_docid = {c.docid: c for c in fused}
        self.assertEqual(by_docid["a"].features["n_ain_hits"], 1.0)
        self.assertEqual(by_docid["z"].features["n_ain_hits"], 0.0)

    def test_from_composite_flags_the_baseline_arm(self):
        fused = fuse_ain_runs([_run("a", "b")], composite_run=_cands("b", "z"))
        by_docid = {c.docid: c for c in fused}
        self.assertEqual(by_docid["a"].features["from_composite"], 0.0)
        self.assertEqual(by_docid["b"].features["from_composite"], 1.0)
        self.assertEqual(by_docid["z"].features["from_composite"], 1.0)

    def test_composite_only_documents_survive_into_an_uncapped_pool(self):
        # With no top_k the fused list keeps every composite-only document. A
        # pool cap can still cut them, so this is not a floor on the final list.
        fused = fuse_ain_runs([_run("a")], composite_run=_cands("z"))
        self.assertIn("z", {c.docid for c in fused})


class TestSubQuestionProvenance(unittest.TestCase):
    """What `evidence.py` selects on: which *asks* a document speaks to."""

    def test_marks_each_contributing_sub_question(self):
        fused = fuse_ain_runs(
            [_run("a", sq="1"), _run("a", sq="2"), _run("b", sq="2")]
        )
        by_docid = {c.docid: c for c in fused}
        self.assertEqual(by_docid["a"].features[sub_question_feature("1")], 1.0)
        self.assertEqual(by_docid["a"].features[sub_question_feature("2")], 1.0)
        self.assertNotIn(sub_question_feature("1"), by_docid["b"].features)
        self.assertEqual(by_docid["b"].features[sub_question_feature("2")], 1.0)

    def test_counts_distinct_sub_questions_not_needs(self):
        # Three needs, all belonging to sub-question 1.
        fused = fuse_ain_runs([_run("a", sq="1"), _run("a", sq="1"), _run("a", sq="1")])
        candidate = fused[0]
        self.assertEqual(candidate.features["n_ain_hits"], 3.0)
        self.assertEqual(candidate.features["n_sub_question_hits"], 1.0)

    def test_features_are_floats_so_they_survive_checkpointing(self):
        fused = fuse_ain_runs([_run("a", sq="2")])
        self.assertTrue(all(isinstance(v, float) for v in fused[0].features.values()))

    def test_composite_documents_carry_no_sub_question_coverage(self):
        fused = fuse_ain_runs([_run("a", sq="1")], composite_run=_cands("z"))
        by_docid = {c.docid: c for c in fused}
        self.assertEqual(by_docid["z"].features["n_sub_question_hits"], 0.0)


class TestFusionMechanics(unittest.TestCase):
    def test_text_survives_the_merge(self):
        runs = [_run("a", text_for={"a": "the passage text"}), _run("a", "b")]
        self.assertEqual({c.docid: c.text for c in fuse_ain_runs(runs)}["a"], "the passage text")

    def test_text_found_only_by_a_later_run_is_kept(self):
        runs = [_run("a"), _run("a", text_for={"a": "found later"})]
        self.assertEqual({c.docid: c.text for c in fuse_ain_runs(runs)}["a"], "found later")

    def test_ranks_renumbered_and_scores_non_increasing(self):
        fused = fuse_ain_runs([_run("a", "b", "c"), _run("c", "b")])
        self.assertEqual([c.rank for c in fused], list(range(1, len(fused) + 1)))
        scores = [c.score for c in fused]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_truncates(self):
        self.assertEqual(len(fuse_ain_runs([_run("a", "b", "c", "d")], top_k=2)), 2)

    def test_empty_runs_are_skipped(self):
        fused = fuse_ain_runs([AinRun("1", 1, []), _run("a"), AinRun("1", 2, [])])
        self.assertEqual([c.docid for c in fused], ["a"])

    def test_no_input_yields_no_output(self):
        self.assertEqual(fuse_ain_runs([]), [])
        self.assertEqual(fuse_ain_runs([AinRun("1", 1, [])]), [])

    def test_agreement_across_needs_outranks_a_single_strong_hit(self):
        # b is 2nd for two needs; a is 1st for one. Agreement should win.
        fused = fuse_ain_runs([_run("a", "b"), _run("c", "b")])
        self.assertEqual(fused[0].docid, "b")


class TestCompositeWeighting(unittest.TestCase):
    def test_raising_composite_weight_promotes_its_documents(self):
        runs = [_run("a"), _run("a"), _run("a")]
        self.assertEqual(fuse_ain_runs(runs, composite_run=_cands("z"))[0].docid, "a")

        runs = [_run("a"), _run("a"), _run("a")]
        weighted = fuse_ain_runs(runs, composite_run=_cands("z"), composite_weight=10.0)
        self.assertEqual(weighted[0].docid, "z")

    def test_weight_zero_makes_the_composite_arm_inert_for_ordering(self):
        weighted = fuse_ain_runs([_run("a")], composite_run=_cands("z"), composite_weight=0.0)
        by_docid = {c.docid: c for c in weighted}
        self.assertEqual(by_docid["z"].score, 0.0)
        self.assertEqual(by_docid["a"].rank, 1)


class TestRrfWeightsBackwardCompatibility(unittest.TestCase):
    """The `weights` argument must not change existing callers' behaviour."""

    def test_omitting_weights_matches_explicit_ones(self):
        lists = [_cands("a", "b", "c"), _cands("c", "a")]
        default = reciprocal_rank_fusion([list(x) for x in lists])
        explicit = reciprocal_rank_fusion([list(x) for x in lists], weights=[1.0, 1.0])
        self.assertEqual(
            [(c.docid, round(c.score, 12)) for c in default],
            [(c.docid, round(c.score, 12)) for c in explicit],
        )

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([_cands("a"), _cands("b")], weights=[1.0])


if __name__ == "__main__":
    unittest.main()
