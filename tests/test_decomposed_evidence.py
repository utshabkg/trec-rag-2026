import unittest

from trecrag.decomposed.evidence import effective_evidence_k, select_evidence
from trecrag.decomposed.fusion import sub_question_feature
from trecrag.schema import Candidate


def _pool(*spec):
    """`spec` is (docid, [sub_question_ids]) in pool-rank order."""
    pool = []
    for rank, (docid, sub_question_ids) in enumerate(spec, start=1):
        features = {sub_question_feature(sq): 1.0 for sq in sub_question_ids}
        features["n_sub_question_hits"] = float(len(sub_question_ids))
        pool.append(
            Candidate(
                docid=docid, score=float(100 - rank), rank=rank,
                text=f"text {docid}", features=features,
            )
        )
    return pool


def _covered(evidence, sub_question_ids):
    return {
        sq for sq in sub_question_ids
        if any(c.features.get(sub_question_feature(sq)) for c in evidence)
    }


class TestCoverageSelection(unittest.TestCase):
    def test_reaches_a_sub_question_buried_below_the_cutoff(self):
        # The exact failure this exists to prevent: sub-question 2's only
        # document sits at rank 6, so rank-ordered top-3 would never see it.
        pool = _pool(
            ("d1", ["1"]), ("d2", ["1"]), ("d3", ["1"]),
            ("d4", ["1"]), ("d5", ["1"]), ("d6", ["2"]),
        )
        rank_ordered = pool[:3]
        self.assertEqual(_covered(rank_ordered, ["1", "2"]), {"1"})

        evidence = select_evidence(pool, ["1", "2"], evidence_k=3)
        self.assertEqual(_covered(evidence, ["1", "2"]), {"1", "2"})
        self.assertIn("d6", [c.docid for c in evidence])

    def test_every_sub_question_gets_one_before_any_gets_two(self):
        pool = _pool(
            ("d1", ["1"]), ("d2", ["1"]), ("d3", ["1"]),
            ("d4", ["2"]), ("d5", ["3"]),
        )
        evidence = select_evidence(pool, ["1", "2", "3"], evidence_k=3)
        self.assertEqual(_covered(evidence, ["1", "2", "3"]), {"1", "2", "3"})

    def test_rarest_sub_question_is_served_first(self):
        # sq2 has a single supporting document; sq1 has many. Serving the
        # plentiful one first can spend the budget before sq2 is reached.
        pool = _pool(
            ("d1", ["1"]), ("d2", ["1"]), ("d3", ["1"]), ("d4", ["2"])
        )
        evidence = select_evidence(pool, ["1", "2"], evidence_k=2)
        self.assertEqual(_covered(evidence, ["1", "2"]), {"1", "2"})

    def test_spare_slots_go_to_the_highest_ranked_remaining(self):
        pool = _pool(("d1", ["1"]), ("d2", ["1"]), ("d3", ["2"]))
        evidence = select_evidence(pool, ["1", "2"], evidence_k=3)
        self.assertEqual({c.docid for c in evidence}, {"d1", "d2", "d3"})

    def test_returned_in_pool_rank_order_not_selection_order(self):
        # The prompt lists evidence [0], [1], ... and asks for citations
        # "strongest support first", so the strongest must still lead.
        pool = _pool(("d1", ["1"]), ("d2", ["1"]), ("d3", ["2"]))
        evidence = select_evidence(pool, ["1", "2"], evidence_k=2)
        self.assertEqual([c.rank for c in evidence], sorted(c.rank for c in evidence))

    def test_documents_are_never_duplicated(self):
        pool = _pool(("d1", ["1", "2", "3"]), ("d2", ["1"]), ("d3", ["2"]))
        evidence = select_evidence(pool, ["1", "2", "3"], evidence_k=3)
        docids = [c.docid for c in evidence]
        self.assertEqual(len(docids), len(set(docids)))

    def test_sub_question_with_no_documents_is_skipped_gracefully(self):
        pool = _pool(("d1", ["1"]), ("d2", ["1"]))
        evidence = select_evidence(pool, ["1", "2", "3"], evidence_k=2)
        self.assertEqual(len(evidence), 2)


class TestSelectionEdgeCases(unittest.TestCase):
    def test_small_pool_is_returned_whole(self):
        pool = _pool(("d1", ["1"]), ("d2", ["2"]))
        self.assertEqual(len(select_evidence(pool, ["1", "2"], evidence_k=10)), 2)

    def test_empty_pool(self):
        self.assertEqual(select_evidence([], ["1"], evidence_k=5), [])

    def test_zero_k(self):
        self.assertEqual(select_evidence(_pool(("d1", ["1"])), ["1"], evidence_k=0), [])

    def test_no_provenance_falls_back_to_rank_order(self):
        pool = _pool(("d1", []), ("d2", []), ("d3", []))
        evidence = select_evidence(pool, ["1", "2"], evidence_k=2)
        self.assertEqual([c.docid for c in evidence], ["d1", "d2"])

    def test_no_sub_questions_falls_back_to_rank_order(self):
        pool = _pool(("d1", ["1"]), ("d2", ["1"]), ("d3", ["1"]))
        self.assertEqual(
            [c.docid for c in select_evidence(pool, [], evidence_k=2)], ["d1", "d2"]
        )


class TestEffectiveEvidenceK(unittest.TestCase):
    def test_is_a_floor_never_a_reduction(self):
        self.assertEqual(effective_evidence_k(10, 2, 1), 10)
        self.assertEqual(effective_evidence_k(10, 8, 1), 10)

    def test_widens_when_sub_questions_exceed_the_budget(self):
        self.assertEqual(effective_evidence_k(10, 12, 1), 12)
        self.assertEqual(effective_evidence_k(10, 8, 2), 16)

    def test_inert_at_default_on_the_real_data_ceiling(self):
        # The widest narrative in the committed decomposition has 8
        # sub-questions; at the default this must not change evidence_k.
        self.assertEqual(effective_evidence_k(10, 8, 1), 10)

    def test_zero_per_sub_question_is_treated_as_one(self):
        self.assertEqual(effective_evidence_k(5, 12, 0), 12)


if __name__ == "__main__":
    unittest.main()
