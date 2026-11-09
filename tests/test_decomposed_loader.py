import json
import tempfile
import unittest
from pathlib import Path

from trecrag.decomposed.loader import (
    DecompositionFormatError,
    attach,
    load_decomposed_topics,
)
from trecrag.decomposed.schema import AtomicNeed
from trecrag.schema import Topic

_REAL_STEP3 = Path("output_decomp/preprocessed_queries_step3.jsonl")


def _record(**overrides):
    record = {
        "query_id": "rag2026-0",
        "comp_query": "A composite narrative asking two things.",
        "query_metadata": "The asker is on a hospital council.",
        "sub_questions": [
            {"sub_question_id": "1", "sub_question_text": "What should we prioritise?"},
            {"sub_question_id": "2", "sub_question_text": "How do we measure it?"},
        ],
        "dependencies": [{"depends_on": "1", "dependent": "2"}],
        "ain": [
            {
                "sub_question_id": "1",
                "ain_array": [
                    {"ain_id": 1, "ain_text": "Evidence on pipeline partnerships."},
                    {"ain_id": 2, "ain_text": "Evidence on hiring practices."},
                ],
            },
            {
                "sub_question_id": "2",
                "ain_array": [{"ain_id": 1, "ain_text": "Evidence on outcome measures."}],
            },
        ],
        "expanded_queries": [
            {
                "sub_question_id": "1",
                "eq_array": [
                    _eq(1, "pipeline partnerships nursing", "pipeline partnerships nursing schools"),
                    _eq(2, "hiring practices nursing", "hiring promotion practices nursing"),
                ],
            },
            {
                "sub_question_id": "2",
                "eq_array": [_eq(1, "outcome measures nursing", "outcome metrics nursing")],
            },
        ],
    }
    record.update(overrides)
    return record


def _eq(ain_id, conditioned, expanded, prf_status="disabled", prf_query=None):
    return {
        "eq_id": ain_id,
        "ain_id": ain_id,
        "step3a_conditioned_query": conditioned,
        "step3b_expansion_terms": expanded.split(),
        "step3b_expanded_query": expanded,
        "step3c_prf_terms": [],
        # Mirrors the real files: when PRF is off this field still carries a
        # verbatim copy of the 3B query rather than being blank.
        "step3c_expanded_query": prf_query if prf_query is not None else expanded,
        "step3c_status": prf_status,
    }


def _write(records) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "step3.jsonl"
    with open(tmp, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return tmp


class TestLoaderJoin(unittest.TestCase):
    def test_joins_needs_to_reformulations(self):
        loaded = load_decomposed_topics(_write([_record()]))
        topic = loaded["rag2026-0"]

        self.assertEqual(len(topic.needs), 3)
        self.assertEqual(topic.composite_text, "A composite narrative asking two things.")
        self.assertEqual(topic.metadata, "The asker is on a hospital council.")
        self.assertEqual(topic.dependencies, [("1", "2")])

        # Joined on (sub_question_id, ain_id), not on position.
        first = topic.needs[0]
        self.assertEqual((first.sub_question_id, first.ain_id), ("1", 1))
        self.assertEqual(first.ain_text, "Evidence on pipeline partnerships.")
        self.assertEqual(first.expanded_query, "pipeline partnerships nursing schools")

    def test_join_survives_reordered_expansion_block(self):
        record = _record()
        record["expanded_queries"][0]["eq_array"].reverse()
        topic = load_decomposed_topics(_write([record]))["rag2026-0"]
        by_key = {(n.sub_question_id, n.ain_id): n for n in topic.needs}
        # ain_id 2 must still carry ain_id 2's text, not the first slot's.
        self.assertEqual(by_key[("1", 2)].ain_text, "Evidence on hiring practices.")
        self.assertEqual(by_key[("1", 2)].expanded_query, "hiring promotion practices nursing")

    def test_missing_reformulation_raises(self):
        record = _record()
        record["expanded_queries"][0]["eq_array"].pop()
        with self.assertRaises(DecompositionFormatError) as ctx:
            load_decomposed_topics(_write([record]))
        self.assertIn("misalignment", str(ctx.exception))

    def test_orphan_reformulation_raises(self):
        record = _record()
        record["ain"][0]["ain_array"].pop()
        with self.assertRaises(DecompositionFormatError):
            load_decomposed_topics(_write([record]))

    def test_duplicate_query_id_raises(self):
        with self.assertRaises(DecompositionFormatError):
            load_decomposed_topics(_write([_record(), _record()]))

    def test_empty_file_raises(self):
        with self.assertRaises(DecompositionFormatError):
            load_decomposed_topics(_write([]))


class TestPrfStatusGate(unittest.TestCase):
    """`step3c_expanded_query` is populated even when PRF never ran."""

    def test_disabled_status_is_treated_as_absent(self):
        topic = load_decomposed_topics(_write([_record()]))["rag2026-0"]
        self.assertIsNone(topic.needs[0].prf_query)

    def test_successful_status_is_honoured(self):
        record = _record()
        record["expanded_queries"][0]["eq_array"][0] = _eq(
            1, "conditioned", "expanded", prf_status="ok", prf_query="expanded plus prf terms"
        )
        topic = load_decomposed_topics(_write([record]))["rag2026-0"]
        self.assertEqual(topic.needs[0].prf_query, "expanded plus prf terms")


class TestQueryVariantFallback(unittest.TestCase):
    def test_falls_through_to_the_next_available_variant(self):
        need = AtomicNeed(
            ain_id=1,
            sub_question_id="1",
            ain_text="raw need",
            conditioned_query="conditioned",
            expanded_query="expanded",
            prf_query=None,
        )
        # 3C is unavailable, so asking for it must yield 3B rather than blank.
        self.assertEqual(need.query_text("step3c_expanded_query"), "expanded")
        self.assertEqual(need.query_text("step3b_expanded_query"), "expanded")
        self.assertEqual(need.query_text("step3a_conditioned_query"), "conditioned")
        self.assertEqual(need.query_text("ain_text"), "raw need")

    def test_falls_all_the_way_to_raw_text(self):
        need = AtomicNeed(ain_id=1, sub_question_id="1", ain_text="raw need")
        self.assertEqual(need.query_text("step3b_expanded_query"), "raw need")

    def test_unknown_variant_raises(self):
        need = AtomicNeed(ain_id=1, sub_question_id="1", ain_text="raw need")
        with self.assertRaises(ValueError):
            need.query_text("step9_nonexistent")

    def test_ain_topic_ids_are_unique_and_traceable(self):
        topic = load_decomposed_topics(_write([_record()]))["rag2026-0"]
        ids = [t.topic_id for t in topic.ain_topics()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "rag2026-0#sq1-ain1")
        self.assertTrue(all(i.startswith("rag2026-0#") for i in ids))


class TestAttach(unittest.TestCase):
    def test_missing_decomposition_raises_naming_the_topics(self):
        loaded = load_decomposed_topics(_write([_record()]))
        topics = [Topic("rag2026-0", "a"), Topic("14", "b"), Topic("31", "c")]
        with self.assertRaises(DecompositionFormatError) as ctx:
            attach(topics, loaded)
        message = str(ctx.exception)
        self.assertIn("14", message)
        self.assertIn("31", message)

    def test_returns_in_input_order(self):
        loaded = load_decomposed_topics(_write([_record(), _record(query_id="rag2026-1")]))
        topics = [Topic("rag2026-1", "b"), Topic("rag2026-0", "a")]
        self.assertEqual([t.topic_id for t in attach(topics, loaded)], ["rag2026-1", "rag2026-0"])


@unittest.skipUnless(_REAL_STEP3.is_file(), "committed step-3 output not present")
class TestAgainstRealDecomposition(unittest.TestCase):
    """The committed 119-narrative file must load and join cleanly."""

    def test_loads_and_joins(self):
        loaded = load_decomposed_topics(_REAL_STEP3)
        self.assertEqual(len(loaded), 119)
        self.assertEqual(sum(len(t.needs) for t in loaded.values()), 1270)

    def test_every_need_yields_a_non_empty_query(self):
        loaded = load_decomposed_topics(_REAL_STEP3)
        for topic in loaded.values():
            for ain_topic in topic.ain_topics("step3b_expanded_query"):
                self.assertTrue(ain_topic.text.strip(), ain_topic.topic_id)

    def test_prf_is_absent_everywhere(self):
        # Guards the config default: if this ever fails, PRF has been run and
        # `query_variant: step3c_expanded_query` becomes meaningful.
        loaded = load_decomposed_topics(_REAL_STEP3)
        self.assertTrue(
            all(need.prf_query is None for t in loaded.values() for need in t.needs)
        )


if __name__ == "__main__":
    unittest.main()
