"""Tests for the step-1 query decomposition parser and its final write gate.

The design deliberately makes the LLM's judgement call once and then extracts it
*deterministically*, so the parser is the component carrying that guarantee. Its
failure mode is silent: a lenient parse ships a wrong metadata/sub-question split
into every downstream retrieval, and nothing later can tell that happened.
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "decompose"))

from preprocess_step1_decompose import (  # noqa: E402
    DecompositionParseError,
    load_topics,
    parse_decomposition,
    validate_final_record,
)

GOOD = (
    "<metadata>I run a bakery.</metadata>\n"
    '<sub_question id="1">Should I raise prices?</sub_question>\n'
    '<sub_question id="2">How do I tell customers?</sub_question>'
)


def _parse(raw):
    return parse_decomposition(raw, "q1", "original text")


class TestParseAccepts(unittest.TestCase):
    def test_metadata_and_ordered_sub_questions(self):
        rec = _parse(GOOD)
        self.assertEqual(rec["query_metadata"], "I run a bakery.")
        self.assertEqual([s["sub_question_id"] for s in rec["sub_questions"]], ["1", "2"])
        self.assertEqual(rec["sub_questions"][0]["sub_question_text"], "Should I raise prices?")

    def test_empty_metadata_is_allowed_and_stays_a_string(self):
        rec = _parse('<metadata></metadata><sub_question id="1">What causes auroras?</sub_question>')
        self.assertEqual(rec["query_metadata"], "")
        self.assertEqual(len(rec["sub_questions"]), 1)

    def test_sub_questions_is_always_a_list_even_with_one_entry(self):
        rec = _parse('<metadata>a</metadata><sub_question id="1">q?</sub_question>')
        self.assertIsInstance(rec["sub_questions"], list)

    def test_markdown_fences_are_tolerated(self):
        """A fenced but otherwise correct answer is a formatting artefact, not a
        failed decomposition -- retrying it would waste a generation."""
        self.assertEqual(len(_parse(f"```xml\n{GOOD}\n```")["sub_questions"]), 2)

    def test_comp_query_is_preserved_verbatim(self):
        self.assertEqual(_parse(GOOD)["comp_query"], "original text")

    def test_handles_many_sub_questions(self):
        raw = "<metadata></metadata>" + "".join(
            f'<sub_question id="{i}">q{i}?</sub_question>' for i in range(1, 6)
        )
        self.assertEqual(len(_parse(raw)["sub_questions"]), 5)


class TestParseRejects(unittest.TestCase):
    def _assert_rejected(self, raw, rule):
        with self.assertRaises(DecompositionParseError) as ctx:
            _parse(raw)
        self.assertIn(rule, str(ctx.exception))

    def test_rule1_requires_exactly_one_metadata_block(self):
        self._assert_rejected(
            '<metadata>a</metadata><metadata>b</metadata><sub_question id="1">q</sub_question>',
            "rule1",
        )
        self._assert_rejected('<sub_question id="1">q</sub_question>', "rule1")

    def test_rule2_requires_at_least_one_sub_question(self):
        self._assert_rejected("<metadata>a</metadata>", "rule2")

    def test_rule2_rejects_an_empty_sub_question(self):
        self._assert_rejected('<metadata>a</metadata><sub_question id="1">   </sub_question>', "rule2")

    def test_rule3_rejects_ids_not_starting_at_one(self):
        self._assert_rejected('<metadata>a</metadata><sub_question id="2">q</sub_question>', "rule3")

    def test_rule3_rejects_gaps_and_disorder(self):
        self._assert_rejected(
            '<metadata>a</metadata><sub_question id="1">q</sub_question>'
            '<sub_question id="3">r</sub_question>',
            "rule3",
        )
        self._assert_rejected(
            '<metadata>a</metadata><sub_question id="2">q</sub_question>'
            '<sub_question id="1">r</sub_question>',
            "rule3",
        )

    def test_rule4_rejects_surrounding_prose(self):
        """'Here is the decomposition:' means the model ignored the format
        instruction; accepting it would normalise drift we cannot detect later."""
        self._assert_rejected(f"Here is the decomposition:\n{GOOD}", "rule4")
        self._assert_rejected(f"{GOOD}\nHope that helps!", "rule4")

    def test_rule5_rejects_a_truncated_tag_bleeding_into_content(self):
        self._assert_rejected(
            '<metadata>a</metadata><sub_question id="1">q <sub_ques</sub_question>', "rule5"
        )

    def test_empty_generation(self):
        with self.assertRaises(DecompositionParseError):
            _parse("")


class TestFinalGate(unittest.TestCase):
    VALID = {
        "query_id": "rag2026-0",
        "comp_query": "c",
        "query_metadata": "m",
        "sub_questions": [{"sub_question_id": "1", "sub_question_text": "q"}],
    }

    def test_accepts_a_valid_record(self):
        validate_final_record(copy.deepcopy(self.VALID))

    def _assert_rejected(self, mutate):
        record = copy.deepcopy(self.VALID)
        mutate(record)
        with self.assertRaises(DecompositionParseError):
            validate_final_record(record)

    def test_rejects_extra_or_missing_keys(self):
        self._assert_rejected(lambda r: r.update({"unexpected": 1}))
        self._assert_rejected(lambda r: r.pop("query_metadata"))

    def test_rejects_empty_or_non_list_sub_questions(self):
        self._assert_rejected(lambda r: r.update({"sub_questions": []}))
        self._assert_rejected(lambda r: r.update({"sub_questions": {"sub_question_id": "1"}}))

    def test_rejects_non_string_or_misnumbered_ids(self):
        self._assert_rejected(lambda r: r["sub_questions"][0].update({"sub_question_id": 1}))
        self._assert_rejected(lambda r: r["sub_questions"][0].update({"sub_question_id": "2"}))

    def test_rejects_blank_text_fields(self):
        self._assert_rejected(lambda r: r["sub_questions"][0].update({"sub_question_text": "  "}))
        self._assert_rejected(lambda r: r.update({"comp_query": ""}))

    def test_gate_is_independent_of_the_parser(self):
        """It must reject a bad record even though parse_decomposition can no
        longer produce one -- that independence is the point of having it."""
        with self.assertRaises(DecompositionParseError):
            validate_final_record({"query_id": "a"})


class TestLoadTopics(unittest.TestCase):
    def _write(self, tmp, text):
        path = Path(tmp) / "t.tsv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_id_and_text_and_skips_blank_lines(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            topics = load_topics(self._write(tmp, "t1\tfirst\n\nt2\tsecond\n"))
        self.assertEqual([t.query_id for t in topics], ["t1", "t2"])

    def test_embedded_tabs_do_not_truncate_the_query(self):
        """Splitting on every tab would silently drop the rest of a narrative."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            topics = load_topics(self._write(tmp, "t1\tpart one\tpart two\n"))
        self.assertEqual(topics[0].query_text, "part one\tpart two")

    def test_alphanumeric_ids_are_kept_as_strings(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            topics = load_topics(self._write(tmp, "1rag2026-0\tq\n"))
        self.assertEqual(topics[0].query_id, "1rag2026-0")

    def test_real_topics_file_parses(self):
        path = Path(__file__).resolve().parent.parent / "trec_rag_2026_test_query_set.tsv"
        if not path.is_file():
            self.skipTest("official topics file not present")
        topics = load_topics(path)
        self.assertEqual(len(topics), 119)
        self.assertEqual(topics[0].query_id, "rag2026-0")


if __name__ == "__main__":
    unittest.main()
