"""Tests for the step-2 AIN extraction parser, its write gate, and the stage around it.

Same argument as step 1: the LLM makes the judgement once and a deterministic parser
extracts it, so the parser carries the guarantee. The failure mode here is quieter
than a malformed record -- a half-parsed <ains> block is a *plausible shorter list*,
and the topics it silently dropped are exactly the ones nothing will ever retrieve
for. Hence all-or-nothing parsing, and a gate that re-derives the invariants.
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "decompose"))

import preprocess_step2_decompose as step2  # noqa: E402
from preprocess_step2_decompose import (  # noqa: E402
    AINParseError,
    build_ain_prompts,
    extract_ains,
    load_step1_records,
    parse_ains,
    validate_final_record,
)

ONE = '<ains><ain id="1">Evidence regarding aurora physics.</ain></ains>'
TWO = (
    "<ains>"
    '<ain id="1">Evidence regarding equitable hiring practices in nursing.</ain>'
    '<ain id="2">Evidence regarding workplace racism reporting systems.</ain>'
    "</ains>"
)


class TestParseAinsAccepts(unittest.TestCase):
    def test_single_ain_maps_onto_the_schema(self):
        self.assertEqual(
            parse_ains(ONE), [{"ain_id": 1, "ain_text": "Evidence regarding aurora physics."}]
        )

    def test_ain_ids_are_ints_starting_at_one(self):
        """sub_question_id is a string and ain_id is an int -- that asymmetry is the
        agreed downstream schema, so it is pinned down here."""
        parsed = parse_ains(TWO)
        self.assertEqual([a["ain_id"] for a in parsed], [1, 2])
        self.assertTrue(all(isinstance(a["ain_id"], int) for a in parsed))

    def test_newlines_and_fences_are_tolerated(self):
        raw = '```xml\n<ains>\n<ain id="1">Evidence regarding A.</ain>\n</ains>\n```'
        self.assertEqual(len(parse_ains(raw)), 1)

    def test_single_quoted_ids_and_loose_spacing(self):
        self.assertEqual(len(parse_ains("<ains><ain id = '1' >Evidence regarding A.</ain></ains>")), 1)

    def test_a_wide_but_legal_split(self):
        """A sub-question crossing several aspects with several options genuinely
        runs to ~24; the cap is a runaway guard and must sit above that."""
        raw = "<ains>" + "".join(
            f'<ain id="{i}">Evidence regarding topic {i}.</ain>' for i in range(1, 31)
        ) + "</ains>"
        self.assertEqual(len(parse_ains(raw)), 30)


class TestParseAinsRejects(unittest.TestCase):
    def _assert_rejected(self, raw, rule):
        with self.assertRaises(AINParseError) as ctx:
            parse_ains(raw)
        self.assertIn(rule, str(ctx.exception))

    def test_empty_generation(self):
        for raw in ("", "   ", None):
            with self.assertRaises(AINParseError):
                parse_ains(raw)

    def test_rule1_requires_exactly_one_wrapper(self):
        self._assert_rejected('<ain id="1">Evidence regarding A.</ain>', "rule1")
        self._assert_rejected(ONE + ONE, "rule1")

    def test_rule2_requires_at_least_one_ain(self):
        """Unlike step 1b's dependencies, an empty answer is never a verdict here:
        every sub-question is about at least one thing."""
        self._assert_rejected("<ains></ains>", "rule2")

    def test_rule2_rejects_an_empty_ain(self):
        self._assert_rejected('<ains><ain id="1">   </ain></ains>', "rule2")

    def test_rule3_rejects_ids_not_starting_at_one_or_with_gaps(self):
        self._assert_rejected('<ains><ain id="2">Evidence regarding A.</ain></ains>', "rule3")
        self._assert_rejected(
            '<ains><ain id="1">A.</ain><ain id="3">B.</ain></ains>', "rule3"
        )
        self._assert_rejected(
            '<ains><ain id="2">A.</ain><ain id="1">B.</ain></ains>', "rule3"
        )

    def test_rule4_rejects_prose_around_the_wrapper(self):
        self._assert_rejected(f"Here are the atomic needs:\n{ONE}", "rule4")
        self._assert_rejected(f"{ONE}\nHope that helps!", "rule4")

    def test_rule4_rejects_prose_between_the_tags(self):
        """A numbered prose list between two tags would otherwise parse as a valid
        shorter answer -- the silent-truncation case this stage cannot afford."""
        self._assert_rejected(
            '<ains><ain id="1">A.</ain> and also promotion practices </ains>', "rule4"
        )

    def test_rule5_rejects_a_truncated_tag_bleeding_into_content(self):
        self._assert_rejected('<ains><ain id="1">Evidence regarding <ain</ain></ains>', "rule5")

    def test_rule6_rejects_duplicate_topics_case_insensitively(self):
        """Two identical AINs are two identical retrievals: the model padded the list
        instead of splitting it."""
        self._assert_rejected(
            "<ains>"
            '<ain id="1">Evidence regarding hiring.</ain>'
            '<ain id="2">EVIDENCE REGARDING HIRING.</ain>'
            "</ains>",
            "rule6",
        )

    def test_rule7_rejects_a_degenerate_over_split(self):
        raw = "<ains>" + "".join(
            f'<ain id="{i}">Evidence regarding topic {i}.</ain>' for i in range(1, 32)
        ) + "</ains>"
        self._assert_rejected(raw, "rule7")


def _record(query_id="q1", n_sub_questions=2, **extra):
    record = {
        "query_id": query_id,
        "comp_query": f"comp {query_id}",
        "query_metadata": "ctx",
        "sub_questions": [
            {"sub_question_id": str(i), "sub_question_text": f"{query_id}-q{i}?"}
            for i in range(1, n_sub_questions + 1)
        ],
        "dependencies": [],
    }
    record.update(extra)
    return record


def _with_ain(record):
    record = copy.deepcopy(record)
    record["ain"] = [
        {
            "sub_question_id": sq["sub_question_id"],
            "ain_array": [{"ain_id": 1, "ain_text": f"Evidence regarding {sq['sub_question_id']}."}],
        }
        for sq in record["sub_questions"]
    ]
    return record


class TestFinalGate(unittest.TestCase):
    def test_accepts_a_valid_record(self):
        validate_final_record(_with_ain(_record()))

    def test_unknown_step1_fields_ride_through(self):
        """`dependencies` -- and whatever step 1 grows next -- must not be rejected
        just because this stage does not know about it."""
        validate_final_record(_with_ain(_record(some_future_field={"a": 1})))

    def _assert_rejected(self, mutate):
        record = _with_ain(_record())
        mutate(record)
        with self.assertRaises(AINParseError):
            validate_final_record(record)

    def test_rejects_missing_required_keys(self):
        self._assert_rejected(lambda r: r.pop("ain"))
        self._assert_rejected(lambda r: r.pop("query_metadata"))

    def test_rejects_ain_not_aligned_with_sub_questions(self):
        """The count, the ids, and the order all have to line up: an `ain` block that
        silently belongs to a different sub-question is unrecoverable downstream."""
        self._assert_rejected(lambda r: r["ain"].pop())
        self._assert_rejected(lambda r: r["ain"][0].update({"sub_question_id": "2"}))
        self._assert_rejected(lambda r: r["ain"].reverse())

    def test_rejects_empty_or_malformed_ain_array(self):
        self._assert_rejected(lambda r: r["ain"][0].update({"ain_array": []}))
        self._assert_rejected(lambda r: r["ain"][0]["ain_array"].append({"ain_id": 2}))
        self._assert_rejected(
            lambda r: r["ain"][0]["ain_array"][0].update({"ain_text": "  "})
        )

    def test_rejects_non_int_or_misnumbered_ain_ids(self):
        self._assert_rejected(lambda r: r["ain"][0]["ain_array"][0].update({"ain_id": "1"}))
        self._assert_rejected(lambda r: r["ain"][0]["ain_array"][0].update({"ain_id": True}))
        self._assert_rejected(lambda r: r["ain"][0]["ain_array"][0].update({"ain_id": 2}))

    def test_rejects_duplicate_ain_text_within_one_sub_question(self):
        self._assert_rejected(
            lambda r: r["ain"][0]["ain_array"].append({"ain_id": 2, "ain_text": "Evidence regarding 1."})
        )

    def test_gate_is_independent_of_the_parser(self):
        with self.assertRaises(AINParseError):
            validate_final_record({"query_id": "a"})


class TestLoadStep1Records(unittest.TestCase):
    def _write(self, tmp, lines):
        path = Path(tmp) / "step1.jsonl"
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def test_reads_records_and_keeps_unknown_fields(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [json.dumps(_record()) + "\n"])
            records = load_step1_records(path)
        self.assertEqual(records[0]["dependencies"], [])

    def test_skips_bad_lines_without_taking_the_run_down(self):
        """A malformed line is an upstream bug, but it must be visible as a smaller
        record count rather than a crash halfway through a GPU run."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [
                "not json\n",
                "\n",
                json.dumps({"query_id": "x", "comp_query": "c"}) + "\n",   # no sub_questions
                json.dumps(_record("keep")) + "\n",
                json.dumps(_record("keep")) + "\n",                        # duplicate id
            ])
            with self.assertLogs("preprocess_step2", level="WARNING"):
                records = load_step1_records(path)
        self.assertEqual([r["query_id"] for r in records], ["keep"])


class TestBuildAinPrompts(unittest.TestCase):
    def test_one_job_per_sub_question(self):
        jobs = build_ain_prompts([_record("a", 2), _record("b", 3)])
        self.assertEqual(
            [(j.query_id, j.sub_question_id) for j in jobs],
            [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("b", "3")],
        )

    def test_prompts_carry_the_sub_question_and_its_context(self):
        job = build_ain_prompts([_record("a", 2, query_metadata="budget is tight")])[1]
        for message in (job.primary[-1]["content"], job.fallback[-1]["content"]):
            self.assertIn("budget is tight", message)
            self.assertIn("a-q2?", message)
            self.assertNotIn("a-q1?", message)   # one sub-question per call

    def test_empty_metadata_is_labelled_rather_than_left_dangling(self):
        job = build_ain_prompts([_record("a", 1, query_metadata="")])[0]
        self.assertIn("(none given)", job.primary[-1]["content"])


class TestExtractAins(unittest.TestCase):
    """End-to-end over the stage's retry/assembly/flag logic, with generation replaced
    by a scripted double -- no GPU, no vLLM import."""

    def setUp(self):
        self._real = step2.run_inference
        self.addCleanup(setattr, step2, "run_inference", self._real)

    def _scripted(self, replies_by_sub_question):
        """Maps "<query_id>-q<n>?" -> [first attempt, fallback attempt]."""
        calls = {key: 0 for key in replies_by_sub_question}

        def fake(llm, tokenizer, conversations, batch_size):
            outputs = []
            for conversation in conversations:
                body = conversation[-1]["content"]
                key = next(k for k in replies_by_sub_question if k in body)
                replies = replies_by_sub_question[key]
                outputs.append(replies[min(calls[key], len(replies) - 1)])
                calls[key] += 1
            return outputs

        step2.run_inference = fake

    def test_attaches_one_ain_entry_per_sub_question_in_order(self):
        self._scripted({"a-q1?": [TWO], "a-q2?": [ONE]})
        good, flagged, stats = extract_ains(None, None, [_record("a", 2)], 0)

        self.assertEqual(flagged, [])
        record = good[0]
        self.assertEqual([e["sub_question_id"] for e in record["ain"]], ["1", "2"])
        self.assertEqual(len(record["ain"][0]["ain_array"]), 2)
        self.assertEqual(len(record["ain"][1]["ain_array"]), 1)
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["ains"], 3)

    def test_step1_fields_are_carried_through_untouched_with_ain_last(self):
        self._scripted({"a-q1?": [ONE]})
        source = _record("a", 1, dependencies=[{"depends_on": "1", "dependent": "2"}])
        good, _, _ = extract_ains(None, None, [copy.deepcopy(source)], 0)

        self.assertEqual(list(good[0])[-1], "ain")
        for key, value in source.items():
            self.assertEqual(good[0][key], value)

    def test_a_failed_sub_question_is_recovered_by_the_fallback(self):
        self._scripted({"a-q1?": ["Sure! Here are the topics: 1. hiring", ONE]})
        good, flagged, stats = extract_ains(None, None, [_record("a", 1)], 0)

        self.assertEqual(flagged, [])
        self.assertEqual((stats["first_attempt"], stats["recovered"]), (0, 1))
        self.assertEqual(len(good[0]["ain"][0]["ain_array"]), 1)

    def test_one_unresolved_sub_question_flags_the_whole_record(self):
        """A partial `ain` block would look complete. Everything the record already
        has is kept, and the reason names which sub-question failed."""
        self._scripted({"a-q1?": [ONE], "a-q2?": ["nope", "still nope"]})
        good, flagged, stats = extract_ains(None, None, [_record("a", 2)], 0)

        self.assertEqual(good, [])
        record = flagged[0]
        self.assertEqual(record["failure_stage"], "parser")
        self.assertIn("sub-question 2", record["failure_reason"])
        self.assertIn("rule1", record["failure_reason"])
        self.assertNotIn("ain", record)
        self.assertEqual(len(record["sub_questions"]), 2)      # step-1 work preserved
        self.assertIn("still nope", record["fallback_attempt_raw_output"])

    def test_records_are_written_or_flagged_but_never_both(self):
        self._scripted({"a-q1?": [ONE], "b-q1?": ["nope", "nope"]})
        good, flagged, _ = extract_ains(None, None, [_record("a", 1), _record("b", 1)], 0)
        self.assertEqual({r["query_id"] for r in good}, {"a"})
        self.assertEqual({r["query_id"] for r in flagged}, {"b"})

    def test_no_records_is_not_an_error(self):
        step2.run_inference = lambda *a, **kw: []
        good, flagged, stats = extract_ains(None, None, [], 0)
        self.assertEqual((good, flagged, stats["calls"]), ([], [], 0))


if __name__ == "__main__":
    unittest.main()
