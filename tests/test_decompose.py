"""Tests for the step-1 query decomposition parser and its final write gate.

The design deliberately makes the LLM's judgement call once and then extracts it
*deterministically*, so the parser is the component carrying that guarantee. Its
failure mode is silent: a lenient parse ships a wrong metadata/sub-question split
into every downstream retrieval, and nothing later can tell that happened.

The same argument covers step 1b (dependency detection) at the bottom of this file,
with one extra edge: there, a dropped tag does not look like a malformed record, it
looks like the opposite verdict -- "no dependency" is itself a valid answer.
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "decompose"))

import preprocess_step1_decompose as step1  # noqa: E402
from preprocess_step1_decompose import (  # noqa: E402
    DecompositionParseError,
    build_dependency_prompts,
    detect_dependencies,
    load_topics,
    parse_decomposition,
    parse_dependencies,
    validate_dependencies,
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


# --------------------------------------------------------------------------------
# step 1b: dependency detection
# --------------------------------------------------------------------------------

DEP = '<dependencies><dependency from="1" to="2"/></dependencies>'
NONE = "<dependencies></dependencies>"


class TestParseDependenciesAccepts(unittest.TestCase):
    def test_single_dependency_maps_from_to_onto_the_schema(self):
        self.assertEqual(
            parse_dependencies(DEP, ["1", "2"]), [{"depends_on": "1", "dependent": "2"}]
        )

    def test_empty_wrapper_is_a_verdict_not_a_failure(self):
        """'No dependency' is the expected answer for most pairs, so the empty
        wrapper has to parse cleanly rather than trip the retry machinery."""
        self.assertEqual(parse_dependencies(NONE, ["1", "2"]), [])

    def test_self_closed_wrapper_is_also_an_empty_verdict(self):
        self.assertEqual(parse_dependencies("<dependencies/>", ["1", "2"]), [])

    def test_whitespace_and_fences_are_tolerated(self):
        self.assertEqual(parse_dependencies(f"```xml\n{NONE}\n```", ["1", "2"]), [])
        self.assertEqual(len(parse_dependencies(f"\n {DEP} \n", ["1", "2"])), 1)

    def test_attribute_order_and_spacing_do_not_matter(self):
        raw = "<dependencies>\n<dependency  to='2'  from='1' />\n</dependencies>"
        self.assertEqual(
            parse_dependencies(raw, ["1", "2"]), [{"depends_on": "1", "dependent": "2"}]
        )

    def test_several_pairs_in_one_batched_verdict(self):
        raw = (
            "<dependencies>"
            '<dependency from="1" to="2"/><dependency from="1" to="3"/>'
            "</dependencies>"
        )
        self.assertEqual(len(parse_dependencies(raw, ["1", "2", "3"])), 2)

    def test_pairwise_mode_accepts_either_direction_for_the_judged_pair(self):
        raw = '<dependencies><dependency from="2" to="1"/></dependencies>'
        self.assertEqual(
            parse_dependencies(raw, ["1", "2"], expected_pair=("1", "2")),
            [{"depends_on": "2", "dependent": "1"}],
        )


class TestParseDependenciesRejects(unittest.TestCase):
    def _assert_rejected(self, raw, rule, valid_ids=("1", "2"), expected_pair=None):
        with self.assertRaises(DecompositionParseError) as ctx:
            parse_dependencies(raw, list(valid_ids), expected_pair)
        self.assertIn(rule, str(ctx.exception))

    def test_blank_output_is_not_a_verdict(self):
        """An empty generation is indistinguishable from a crashed one, which is
        exactly why the prompts demand the wrapper even when there is nothing in it."""
        for raw in ("", "   ", None):
            with self.assertRaises(DecompositionParseError):
                parse_dependencies(raw, ["1", "2"])

    def test_rule1_requires_exactly_one_wrapper(self):
        self._assert_rejected('<dependency from="1" to="2"/>', "dep-rule1")
        self._assert_rejected(NONE + NONE, "dep-rule1")

    def test_rule1_rejects_a_tag_missing_from_or_to(self):
        self._assert_rejected('<dependencies><dependency to="2"/></dependencies>', "dep-rule1")
        self._assert_rejected(
            '<dependencies><dependency src="1" to="2"/></dependencies>', "dep-rule1"
        )

    def test_rule2_rejects_a_hallucinated_id(self):
        self._assert_rejected(
            '<dependencies><dependency from="1" to="9"/></dependencies>', "dep-rule2"
        )

    def test_rule3_rejects_a_self_dependency(self):
        self._assert_rejected(
            '<dependencies><dependency from="1" to="1"/></dependencies>', "dep-rule3"
        )

    def test_rule4_rejects_a_repeated_pair(self):
        self._assert_rejected(
            "<dependencies>"
            '<dependency from="1" to="2"/><dependency from="1" to="2"/>'
            "</dependencies>",
            "dep-rule4",
        )

    def test_rule5_rejects_prose_around_the_wrapper(self):
        self._assert_rejected(f"Here is my analysis:\n{NONE}", "dep-rule5")
        self._assert_rejected(f"{DEP}\nLet me know if that helps!", "dep-rule5")

    def test_rule5_rejects_prose_inside_the_wrapper(self):
        """This is the dangerous one: without the inside-the-wrapper guard, a
        model that answers in words parses as the empty list, i.e. as 'no
        dependency' -- a wrong verdict rather than a visible failure."""
        self._assert_rejected("<dependencies>1 must be answered first</dependencies>", "dep-rule5")

    def test_rule5_rejects_a_non_self_closing_tag(self):
        self._assert_rejected(
            '<dependencies><dependency from="1" to="2"></dependency></dependencies>', "dep-rule5"
        )

    def test_rule6_rejects_a_pair_asserted_in_both_directions(self):
        """Picking a winner would mean inventing a verdict the model did not give."""
        self._assert_rejected(
            "<dependencies>"
            '<dependency from="1" to="2"/><dependency from="2" to="1"/>'
            "</dependencies>",
            "dep-rule6",
        )

    def test_rule7_rejects_an_id_outside_the_judged_pair(self):
        self._assert_rejected(
            '<dependencies><dependency from="1" to="3"/></dependencies>',
            "dep-rule7",
            valid_ids=("1", "2", "3"),
            expected_pair=("1", "2"),
        )

    def test_rule7_rejects_more_than_one_tag_for_one_pair(self):
        self._assert_rejected(
            "<dependencies>"
            '<dependency from="1" to="2"/><dependency from="2" to="3"/>'
            "</dependencies>",
            "dep-rule7",
            valid_ids=("1", "2", "3"),
            expected_pair=("1", "2"),
        )


class TestValidateDependencies(unittest.TestCase):
    SUBQS = [
        {"sub_question_id": "1", "sub_question_text": "a?"},
        {"sub_question_id": "2", "sub_question_text": "b?"},
    ]

    def test_accepts_an_empty_list_and_a_valid_edge(self):
        validate_dependencies([], self.SUBQS)
        validate_dependencies([{"depends_on": "1", "dependent": "2"}], self.SUBQS)

    def _assert_rejected(self, dependencies):
        with self.assertRaises(DecompositionParseError):
            validate_dependencies(dependencies, self.SUBQS)

    def test_rejects_wrong_shape(self):
        self._assert_rejected({"depends_on": "1", "dependent": "2"})
        self._assert_rejected([{"depends_on": "1"}])
        self._assert_rejected([{"depends_on": "1", "dependent": "2", "why": "x"}])

    def test_rejects_non_string_ids(self):
        self._assert_rejected([{"depends_on": 1, "dependent": "2"}])

    def test_rejects_unanchored_self_duplicate_and_contradictory_edges(self):
        self._assert_rejected([{"depends_on": "1", "dependent": "3"}])
        self._assert_rejected([{"depends_on": "1", "dependent": "1"}])
        self._assert_rejected([{"depends_on": "1", "dependent": "2"}] * 2)
        self._assert_rejected(
            [{"depends_on": "1", "dependent": "2"}, {"depends_on": "2", "dependent": "1"}]
        )

    def test_gate_is_independent_of_the_parser(self):
        """Same reason the decomposition gate exists: a future parser bug must not
        be able to put an unanchored dependency in the output file."""
        self._assert_rejected([{"depends_on": "1", "dependent": "2"}, "not a dict"])


def _record(query_id, n_sub_questions, metadata="ctx"):
    return {
        "query_id": query_id,
        "comp_query": f"comp {query_id}",
        "query_metadata": metadata,
        "sub_questions": [
            {"sub_question_id": str(i), "sub_question_text": f"{query_id}-q{i}?"}
            for i in range(1, n_sub_questions + 1)
        ],
    }


class TestBuildDependencyPrompts(unittest.TestCase):
    def test_a_lone_sub_question_costs_no_generation(self):
        """Nothing to judge, so `dependencies` is [] without an LLM call."""
        self.assertEqual(build_dependency_prompts([_record("a", 1)], False), [])
        self.assertEqual(build_dependency_prompts([_record("a", 1)], True), [])

    def test_batched_mode_is_one_call_per_query(self):
        jobs = build_dependency_prompts([_record("a", 2), _record("b", 3)], False)
        self.assertEqual([j.query_id for j in jobs], ["a", "b"])
        self.assertTrue(all(j.expected_pair is None for j in jobs))

    def test_pairwise_mode_is_one_call_per_pair(self):
        jobs = build_dependency_prompts([_record("a", 2), _record("b", 3)], True)
        self.assertEqual([j.query_id for j in jobs], ["a", "b", "b", "b"])
        self.assertEqual(
            [j.expected_pair for j in jobs if j.query_id == "b"],
            [("1", "2"), ("1", "3"), ("2", "3")],
        )

    def test_pairwise_valid_ids_stay_query_wide(self):
        """rule2 (id exists in this query) and rule7 (id was in *this* pair) are
        separate checks, so valid_ids must not be narrowed to the pair."""
        job = build_dependency_prompts([_record("b", 3)], True)[0]
        self.assertEqual(job.valid_ids, ["1", "2", "3"])

    def test_both_modes_carry_the_metadata_and_the_sub_question_text(self):
        for pairwise in (False, True):
            job = build_dependency_prompts([_record("a", 2, metadata="budget is tight")], pairwise)[0]
            for message in (job.primary[-1]["content"], job.fallback[-1]["content"]):
                self.assertIn("budget is tight", message)
                self.assertIn("a-q1?", message)
                self.assertIn("a-q2?", message)

    def test_empty_metadata_is_labelled_rather_than_left_dangling(self):
        job = build_dependency_prompts([_record("a", 2, metadata="")], False)[0]
        self.assertIn("(none given)", job.primary[-1]["content"])


class TestDetectDependencies(unittest.TestCase):
    """End-to-end over the stage's own retry/merge/flag logic, with generation
    replaced by a scripted double -- no GPU, no vLLM import."""

    def setUp(self):
        self._real_run_inference = step1.run_inference
        self.addCleanup(setattr, step1, "run_inference", self._real_run_inference)

    def _scripted(self, replies_by_query):
        """replies_by_query maps query_id -> [first attempt, fallback attempt]."""
        calls = {query_id: 0 for query_id in replies_by_query}

        def fake(llm, tokenizer, conversations, batch_size, max_tokens=None):
            outputs = []
            for conversation in conversations:
                body = conversation[-1]["content"]
                query_id = next(q for q in replies_by_query if f"{q}-q1?" in body)
                replies = replies_by_query[query_id]
                index = min(calls[query_id], len(replies) - 1)
                calls[query_id] += 1
                outputs.append(replies[index])
            return outputs

        step1.run_inference = fake

    def test_attaches_dependencies_recovers_via_fallback_and_flags_the_rest(self):
        records = [_record("a", 1), _record("b", 2), _record("c", 2), _record("d", 2)]
        self._scripted(
            {
                "b": [DEP],
                "c": ["Sure! There is no dependency here.", NONE],
                "d": ["nope", "still nope"],
            }
        )
        good, flagged, stats = detect_dependencies(None, None, records, 0, False)

        self.assertEqual([r["query_id"] for r in good], ["a", "b", "c"])
        self.assertEqual(good[0]["dependencies"], [])   # single sub-question
        self.assertEqual(good[1]["dependencies"], [{"depends_on": "1", "dependent": "2"}])
        self.assertEqual(good[2]["dependencies"], [])   # recovered by the fallback
        self.assertEqual(
            stats,
            {"mode": "batched", "calls": 3, "fallback_recovered": 1, "flagged": 1,
             "with_dependency": 1},
        )

    def test_a_dependency_failure_keeps_the_decomposition_and_names_its_stage(self):
        """The record decomposed fine; only the ordering judgement failed. Losing
        that work would be worse than flagging it, and it must not be confusable
        with a decomposition failure when the log is scanned later."""
        self._scripted({"d": ["nope", "still nope"]})
        good, flagged, _ = detect_dependencies(None, None, [_record("d", 2)], 0, False)

        self.assertEqual(good, [])
        record = flagged[0]
        self.assertEqual(record["failure_stage"], "dependency-detection")
        self.assertEqual(len(record["sub_questions"]), 2)
        self.assertEqual(record["first_attempt_raw_output"], "nope")
        self.assertEqual(record["fallback_attempt_raw_output"], "still nope")
        self.assertIn("dep-rule1", record["failure_reason"])
        self.assertNotIn("dependencies", record)

    def test_flagged_records_never_reach_the_main_output(self):
        """The invariant the whole stage exists to keep: every written record has a
        validated `dependencies` field, so nothing carries a guessed one."""
        self._scripted({"b": [DEP], "d": ["nope", "nope"]})
        good, flagged, _ = detect_dependencies(
            None, None, [_record("b", 2), _record("d", 2)], 0, False
        )
        self.assertTrue(all("dependencies" in r for r in good))
        self.assertEqual({r["query_id"] for r in good} & {r["query_id"] for r in flagged}, set())

    def test_pairwise_mode_merges_one_verdict_per_pair(self):
        record = _record("e", 3)
        pair_replies = {("1", "2"): DEP}

        def fake(llm, tokenizer, conversations, batch_size, max_tokens=None):
            outputs = []
            for conversation in conversations:
                body = conversation[-1]["content"]
                pair = tuple(i for i in ("1", "2", "3") if f"e-q{i}?" in body)
                outputs.append(pair_replies.get(pair, NONE))
            return outputs

        step1.run_inference = fake
        good, flagged, stats = detect_dependencies(None, None, [record], 0, True)

        self.assertEqual(flagged, [])
        self.assertEqual(stats["calls"], 3)   # 3 sub-questions -> 3 pairs
        self.assertEqual(good[0]["dependencies"], [{"depends_on": "1", "dependent": "2"}])

    def test_one_failed_pair_flags_the_whole_query(self):
        """A partial verdict would silently under-report dependencies, which is the
        same class of error as a partial decomposition."""
        record = _record("e", 3)

        def fake(llm, tokenizer, conversations, batch_size, max_tokens=None):
            return [
                NONE if "e-q3?" not in conversation[-1]["content"] else "I cannot tell."
                for conversation in conversations
            ]

        step1.run_inference = fake
        good, flagged, stats = detect_dependencies(None, None, [record], 0, True)

        self.assertEqual(good, [])
        self.assertEqual(stats["flagged"], 1)
        self.assertIn("pair 1->3", flagged[0]["failure_reason"])

    def test_no_records_is_not_an_error(self):
        step1.run_inference = lambda *a, **kw: []
        self.assertEqual(detect_dependencies(None, None, [], 0, False)[:2], ([], []))


if __name__ == "__main__":
    unittest.main()
