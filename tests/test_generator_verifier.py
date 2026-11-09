import json
import unittest

from tests.test_generator_prompt import ScriptedLLM
from trecrag.generation.generator import ExtractiveGenerator, LLMGenerator
from trecrag.generation.prompts import render_verification_prompt
from trecrag.generation.verifier import (
    AnswerVerifier,
    CompositeVerifier,
    LexicalGroundingVerifier,
    LLMVerifier,
    VerificationResult,
    VerifierUnavailableError,
)
from trecrag.schema import AnswerSentence, Candidate, RagAnswer, Topic

_TOPIC = Topic("rag2026-0", "What are the tradeoffs of congestion pricing?")


def _evidence(*texts, features=None):
    return [
        Candidate(
            docid=f"d{i}", score=float(10 - i), rank=i + 1, text=text,
            features=dict((features or {}).get(i, {})),
        )
        for i, text in enumerate(texts)
    ]


def _answer(*pairs, refs=("d0", "d1")):
    return RagAnswer(
        references=list(refs),
        sentences=[AnswerSentence(text=t, citations=list(c)) for t, c in pairs],
    )


def _answer_json(*pairs):
    return json.dumps([{"text": t, "citations": list(c)} for t, c in pairs])


class TestHallucinationDetection(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence(
            "Congestion pricing generates dedicated revenue for transit improvements.",
            "Public reporting holds transit agencies accountable for spending.",
        )
        self.verifier = LexicalGroundingVerifier(check_omission=False)

    def test_grounded_answer_passes(self):
        answer = _answer(
            ("Congestion pricing generates dedicated revenue for transit.", [0]),
            ("Public reporting holds agencies accountable for spending.", [1]),
        )
        self.assertTrue(self.verifier.verify(_TOPIC, self.evidence, answer).ok)

    def test_sentence_unrelated_to_its_citation_is_caught(self):
        answer = _answer(
            ("Banana bread requires flour, sugar and ripe bananas.", [0]),
            ("Jazz music originated in New Orleans nightclubs.", [1]),
        )
        result = self.verifier.verify(_TOPIC, self.evidence, answer)
        self.assertFalse(result.ok)
        self.assertIn("do not support", result.issues[0])

    def test_uncited_sentence_counts_as_ungrounded(self):
        answer = _answer(
            ("Congestion pricing generates dedicated revenue.", [0]),
            ("An entirely uncited assertion about tolling.", []),
            ("Another uncited assertion about policy.", []),
        )
        self.assertFalse(self.verifier.verify(_TOPIC, self.evidence, answer).ok)

    def test_paraphrase_is_tolerated(self):
        # The threshold is a floor; rejecting paraphrase would cost a
        # regeneration and buy nothing.
        answer = _answer(
            ("Revenue from congestion pricing is dedicated to transit improvements.", [0]),
        )
        self.assertTrue(self.verifier.verify(_TOPIC, self.evidence, answer).ok)

    def test_a_single_bad_sentence_among_many_does_not_trip_it(self):
        answer = _answer(
            ("Congestion pricing generates dedicated revenue for transit.", [0]),
            ("Public reporting holds transit agencies accountable.", [1]),
            ("Congestion pricing revenue funds transit improvements.", [0]),
            ("Unrelated banana bread flour sugar.", [1]),
        )
        self.assertTrue(self.verifier.verify(_TOPIC, self.evidence, answer).ok)

    def test_can_be_switched_off_independently(self):
        answer = _answer(("Banana bread flour sugar ripe bananas.", [0]))
        off = LexicalGroundingVerifier(check_hallucination=False, check_omission=False)
        self.assertTrue(off.verify(_TOPIC, self.evidence, answer).ok)


class TestOmissionDetection(unittest.TestCase):
    def test_unused_evidence_is_flagged(self):
        evidence = _evidence("alpha content here", "beta content here", "gamma content here", "delta content here")
        answer = _answer(("alpha content here", [0]))
        result = LexicalGroundingVerifier(check_hallucination=False).verify(
            _TOPIC, evidence, answer
        )
        self.assertFalse(result.ok)
        self.assertIn("evidence passages were cited", result.issues[0])

    def test_fully_used_evidence_passes(self):
        evidence = _evidence("alpha content here", "beta content here")
        answer = _answer(("alpha content here", [0]), ("beta content here", [1]))
        self.assertTrue(
            LexicalGroundingVerifier(check_hallucination=False)
            .verify(_TOPIC, evidence, answer).ok
        )

    def test_uncovered_sub_question_is_named(self):
        # Provenance from the decomposed arm: sq2 has a passage but no citation.
        evidence = _evidence(
            "alpha content here", "beta content here",
            features={0: {"ain_hit_sq1": 1.0}, 1: {"ain_hit_sq2": 1.0}},
        )
        answer = _answer(("alpha content here", [0]))
        result = LexicalGroundingVerifier(
            check_hallucination=False, min_evidence_used=0.0
        ).verify(_TOPIC, evidence, answer)
        self.assertFalse(result.ok)
        self.assertIn("sub-question", result.issues[0])
        self.assertIn("2", result.issues[0])

    def test_all_sub_questions_covered_passes(self):
        evidence = _evidence(
            "alpha content here", "beta content here",
            features={0: {"ain_hit_sq1": 1.0}, 1: {"ain_hit_sq2": 1.0}},
        )
        answer = _answer(("alpha content here", [0]), ("beta content here", [1]))
        self.assertTrue(
            LexicalGroundingVerifier(check_hallucination=False)
            .verify(_TOPIC, evidence, answer).ok
        )

    def test_composite_mode_without_provenance_still_works(self):
        evidence = _evidence("alpha content here", "beta content here")
        answer = _answer(("alpha content here", [0]), ("beta content here", [1]))
        self.assertTrue(
            LexicalGroundingVerifier(check_hallucination=False)
            .verify(_TOPIC, evidence, answer).ok
        )


class TestVerifierEdgeCases(unittest.TestCase):
    def test_empty_answer_is_not_rejected(self):
        # The generator already falls back on an empty answer; rejecting here
        # would burn a regeneration to reach the same fallback.
        self.assertTrue(
            LexicalGroundingVerifier().verify(_TOPIC, _evidence("a"), RagAnswer([], [])).ok
        )

    def test_no_evidence_is_not_rejected(self):
        self.assertTrue(
            LexicalGroundingVerifier().verify(_TOPIC, [], _answer(("x", [0]))).ok
        )

    def test_out_of_range_citation_is_ignored_not_crashed(self):
        answer = _answer(("some sentence here", [99]))
        self.assertIsInstance(
            LexicalGroundingVerifier().verify(_TOPIC, _evidence("a b c"), answer),
            VerificationResult,
        )


class TestVerificationTemplate(unittest.TestCase):
    def test_carries_question_passages_and_sentences(self):
        prompt = render_verification_prompt(
            question="Why does X?",
            passage_texts=["alpha text", "beta text"],
            sentences=[("first sentence", [0]), ("second sentence", [1])],
        )
        self.assertIn("Why does X?", prompt)
        self.assertIn("[0] alpha text", prompt)
        self.assertIn("[1] beta text", prompt)
        self.assertIn("first sentence", prompt)

    def test_sentences_are_numbered_from_one(self):
        # `_parse` range-checks reported numbers against 1..n, so numbering from
        # zero would silently discard every finding and report a clean answer.
        prompt = render_verification_prompt(
            question="q", passage_texts=["p"],
            sentences=[("first", [0]), ("second", [0])],
        )
        self.assertIn("1. first", prompt)
        self.assertIn("2. second", prompt)
        self.assertNotIn("0. first", prompt)

    def test_reply_labels_match_what_the_parser_reads(self):
        prompt = render_verification_prompt(
            question="q", passage_texts=["p"], sentences=[("s", [0])]
        )
        self.assertIn("UNSUPPORTED:", prompt)
        self.assertIn("MISSING:", prompt)
        self.assertIn("NONE", prompt)

    def test_passages_are_capped(self):
        prompt = render_verification_prompt(
            question="q", passage_texts=["y" * 4000], sentences=[("s", [0])],
            passage_char_cap=50,
        )
        self.assertIn("y" * 50, prompt)
        self.assertNotIn("y" * 51, prompt)

    def test_thinking_absent_by_default(self):
        prompt = render_verification_prompt(
            question="q", passage_texts=["p"], sentences=[("s", [0])]
        )
        self.assertNotIn("<think>", prompt)

    def test_thinking_adds_judging_specific_steps(self):
        prompt = render_verification_prompt(
            question="q", passage_texts=["p"], sentences=[("s", [0])], thinking=True
        )
        self.assertIn("<think>", prompt)
        self.assertIn("</think>", prompt)
        self.assertIn("paraphrase and drift", prompt)
        self.assertIn("does NOT cite is still unsupported", prompt)
        # The report contract must survive the added block.
        self.assertIn("UNSUPPORTED:", prompt)
        self.assertIn("MISSING:", prompt)

    def test_thinking_raises_the_judge_token_budget(self):
        # A truncated block yields an empty judgment, which reads as "passed" --
        # verification would switch itself off rather than fail.
        plain = LLMVerifier(ScriptedLLM(), thinking=False)
        thinking = LLMVerifier(ScriptedLLM(), thinking=True)
        self.assertGreater(thinking.max_tokens, plain.max_tokens)

    def test_judge_thinking_is_stripped_before_parsing(self):
        llm = ScriptedLLM(
            "<think>Sentence 2 cites [0] but [0] is about revenue, not oversight. "
            "Numbers 1, 3, 7 appear here as noise.</think>\n"
            "UNSUPPORTED: 2\nMISSING: NONE"
        )
        verifier = LLMVerifier(llm, thinking=True)
        result = verifier.verify(
            _TOPIC, _evidence("alpha content", "beta content"),
            _answer(("alpha content", [0]), ("beta content", [1])),
        )
        self.assertFalse(result.ok)
        # Only sentence 2 -- the digits inside the think block must not be read.
        self.assertIn("2", result.issues[0])
        self.assertNotIn("1, 3, 7", result.issues[0])

    def test_truncated_judge_thinking_passes_rather_than_misreporting(self):
        llm = ScriptedLLM("<think>Checking sentence 1 against passage [0] and it ran out")
        verifier = LLMVerifier(llm, thinking=True)
        result = verifier.verify(
            _TOPIC, _evidence("alpha content"), _answer(("alpha content", [0]))
        )
        self.assertTrue(result.ok)

    def test_round_trips_through_the_parser(self):
        # The template's own worked example must be readable by the parser it
        # is instructing -- otherwise verification silently no-ops.
        result = LLMVerifier._parse("UNSUPPORTED: 2, 5\nMISSING: retention outcomes", 6)
        self.assertFalse(result.ok)
        self.assertIn("2, 5", result.issues[0])
        self.assertIn("retention outcomes", result.issues[1])


class TestLLMVerifier(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence("alpha content", "beta content")
        self.answer = _answer(("alpha content", [0]), ("beta content", [1]))

    def test_clean_judgment_passes(self):
        verifier = LLMVerifier(ScriptedLLM("UNSUPPORTED: NONE\nMISSING: NONE"))
        self.assertTrue(verifier.verify(_TOPIC, self.evidence, self.answer).ok)

    def test_unsupported_sentences_are_reported(self):
        verifier = LLMVerifier(ScriptedLLM("UNSUPPORTED: 2\nMISSING: NONE"))
        result = verifier.verify(_TOPIC, self.evidence, self.answer)
        self.assertFalse(result.ok)
        self.assertIn("2", result.issues[0])
        self.assertTrue(result.instructions)

    def test_missing_material_is_reported(self):
        verifier = LLMVerifier(
            ScriptedLLM("UNSUPPORTED: NONE\nMISSING: retention outcomes are never mentioned")
        )
        result = verifier.verify(_TOPIC, self.evidence, self.answer)
        self.assertFalse(result.ok)
        self.assertIn("retention outcomes", result.issues[0])

    def test_out_of_range_line_numbers_are_discarded(self):
        verifier = LLMVerifier(ScriptedLLM("UNSUPPORTED: 42\nMISSING: NONE"))
        self.assertTrue(verifier.verify(_TOPIC, self.evidence, self.answer).ok)

    def test_unparsable_judgment_passes_the_answer(self):
        # A verifier that rejected on its own parse errors would regenerate good
        # answers and ship the last draft anyway.
        verifier = LLMVerifier(ScriptedLLM("I think the answer looks fine overall!"))
        self.assertTrue(verifier.verify(_TOPIC, self.evidence, self.answer).ok)


class TestUnreadableJudgementsAreVisible(unittest.TestCase):
    """A verifier that cannot judge must not look like one finding nothing."""

    def setUp(self):
        self.evidence = _evidence("alpha content", "beta content")
        self.answer = _answer(("alpha content", [0]), ("beta content", [1]))

    def _verify(self, verifier):
        return verifier.verify(_TOPIC, self.evidence, self.answer)

    def test_clean_verdict_is_marked_checked(self):
        result = self._verify(LLMVerifier(ScriptedLLM("UNSUPPORTED: NONE\nMISSING: NONE")))
        self.assertTrue(result.ok)
        self.assertTrue(result.checked)

    def test_unparsable_verdict_is_marked_unchecked(self):
        result = self._verify(LLMVerifier(ScriptedLLM("looks fine to me!")))
        self.assertTrue(result.ok)
        self.assertFalse(result.checked)

    def test_empty_response_is_marked_unchecked_and_counted(self):
        verifier = LLMVerifier(ScriptedLLM(""), thinking=True)
        result = self._verify(verifier)
        self.assertFalse(result.checked)
        self.assertEqual(verifier.stats["empty_responses"], 1)

    def test_lexical_verifier_always_reports_checked(self):
        result = LexicalGroundingVerifier().verify(_TOPIC, self.evidence, self.answer)
        self.assertTrue(result.checked)

    def test_stats_track_what_was_actually_judged(self):
        verifier = LLMVerifier(
            ScriptedLLM("UNSUPPORTED: NONE\nMISSING: NONE", "prose", "UNSUPPORTED: NONE\nMISSING: NONE")
        )
        for _ in range(3):
            self._verify(verifier)
        self.assertEqual(verifier.stats["judged"], 3)
        self.assertEqual(verifier.stats["parsed"], 2)
        self.assertEqual(verifier.stats["unparsable"], 1)

    def test_a_good_verdict_resets_the_consecutive_counter(self):
        verifier = LLMVerifier(ScriptedLLM("prose", "UNSUPPORTED: NONE\nMISSING: NONE", "prose"))
        for _ in range(3):
            self._verify(verifier)
        self.assertEqual(verifier.stats["consecutive_unparsable"], 1)
        self.assertFalse(verifier.degraded)


class TestBrokenVerifierDegradesLoudly(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence("alpha content", "beta content")
        self.answer = _answer(("alpha content", [0]), ("beta content", [1]))

    def _run(self, verifier, n):
        return [verifier.verify(_TOPIC, self.evidence, self.answer) for _ in range(n)]

    def test_degrades_after_consecutive_failures(self):
        verifier = LLMVerifier(ScriptedLLM(*[""] * 5), max_consecutive_unparsable=3)
        self._run(verifier, 3)
        self.assertTrue(verifier.degraded)

    def test_stops_calling_a_broken_judge(self):
        # Each call is a generation-sized request; 119 of them to learn the
        # same thing is the cost this avoids.
        llm = ScriptedLLM(*[""] * 10)
        verifier = LLMVerifier(llm, max_consecutive_unparsable=3)
        self._run(verifier, 8)
        self.assertEqual(len(llm.prompts), 3)

    def test_degraded_results_stay_unchecked(self):
        verifier = LLMVerifier(ScriptedLLM(*[""] * 6), max_consecutive_unparsable=2)
        results = self._run(verifier, 5)
        self.assertTrue(all(r.ok for r in results))
        self.assertTrue(all(not r.checked for r in results))

    def test_strict_mode_raises_instead_of_degrading(self):
        verifier = LLMVerifier(
            ScriptedLLM(*[""] * 5), max_consecutive_unparsable=2, strict=True
        )
        with self.assertRaises(VerifierUnavailableError) as ctx:
            self._run(verifier, 5)
        self.assertIn("NOT running", str(ctx.exception))

    def test_error_message_names_the_likely_cause(self):
        verifier = LLMVerifier(
            ScriptedLLM(*[""] * 3), thinking=True, max_consecutive_unparsable=2, strict=True
        )
        with self.assertRaises(VerifierUnavailableError) as ctx:
            self._run(verifier, 3)
        message = str(ctx.exception)
        self.assertIn("max_tokens", message)
        self.assertIn("thinking", message)

    def test_composite_reports_checked_when_the_free_check_ran(self):
        # The lexical member judged it even though the LLM member could not.
        composite = CompositeVerifier(
            [LexicalGroundingVerifier(), LLMVerifier(ScriptedLLM("prose"))]
        )
        result = composite.verify(_TOPIC, self.evidence, self.answer)
        self.assertTrue(result.checked)

    def test_composite_of_only_broken_members_reports_unchecked(self):
        composite = CompositeVerifier([LLMVerifier(ScriptedLLM("prose"))])
        self.assertFalse(composite.verify(_TOPIC, self.evidence, self.answer).checked)


class AlwaysFails(AnswerVerifier):
    name = "always_fails"

    def __init__(self):
        self.calls = 0

    def verify(self, topic, evidence, answer):
        self.calls += 1
        return VerificationResult(
            ok=False, issues=["sentence 1 is unsupported"], instructions=["cite the passage"]
        )


class FailsOnce(AnswerVerifier):
    name = "fails_once"

    def __init__(self):
        self.calls = 0

    def verify(self, topic, evidence, answer):
        self.calls += 1
        if self.calls == 1:
            return VerificationResult(ok=False, issues=["bad draft"], instructions=["fix it"])
        return VerificationResult(ok=True)


class TestRegenerationLoop(unittest.TestCase):
    def setUp(self):
        self.evidence = _evidence(
            "Congestion pricing generates dedicated revenue for transit improvements.",
            "Public reporting holds transit agencies accountable for spending.",
        )

    def test_no_verifier_means_exactly_one_call(self):
        llm = ScriptedLLM(_answer_json(("Congestion pricing generates revenue.", [0])))
        LLMGenerator(llm).generate(_TOPIC, self.evidence, 1024, 3)
        self.assertEqual(len(llm.prompts), 1)

    def test_failed_check_triggers_one_regeneration(self):
        llm = ScriptedLLM(
            _answer_json(("first draft about revenue", [0])),
            _answer_json(("second draft about revenue", [0])),
        )
        verifier = FailsOnce()
        answer = LLMGenerator(llm, verifier=verifier, max_attempts=2).generate(
            _TOPIC, self.evidence, 1024, 3
        )
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(answer.sentences[0].text, "second draft about revenue")

    def test_retry_prompt_carries_the_validator_instructions(self):
        llm = ScriptedLLM(
            _answer_json(("first draft about revenue", [0])),
            _answer_json(("second draft about revenue", [0])),
        )
        LLMGenerator(llm, verifier=FailsOnce(), max_attempts=2).generate(
            _TOPIC, self.evidence, 1024, 3
        )
        retry = llm.prompts[1]
        self.assertIn("REVISION REQUIRED", retry)
        self.assertIn("bad draft", retry)
        self.assertIn("fix it", retry)
        self.assertNotIn("REVISION REQUIRED", llm.prompts[0])

    def test_passing_check_does_not_regenerate(self):
        llm = ScriptedLLM(_answer_json(("Congestion pricing generates revenue.", [0])))
        verifier = LexicalGroundingVerifier(check_omission=False)
        LLMGenerator(llm, verifier=verifier, max_attempts=2).generate(
            _TOPIC, self.evidence, 1024, 3
        )
        self.assertEqual(len(llm.prompts), 1)

    def test_attempts_are_capped_and_the_last_draft_ships(self):
        llm = ScriptedLLM(*[_answer_json((f"draft {i} about revenue", [0])) for i in range(5)])
        verifier = AlwaysFails()
        answer = LLMGenerator(llm, verifier=verifier, max_attempts=3).generate(
            _TOPIC, self.evidence, 1024, 3
        )
        self.assertEqual(len(llm.prompts), 3)
        # Verified after drafts 1 and 2; draft 3 ships without a further check.
        self.assertEqual(verifier.calls, 2)
        self.assertEqual(answer.sentences[0].text, "draft 2 about revenue")

    def test_unparsable_draft_falls_back_without_retrying(self):
        llm = ScriptedLLM("not json at all")
        answer = LLMGenerator(
            llm, verifier=AlwaysFails(), max_attempts=3, fallback=ExtractiveGenerator()
        ).generate(_TOPIC, self.evidence, 1024, 3)
        self.assertEqual(len(llm.prompts), 1)
        self.assertTrue(answer.sentences)

    def test_verification_off_is_byte_identical_to_single_pass(self):
        script = _answer_json(("Congestion pricing generates revenue.", [0]))
        without = LLMGenerator(ScriptedLLM(script)).generate(_TOPIC, self.evidence, 1024, 3)
        explicit = LLMGenerator(ScriptedLLM(script), verifier=None, max_attempts=1).generate(
            _TOPIC, self.evidence, 1024, 3
        )
        self.assertEqual(
            [(s.text, s.citations) for s in without.sentences],
            [(s.text, s.citations) for s in explicit.sentences],
        )


class TestCompositeVerifier(unittest.TestCase):
    def test_short_circuits_on_the_first_failure(self):
        first, second = AlwaysFails(), AlwaysFails()
        result = CompositeVerifier([first, second]).verify(
            _TOPIC, _evidence("a"), _answer(("x", [0]))
        )
        self.assertFalse(result.ok)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)  # the LLM judge never paid for

    def test_passes_when_all_pass(self):
        evidence = _evidence("alpha content here", "beta content here")
        answer = _answer(("alpha content here", [0]), ("beta content here", [1]))
        composite = CompositeVerifier(
            [LexicalGroundingVerifier(), LLMVerifier(ScriptedLLM("UNSUPPORTED: NONE\nMISSING: NONE"))]
        )
        self.assertTrue(composite.verify(_TOPIC, evidence, answer).ok)


if __name__ == "__main__":
    unittest.main()
