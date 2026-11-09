import json
import unittest

from trecrag.clients.llm_client import LLMClient, strip_reasoning
from trecrag.generation.generator import ExtractiveGenerator, LLMGenerator
from trecrag.generation.prompts import Revision, render_answer_prompt
from trecrag.schema import Candidate, Topic

_EVIDENCE = [
    Candidate(docid="d1", score=3.0, rank=1, text="Congestion pricing funds transit improvements."),
    Candidate(docid="d2", score=2.0, rank=2, text="Revenue accountability requires public reporting."),
]
_TOPIC = Topic("rag2026-0", "What are the tradeoffs of congestion pricing?")


class ScriptedLLM(LLMClient):
    """Replays canned completions and records the prompts it was given."""

    def __init__(self, *responses, strip=True):
        self.responses = list(responses)
        self.prompts = []
        self.max_tokens_seen = []
        self.strip = strip

    def complete(self, prompt, *, max_tokens=1024):
        self.prompts.append(prompt)
        self.max_tokens_seen.append(max_tokens)
        response = self.responses.pop(0) if self.responses else ""
        # Real clients strip before returning; mirror that so tests exercise
        # the same text the generator actually sees.
        return strip_reasoning(response) if self.strip else response


def _answer_json(*sentences):
    return json.dumps([{"text": t, "citations": c} for t, c in sentences])


class TestTemplateRendering(unittest.TestCase):
    def test_carries_question_passages_and_limits(self):
        prompt = render_answer_prompt(
            question="Why does X?", passage_texts=["alpha text", "beta text"],
            max_words=1024, max_citations=3,
        )
        self.assertIn("Why does X?", prompt)
        self.assertIn("[0] alpha text", prompt)
        self.assertIn("[1] beta text", prompt)
        self.assertIn("at most 3 zero-based indices", prompt)
        self.assertIn("hard maximum 1024", prompt)

    def test_states_an_explicit_word_range_not_only_a_ceiling(self):
        # "at most N words" alone produced ~250-word answers against 1024.
        prompt = render_answer_prompt("q", ["p"], max_words=1024, max_citations=3)
        self.assertIn("614-921 words", prompt)

    def test_retains_the_measured_failure_instructions(self):
        prompt = render_answer_prompt("q", ["p"], max_words=1024, max_citations=3)
        for rule in (
            "Do not restate or summarise the question",
            "Never mention the evidence passages as objects",
            "breadth of coverage matters more than depth",
            "conditional guidance",
        ):
            self.assertIn(rule, prompt)

    def test_passages_are_capped(self):
        prompt = render_answer_prompt(
            "q", ["x" * 5000], max_words=1024, max_citations=3, passage_char_cap=100
        )
        self.assertIn("x" * 100, prompt)
        self.assertNotIn("x" * 101, prompt)

    def test_undefined_variables_fail_loudly(self):
        from jinja2 import StrictUndefined

        from trecrag.generation.prompts import _environment

        self.assertIs(_environment().undefined, StrictUndefined)


class TestThinkingBlock(unittest.TestCase):
    def test_absent_by_default(self):
        prompt = render_answer_prompt("q", ["p"], max_words=1024, max_citations=3)
        self.assertNotIn("<think>", prompt)

    def test_requested_with_numbered_reasoning_steps(self):
        prompt = render_answer_prompt(
            "q", ["p"], max_words=1024, max_citations=3, thinking=True
        )
        self.assertIn("<think>", prompt)
        self.assertIn("</think>", prompt)
        for step in ("1.", "2.", "3.", "4.", "5."):
            self.assertIn(step, prompt)
        self.assertIn("List every distinct thing the request asks for", prompt)

    def test_thinking_raises_the_token_budget(self):
        # The block is spent before any JSON is emitted; without headroom this
        # reintroduces the truncated-array parse failure.
        plain = LLMGenerator(ScriptedLLM(), thinking=False)
        thinking = LLMGenerator(ScriptedLLM(), thinking=True)
        self.assertGreater(thinking.max_tokens, plain.max_tokens)

    def test_explicit_max_tokens_wins(self):
        self.assertEqual(LLMGenerator(ScriptedLLM(), thinking=True, max_tokens=999).max_tokens, 999)


class TestThinkingIsStrippedFromOutput(unittest.TestCase):
    def test_answer_survives_a_think_block(self):
        llm = ScriptedLLM(
            "<think>Ask 1 is cost, ask 2 is equity. Passage [0] covers cost.</think>"
            + _answer_json(("Congestion pricing funds transit.", [0]))
        )
        answer = LLMGenerator(llm, thinking=True).generate(_TOPIC, _EVIDENCE, 1024, 3)
        self.assertEqual(len(answer.sentences), 1)
        self.assertNotIn("<think>", answer.sentences[0].text)
        self.assertNotIn("Ask 1 is cost", answer.sentences[0].text)

    def test_reasoning_digits_do_not_become_the_answer(self):
        # The failure this guards: brackets inside reasoning parsed as the JSON.
        llm = ScriptedLLM(
            "<think>Consider [0], [1] and rank them 2, 1</think>"
            + _answer_json(("Real sentence.", [1]))
        )
        answer = LLMGenerator(llm, thinking=True).generate(_TOPIC, _EVIDENCE, 1024, 3)
        self.assertEqual(answer.sentences[0].text, "Real sentence.")

    def test_unterminated_think_block_does_not_corrupt_the_answer(self):
        # A block cut off by the token budget keeps no closing tag.
        truncated = "<think>Step 1: the asks are [0] and [1], I should rank"
        self.assertEqual(strip_reasoning(truncated), "")

    def test_unterminated_block_falls_back_rather_than_fabricating(self):
        llm = ScriptedLLM("<think>reasoning that ran out of budget with [0] in it")
        answer = LLMGenerator(llm, thinking=True, fallback=ExtractiveGenerator()).generate(
            _TOPIC, _EVIDENCE, 1024, 3
        )
        self.assertTrue(answer.sentences)  # extractive fallback, not garbage
        self.assertNotIn("reasoning that ran out", answer.sentences[0].text)

    def test_closed_block_after_the_answer_is_still_removed(self):
        llm = ScriptedLLM(_answer_json(("A sentence.", [0])) + "<think>afterthought</think>")
        answer = LLMGenerator(llm, thinking=True).generate(_TOPIC, _EVIDENCE, 1024, 3)
        self.assertEqual(answer.sentences[0].text, "A sentence.")


class TestRevisionBlock(unittest.TestCase):
    def test_absent_on_a_first_attempt(self):
        prompt = render_answer_prompt("q", ["p"], max_words=1024, max_citations=3)
        self.assertNotIn("REVISION REQUIRED", prompt)

    def test_empty_revision_is_treated_as_absent(self):
        prompt = render_answer_prompt(
            "q", ["p"], max_words=1024, max_citations=3, revision=Revision()
        )
        self.assertNotIn("REVISION REQUIRED", prompt)

    def test_carries_issues_and_instructions(self):
        prompt = render_answer_prompt(
            "q", ["p"], max_words=1024, max_citations=3,
            revision=Revision(issues=["sentence 2 is unsupported"], instructions=["drop it"]),
        )
        self.assertIn("REVISION REQUIRED", prompt)
        self.assertIn("sentence 2 is unsupported", prompt)
        self.assertIn("drop it", prompt)

    def test_tells_the_model_not_to_mention_the_correction(self):
        prompt = render_answer_prompt(
            "q", ["p"], max_words=1024, max_citations=3,
            revision=Revision(issues=["x"], instructions=["y"]),
        )
        self.assertIn("Do not refer to the previous draft", prompt)


if __name__ == "__main__":
    unittest.main()
