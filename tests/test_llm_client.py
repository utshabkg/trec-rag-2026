import unittest

from trecrag.clients.llm_client import strip_reasoning


class TestStripReasoning(unittest.TestCase):
    def test_plain_text_is_untouched(self):
        self.assertEqual(strip_reasoning("just an answer"), "just an answer")

    def test_closed_block_is_removed(self):
        self.assertEqual(
            strip_reasoning("<think>weighing the options</think>the answer"), "the answer"
        )

    def test_all_recognised_tag_names(self):
        for tag in ("think", "thinking", "reasoning"):
            self.assertEqual(strip_reasoning(f"<{tag}>notes</{tag}>answer"), "answer")

    def test_tag_matching_is_case_insensitive(self):
        self.assertEqual(strip_reasoning("<THINK>notes</THINK>answer"), "answer")

    def test_block_after_the_payload_is_removed(self):
        self.assertEqual(strip_reasoning("answer<think>afterthought</think>"), "answer")

    def test_multiple_blocks_are_removed(self):
        self.assertEqual(strip_reasoning("<think>a</think>X<think>b</think>Y"), "XY")


class TestUnterminatedBlock(unittest.TestCase):
    """A block cut off by the token budget keeps no closing tag.

    Matching only balanced pairs left the whole reasoning trace in place, and
    the generator's `_JSON_ARRAY_RE` then took the first bracketed span it found
    -- inside the reasoning. That parses, validates and submits without ever
    erroring, so the failure is silent.
    """

    def test_unterminated_block_is_dropped_to_end_of_string(self):
        self.assertEqual(strip_reasoning("<think>ran out of budget mid-thou"), "")

    def test_reasoning_brackets_cannot_survive_as_an_answer(self):
        truncated = '<think>Ranking these: [0] then [1], so the order is [{"text": "x"}]'
        self.assertEqual(strip_reasoning(truncated), "")

    def test_payload_before_an_unterminated_block_is_kept(self):
        self.assertEqual(strip_reasoning("the answer<think>trailing notes"), "the answer")

    def test_closed_block_then_unterminated_one(self):
        self.assertEqual(
            strip_reasoning("<think>a</think>answer<think>cut off here"), "answer"
        )

    def test_a_bare_closing_tag_is_not_treated_as_an_opener(self):
        self.assertEqual(strip_reasoning("answer</think>"), "answer</think>")


if __name__ == "__main__":
    unittest.main()
