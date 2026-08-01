import unittest

from trecrag.generation.postprocess import postprocess_rag_answer
from trecrag.schema import AnswerSentence


class TestPostprocessRagAnswer(unittest.TestCase):
    def test_clamps_citations_to_max_per_sentence(self):
        references = ["a", "b", "c", "d"]
        sentences = [AnswerSentence(text="claim.", citations=[0, 1, 2, 3])]
        result = postprocess_rag_answer(references, sentences, max_words=1024, max_citations=2)
        self.assertEqual(result.sentences[0].citations, [0, 1])

    def test_dedupes_citations_before_capping(self):
        references = ["a", "b", "c"]
        sentences = [AnswerSentence(text="claim.", citations=[2, 2, 0])]
        result = postprocess_rag_answer(references, sentences, max_words=1024, max_citations=2)
        # Deduped to [2, 0] (not capped-with-a-duplicate [2, 2]); references
        # remap sorts old indices ascending ([0, 2] -> new 0, 1), so citation
        # order [2, 0] becomes [1, 0].
        self.assertEqual(result.references, ["a", "c"])
        self.assertEqual(result.sentences[0].citations, [1, 0])

    def test_drops_invalid_citation_indices(self):
        references = ["a", "b"]
        sentences = [AnswerSentence(text="claim.", citations=[0, 5, -1])]
        result = postprocess_rag_answer(references, sentences, max_words=1024, max_citations=3)
        self.assertEqual(result.sentences[0].citations, [0])

    def test_drops_sentence_with_no_valid_citations(self):
        references = ["a"]
        sentences = [
            AnswerSentence(text="ungrounded claim.", citations=[7]),
            AnswerSentence(text="grounded claim.", citations=[0]),
        ]
        result = postprocess_rag_answer(references, sentences, max_words=1024, max_citations=3)
        self.assertEqual(len(result.sentences), 1)
        self.assertEqual(result.sentences[0].text, "grounded claim.")

    def test_drops_uncited_references_and_remaps_indices(self):
        references = ["used", "unused", "also_used"]
        sentences = [
            AnswerSentence(text="first.", citations=[0]),
            AnswerSentence(text="second.", citations=[2]),
        ]
        result = postprocess_rag_answer(references, sentences, max_words=1024, max_citations=3)
        self.assertEqual(result.references, ["used", "also_used"])
        self.assertEqual(result.sentences[0].citations, [0])
        self.assertEqual(result.sentences[1].citations, [1])

    def test_enforces_word_budget_by_dropping_trailing_sentences(self):
        references = ["a", "b"]
        sentences = [
            AnswerSentence(text=" ".join(["word"] * 5), citations=[0]),
            AnswerSentence(text=" ".join(["word"] * 5), citations=[1]),
        ]
        result = postprocess_rag_answer(references, sentences, max_words=7, max_citations=3)
        self.assertEqual(len(result.sentences), 1)
        self.assertEqual(result.references, ["a"])

    def test_oversized_sentence_is_skipped_not_fatal_to_later_sentences(self):
        """A single over-budget "sentence" must not discard everything after
        it -- split_sentences only breaks on [.!?], so an unpunctuated blob
        stays one giant unit, and a `break` here would silently zero out an
        otherwise-valid answer."""
        references = ["a", "b", "c"]
        sentences = [
            AnswerSentence(text="word word", citations=[0]),
            AnswerSentence(text=" ".join(["word"] * 10), citations=[1]),
            AnswerSentence(text="word word word", citations=[2]),
        ]
        result = postprocess_rag_answer(references, sentences, max_words=6, max_citations=3)
        self.assertEqual(len(result.sentences), 2)
        self.assertEqual(result.references, ["a", "c"])

    def test_empty_sentences_are_skipped(self):
        references = ["a"]
        sentences = [AnswerSentence(text="   ", citations=[0])]
        result = postprocess_rag_answer(references, sentences, max_words=1024, max_citations=3)
        self.assertEqual(result.sentences, [])
        self.assertEqual(result.references, [])


if __name__ == "__main__":
    unittest.main()
