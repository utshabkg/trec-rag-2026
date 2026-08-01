"""Passage-selection tests.

The failure mode this module exists to prevent is silent: if selection picks the
wrong span, or quietly returns the document prefix, everything still runs and
still validates -- the run just reranks and cites book front matter, which is
exactly what was happening before.
"""

import unittest

from trecrag.chunking import PassageSelector, iter_windows, select_passage
from trecrag.schema import Candidate
from trecrag.text_utils import tokenize

FILLER = "Copyright notice and table of contents boilerplate text here. " * 40
TARGET = (
    "Congestion pricing funds transit capital projects and the revenue is "
    "dedicated to subway signal modernization under the MTA capital plan. "
)


class TestIterWindows(unittest.TestCase):
    def test_short_text_is_one_window(self):
        self.assertEqual(iter_windows("short", 100, 50), [(0, "short")])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(iter_windows("", 100, 50), [])

    def test_windows_overlap_and_cover_the_whole_text(self):
        text = "x" * 1000
        windows = iter_windows(text, 400, 200)
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        # Every character must appear in some window, or a relevant span could
        # fall in a gap and be unreachable.
        covered = set()
        for offset, window in windows:
            covered.update(range(offset, offset + len(window)))
        self.assertEqual(covered, set(range(len(text))))


class TestSelectPassage(unittest.TestCase):
    def test_finds_relevant_span_buried_deep_in_the_document(self):
        query_tokens = set(tokenize("congestion pricing MTA subway revenue"))
        doc = FILLER + TARGET + FILLER
        selected = select_passage(doc, query_tokens, window_chars=500, stride_chars=250)
        self.assertIn("congestion pricing", selected.lower())

    def test_selection_beats_the_prefix(self):
        """The whole point: the opening of a long document is usually not the
        most query-relevant part of it."""
        query_tokens = set(tokenize("congestion pricing MTA subway revenue"))
        doc = FILLER + TARGET + FILLER

        def overlap(text):
            tokens = tokenize(text)
            return sum(1 for t in tokens if t in query_tokens) / (len(tokens) ** 0.5)

        self.assertGreater(
            overlap(select_passage(doc, query_tokens, 500, 250)), overlap(doc[:500])
        )

    def test_multiple_windows_cover_separated_facets(self):
        """Narratives bundle several sub-questions; the spans answering them are
        often far apart in the same document."""
        facet_a = "Nursing pipeline partnerships with local schools. " * 4
        facet_b = "Leadership accountability metrics and promotion practices. " * 4
        doc = facet_a + FILLER + facet_b
        query_tokens = set(tokenize("pipeline partnerships leadership accountability promotion"))
        selected = select_passage(
            doc, query_tokens, window_chars=400, stride_chars=200, windows_per_doc=2
        )
        self.assertIn("pipeline partnerships", selected.lower())
        self.assertIn("accountability", selected.lower())

    def test_selected_windows_stay_in_document_order(self):
        early = "Alpha keyword sentence. " * 3
        late = "Alpha keyword sentence again. " * 3
        doc = early + FILLER + late
        selected = select_passage(
            doc, set(tokenize("alpha keyword")), window_chars=300, stride_chars=150, windows_per_doc=2
        )
        parts = selected.split(" ... ")
        if len(parts) == 2:
            self.assertLess(doc.index(parts[0][:40]), doc.index(parts[1][:40]))

    def test_empty_query_tokens_returns_something_usable(self):
        doc = FILLER
        self.assertTrue(select_passage(doc, set(), 400, 200))


class TestPassageSelector(unittest.TestCase):
    def test_rewrites_long_candidates_and_leaves_short_ones_alone(self):
        long_doc = FILLER + TARGET + FILLER
        short_doc = "Already passage sized."
        candidates = [
            Candidate(docid="long", score=1.0, rank=1, text=long_doc),
            Candidate(docid="short", score=1.0, rank=2, text=short_doc),
        ]
        PassageSelector(window_chars=500, stride_chars=250, min_doc_chars=1000).apply(
            "congestion pricing MTA subway revenue", candidates
        )
        self.assertLess(len(candidates[0].text), len(long_doc))
        self.assertIn("congestion pricing", candidates[0].text.lower())
        self.assertEqual(candidates[1].text, short_doc)

    def test_textless_candidates_survive_untouched(self):
        candidates = [Candidate(docid="none", score=1.0, rank=1, text=None)]
        PassageSelector().apply("any query", candidates)
        self.assertIsNone(candidates[0].text)

    def test_empty_query_is_a_no_op(self):
        doc = FILLER + TARGET
        candidates = [Candidate(docid="d", score=1.0, rank=1, text=doc)]
        PassageSelector().apply("!!!", candidates)
        self.assertEqual(candidates[0].text, doc)


if __name__ == "__main__":
    unittest.main()
