import tempfile
import unittest
from pathlib import Path

from trecrag.data.topics import TopicFormatError, load_topics


class TestLoadTopics(unittest.TestCase):
    def test_loads_two_column_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.tsv"
            path.write_text("t1\tfirst topic text\nt2\tsecond topic text\n", encoding="utf-8")
            topics = load_topics(path)
        self.assertEqual(len(topics), 2)
        self.assertEqual(topics[0].topic_id, "t1")
        self.assertEqual(topics[0].text, "first topic text")
        self.assertEqual(topics[1].topic_id, "t2")

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.tsv"
            path.write_text("t1\ttext one\n\nt2\ttext two\n", encoding="utf-8")
            topics = load_topics(path)
        self.assertEqual([t.topic_id for t in topics], ["t1", "t2"])

    def test_rejects_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.tsv"
            path.write_text("t1\ttoo\tmany\tcolumns\n", encoding="utf-8")
            with self.assertRaises(TopicFormatError):
                load_topics(path)

    def test_rejects_duplicate_topic_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.tsv"
            path.write_text("t1\ttext one\nt1\ttext two\n", encoding="utf-8")
            with self.assertRaises(TopicFormatError):
                load_topics(path)

    def test_preserves_text_exactly_including_tabs_in_content_are_rejected(self):
        # Topic text must not itself contain a tab (would break the 2-column contract);
        # this documents that expectation rather than silently mangling data.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "topics.tsv"
            path.write_text("t1\ttext\twith\ttabs\n", encoding="utf-8")
            with self.assertRaises(TopicFormatError):
                load_topics(path)


if __name__ == "__main__":
    unittest.main()
