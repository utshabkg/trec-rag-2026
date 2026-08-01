import json
import tempfile
import unittest
from pathlib import Path

from trecrag.schema import Topic
from trecrag.validate import validate_rag_jsonl, validate_retrieval_run


def _write(tmp: Path, name: str, content: str) -> Path:
    path = tmp / name
    path.write_text(content, encoding="utf-8")
    return path


class TestValidateRetrievalRun(unittest.TestCase):
    def test_valid_run_has_no_violations(self):
        content = "1 Q0 d1 1 10.0 run\n1 Q0 d2 2 9.0 run\n2 Q0 d3 1 5.0 run\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "run.tsv", content)
            self.assertEqual(validate_retrieval_run(path), [])

    def test_detects_wrong_column_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "run.tsv", "1 Q0 d1 1 10.0\n")
            violations = validate_retrieval_run(path)
        self.assertTrue(any("6 whitespace" in v for v in violations))

    def test_detects_rank_not_restarting_at_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "run.tsv", "1 Q0 d1 2 10.0 run\n")
            violations = validate_retrieval_run(path)
        self.assertTrue(any("out of sequence" in v for v in violations))

    def test_detects_score_increase_within_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "run.tsv", "1 Q0 d1 1 5.0 run\n1 Q0 d2 2 9.0 run\n")
            violations = validate_retrieval_run(path)
        self.assertTrue(any("increased" in v for v in violations))

    def test_detects_missing_topic_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "run.tsv", "1 Q0 d1 1 10.0 run\n")
            violations = validate_retrieval_run(path, expected_topic_ids={"1", "2"})
        self.assertTrue(any("missing output" in v for v in violations))


class TestValidateRagJsonl(unittest.TestCase):
    def _valid_record(self):
        return {
            "metadata": {
                "team_id": "team",
                "narrative_id": "t1",
                "narrative": "the topic text",
                "run_id": "run",
                "run_desc": "desc",
            },
            "references": ["d1", "d2"],
            "answer": [{"text": "a claim.", "citations": [0]}, {"text": "another claim.", "citations": [1]}],
        }

    def test_valid_record_has_no_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(self._valid_record()) + "\n")
            self.assertEqual(validate_rag_jsonl(path), [])

    def test_detects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", "{not json}\n")
            violations = validate_rag_jsonl(path)
        self.assertTrue(any("invalid JSON" in v for v in violations))

    def test_allows_extra_metadata_keys(self):
        """rag-task.md: metadata "may also contain any additional
        participant-defined fields", and validators must not reject them."""
        record = self._valid_record()
        record["metadata"]["generator"] = "llama3.3:70b"
        record["metadata"]["retrieval_depth"] = 50
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path)
        self.assertEqual(violations, [])

    def test_detects_too_many_citations(self):
        record = self._valid_record()
        record["references"] = ["d1", "d2", "d3", "d4"]
        record["answer"] = [{"text": "claim.", "citations": [0, 1, 2, 3]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path, max_citations_per_sentence=3)
        self.assertTrue(any("exceeds max" in v for v in violations))

    def test_detects_invalid_citation_index(self):
        record = self._valid_record()
        record["answer"][0]["citations"] = [99]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path)
        self.assertTrue(any("citation index" in v for v in violations))

    def test_allows_uncited_references(self):
        """rag-task.md: "Do not reject or penalize a RAG object because
        `references` contains uncited document IDs." """
        record = self._valid_record()
        record["references"] = ["d1", "d2", "d3"]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path)
        self.assertEqual(violations, [])

    def test_detects_empty_answer(self):
        """Syntactically valid, scores zero, and passes every other check."""
        record = self._valid_record()
        record["answer"] = []
        record["references"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path)
        self.assertTrue(any("would score zero" in v for v in violations))

    def test_detects_word_count_over_limit(self):
        record = self._valid_record()
        record["references"] = ["d1"]
        record["answer"] = [{"text": " ".join(["word"] * 10), "citations": [0]}]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path, max_words=5)
        self.assertTrue(any("exceeds max" in v for v in violations))

    def test_checks_narrative_matches_topic_exactly(self):
        record = self._valid_record()
        record["metadata"]["narrative"] = "a different narrative"
        topics = [Topic(topic_id="t1", text="the topic text")]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", json.dumps(record) + "\n")
            violations = validate_rag_jsonl(path, topics=topics)
        self.assertTrue(any("does not match" in v for v in violations))

    def test_detects_duplicate_narrative_id(self):
        record = self._valid_record()
        content = json.dumps(record) + "\n" + json.dumps(record) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp), "out.jsonl", content)
            violations = validate_rag_jsonl(path)
        self.assertTrue(any("duplicate narrative_id" in v for v in violations))


if __name__ == "__main__":
    unittest.main()
