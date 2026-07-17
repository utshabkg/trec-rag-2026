import json
import tempfile
import unittest
from pathlib import Path

from trecrag.output.writers import build_rag_record, write_rag_jsonl, write_retrieval_run
from trecrag.schema import AnswerSentence, Candidate, RagAnswer, Topic


class TestWriteRetrievalRun(unittest.TestCase):
    def test_writes_six_column_format(self):
        results = {
            "t1": [
                Candidate(docid="d1", score=12.5, rank=1),
                Candidate(docid="d2", score=9.25, rank=2),
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.tsv"
            write_retrieval_run(path, "my-run", results)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        fields = lines[0].split()
        self.assertEqual(len(fields), 6)
        self.assertEqual(fields, ["t1", "Q0", "d1", "1", "12.500000", "my-run"])


class TestRagRecord(unittest.TestCase):
    def test_build_rag_record_shape(self):
        topic = Topic(topic_id="rag2026-1", text="narrative text")
        rag_answer = RagAnswer(
            references=["d1", "d2"],
            sentences=[AnswerSentence(text="claim.", citations=[0, 1])],
        )
        record = build_rag_record("team-a", topic, "run-a", "desc", rag_answer)
        self.assertEqual(
            set(record["metadata"].keys()),
            {"team_id", "narrative_id", "narrative", "run_id", "run_desc"},
        )
        self.assertEqual(record["metadata"]["narrative_id"], "rag2026-1")
        self.assertEqual(record["metadata"]["narrative"], "narrative text")
        self.assertEqual(record["references"], ["d1", "d2"])
        self.assertEqual(record["answer"], [{"text": "claim.", "citations": [0, 1]}])

    def test_write_rag_jsonl_one_object_per_line(self):
        records = [{"a": 1}, {"a": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            write_rag_jsonl(path, records)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line) for line in lines], records)


if __name__ == "__main__":
    unittest.main()
