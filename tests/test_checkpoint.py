"""Checkpoint round-trip tests.

The checkpoint store is what makes a multi-hour run survivable, so its failure
modes are expensive: a save that raises loses the topic, and a load that returns
subtly different candidates corrupts the submission silently.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trecrag.output.checkpoint import TopicCheckpointStore
from trecrag.schema import AnswerSentence, Candidate, RagAnswer


def _answer() -> RagAnswer:
    return RagAnswer(
        references=["shard_00459_61697", "shard_01012_88420"],
        sentences=[
            AnswerSentence(text="First sentence.", citations=[0]),
            AnswerSentence(text="Second sentence.", citations=[0, 1]),
        ],
    )


class TestTopicCheckpointStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = TopicCheckpointStore(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_preserves_ranking_and_answer(self):
        ranked = [
            Candidate(docid="a", score=9.5, rank=1, features={"ce_score": 9.5}),
            Candidate(docid="b", score=8.25, rank=2, features={"ce_score": 8.25}),
        ]
        self.store.save("rag2026-0", ranked, _answer())

        loaded_ranked, loaded_answer = self.store.load("rag2026-0")
        self.assertEqual([c.docid for c in loaded_ranked], ["a", "b"])
        self.assertEqual([c.rank for c in loaded_ranked], [1, 2])
        self.assertEqual([c.score for c in loaded_ranked], [9.5, 8.25])
        self.assertEqual(loaded_answer.references, _answer().references)
        self.assertEqual([s.citations for s in loaded_answer.sentences], [[0], [0, 1]])

    def test_numpy_scores_are_serializable(self):
        """The neural stages emit np.float32, which json.dumps rejects outright."""
        ranked = [
            Candidate(
                docid="a",
                score=np.float32(3.5),
                rank=np.int64(1),
                features={"ce_score": np.float32(3.5), "rrf_score": np.float64(0.1)},
            )
        ]
        self.store.save("rag2026-1", ranked, _answer())
        loaded_ranked, _ = self.store.load("rag2026-1")
        self.assertEqual(loaded_ranked[0].score, 3.5)
        self.assertIsInstance(loaded_ranked[0].score, float)

    def test_has_reflects_saved_state(self):
        self.assertFalse(self.store.has("rag2026-2"))
        self.store.save("rag2026-2", [], _answer())
        self.assertTrue(self.store.has("rag2026-2"))

    def test_missing_topic_loads_as_none(self):
        self.assertIsNone(self.store.load("rag2026-404"))

    def test_corrupt_checkpoint_loads_as_none_rather_than_raising(self):
        """A truncated file from a killed process must re-run, not crash the run."""
        self.store.save("rag2026-3", [], _answer())
        Path(self._tmp.name, "rag2026-3.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.store.load("rag2026-3"))

    def test_save_leaves_no_temp_files_behind(self):
        self.store.save("rag2026-4", [], _answer())
        self.assertEqual(
            sorted(p.name for p in Path(self._tmp.name).iterdir()), ["rag2026-4.json"]
        )

    def test_saved_payload_is_valid_json(self):
        self.store.save("rag2026-5", [Candidate(docid="a", score=1.0, rank=1)], _answer())
        payload = json.loads(Path(self._tmp.name, "rag2026-5.json").read_text())
        self.assertEqual(payload["topic_id"], "rag2026-5")


if __name__ == "__main__":
    unittest.main()
