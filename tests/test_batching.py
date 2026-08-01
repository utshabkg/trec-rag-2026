"""Batched backends must stay observationally identical to the per-item path.

The model-backed encoders/scorers override the batch hooks for GPU throughput.
If a batched override ever diverges from its single-item counterpart (a padding
mask left in, a truncation applied on one path only) the pipeline still runs and
still validates — it just silently ranks worse. These tests pin the equivalence
at the interface level, using the dependency-free backends so they stay offline.
"""

import unittest

from trecrag.clients.llm_client import strip_reasoning
from trecrag.rerank.colbert import ColBERTReranker, HashingTokenEncoder
from trecrag.rerank.cross_encoder import (
    CrossEncoderReranker,
    LexicalOverlapScorer,
    PairScorer,
)
from trecrag.retrieval.biencoder import HashingEncoder
from trecrag.schema import Candidate, Topic


class CountingScorer(PairScorer):
    """Uses the default `score_batch`, and records how it was driven."""

    def __init__(self):
        self.calls = 0

    def score(self, query: str, passage: str) -> float:
        self.calls += 1
        return float(len(passage))


class TestPairScorerBatching(unittest.TestCase):
    def test_default_score_batch_matches_individual_scores(self):
        scorer = LexicalOverlapScorer()
        pairs = [("congestion pricing", "congestion pricing transit"), ("a b", "c d")]
        self.assertEqual(
            scorer.score_batch(pairs), [scorer.score(q, p) for q, p in pairs]
        )

    def test_empty_batch(self):
        self.assertEqual(LexicalOverlapScorer().score_batch([]), [])

    def test_reranker_scores_every_candidate_with_text_exactly_once(self):
        scorer = CountingScorer()
        candidates = [
            Candidate(docid="a", score=0.0, rank=1, text="short"),
            Candidate(docid="b", score=0.0, rank=2, text="a much longer passage"),
        ]
        CrossEncoderReranker(scorer).rerank(Topic("t1", "q"), candidates, k=2)
        self.assertEqual(scorer.calls, 2)

    def test_reranker_keeps_textless_candidates_last(self):
        candidates = [
            Candidate(docid="no_text", score=0.0, rank=1, text=None),
            Candidate(docid="has_text", score=0.0, rank=2, text="q matching text"),
        ]
        reranked = CrossEncoderReranker(LexicalOverlapScorer()).rerank(
            Topic("t1", "q"), candidates, k=2
        )
        self.assertEqual(reranked[-1].docid, "no_text")


class TestTokenEncoderBatching(unittest.TestCase):
    def test_default_encode_tokens_batch_matches_individual(self):
        encoder = HashingTokenEncoder(dims=64)
        texts = ["congestion pricing", "", "nursing retention plan"]
        batched = encoder.encode_tokens_batch(texts, max_tokens=16)
        for got, text in zip(batched, texts):
            self.assertTrue((got == encoder.encode_tokens(text, 16)).all())

    def test_colbert_rerank_unaffected_by_textless_candidates(self):
        candidates = [
            Candidate(docid="no_text", score=0.0, rank=1, text=None),
            Candidate(docid="on", score=0.0, rank=2, text="congestion pricing transit"),
            Candidate(docid="off", score=0.0, rank=3, text="unrelated cooking recipe"),
        ]
        reranked = ColBERTReranker(HashingTokenEncoder(dims=256)).rerank(
            Topic("t1", "congestion pricing transit"), candidates, k=3
        )
        self.assertEqual(reranked[0].docid, "on")
        self.assertEqual(reranked[-1].docid, "no_text")


class TestEncoderQueryPassageHooks(unittest.TestCase):
    def test_fallback_encoder_treats_queries_and_passages_alike(self):
        encoder = HashingEncoder(dims=64)
        text = ["congestion pricing"]
        self.assertTrue((encoder.encode_queries(text) == encoder.encode(text)).all())
        self.assertTrue((encoder.encode_passages(text) == encoder.encode(text)).all())


class TestStripReasoning(unittest.TestCase):
    def test_removes_think_block_that_would_corrupt_rank_parsing(self):
        raw = "<think>maybe 5 then 9 then 12</think>[2] > [1] > [3]"
        self.assertEqual(strip_reasoning(raw), "[2] > [1] > [3]")

    def test_removes_multiline_and_alternate_tags(self):
        self.assertEqual(strip_reasoning("<reasoning>\na\nb\n</reasoning>ok"), "ok")
        self.assertEqual(strip_reasoning("<thinking>x</thinking> ok"), "ok")

    def test_leaves_ordinary_text_untouched(self):
        self.assertEqual(strip_reasoning('[{"text": "a"}]'), '[{"text": "a"}]')


if __name__ == "__main__":
    unittest.main()
