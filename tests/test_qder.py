import unittest

from trecrag.rerank.qder.entities import HeuristicEntityProvider, NullEntityProvider
from trecrag.rerank.qder.reranker import HashingQDERScorer, QDERReranker, _minmax_normalize
from trecrag.schema import Candidate, Topic


class TestMinMaxNormalize(unittest.TestCase):
    def test_scales_into_zero_one(self):
        self.assertEqual(_minmax_normalize([0.0, 5.0, 10.0]), [0.0, 0.5, 1.0])

    def test_constant_values_return_neutral_half(self):
        self.assertEqual(_minmax_normalize([3.0, 3.0, 3.0]), [0.5, 0.5, 0.5])


class TestNullEntityProvider(unittest.TestCase):
    def test_always_returns_zero_entities(self):
        provider = NullEntityProvider()
        vectors = provider.get_entity_vectors("Barack Obama visited Paris.", max_entities=8)
        self.assertEqual(vectors.shape, (0, 300))


class TestHeuristicEntityProvider(unittest.TestCase):
    def test_extracts_capitalized_phrases(self):
        provider = HeuristicEntityProvider(dims=32)
        vectors = provider.get_entity_vectors(
            "Barack Obama met with the United Nations in New York.", max_entities=8
        )
        self.assertGreater(vectors.shape[0], 0)
        self.assertEqual(vectors.shape[1], 32)

    def test_deduplicates_repeated_mentions(self):
        provider = HeuristicEntityProvider(dims=16)
        mentions = provider._mentions("New York is beautiful. New York is old.", max_entities=8)
        self.assertEqual(mentions.count("New York"), 1)

    def test_skips_leading_stopwords(self):
        provider = HeuristicEntityProvider(dims=16)
        mentions = provider._mentions("The Eiffel Tower is tall.", max_entities=8)
        self.assertNotIn("The Eiffel Tower", mentions)
        self.assertIn("Eiffel Tower", mentions)

    def test_allows_short_acronyms(self):
        provider = HeuristicEntityProvider(dims=16)
        mentions = provider._mentions("The US and AI policy debate.", max_entities=8)
        self.assertIn("US", mentions)
        self.assertIn("AI", mentions)

    def test_drops_bare_sentence_initial_capitalized_word(self):
        provider = HeuristicEntityProvider(dims=16)
        mentions = provider._mentions("Congestion pricing helps cities.", max_entities=8)
        self.assertNotIn("Congestion", mentions)

    def test_respects_max_entities(self):
        provider = HeuristicEntityProvider(dims=16)
        text = "Alpha Beta. Gamma Delta. Epsilon Zeta. Eta Theta."
        mentions = provider._mentions(text, max_entities=2)
        self.assertLessEqual(len(mentions), 2)

    def test_deterministic_vectors_are_unit_norm(self):
        provider = HeuristicEntityProvider(dims=64)
        v1 = provider.get_entity_vectors("Barack Obama spoke today.", max_entities=4)
        v2 = provider.get_entity_vectors("Barack Obama spoke today.", max_entities=4)
        self.assertTrue((v1 == v2).all())
        norms = (v1**2).sum(axis=1) ** 0.5
        for norm in norms:
            self.assertAlmostEqual(float(norm), 1.0, places=5)

    def test_no_entities_returns_empty_array(self):
        provider = HeuristicEntityProvider(dims=16)
        vectors = provider.get_entity_vectors("lowercase text with no entities here", max_entities=8)
        self.assertEqual(vectors.shape, (0, 16))


class TestHashingQDERScorer(unittest.TestCase):
    def test_relevant_document_scores_higher(self):
        scorer = HashingQDERScorer(max_query_tokens=16, max_doc_tokens=64)
        query = "congestion pricing transit funding accountability"
        on_topic = "Congestion pricing provides dedicated transit funding with strong accountability."
        off_topic = "A recipe for banana bread requires flour, sugar, and ripe bananas."
        scores = scorer.score_batch(query, [off_topic, on_topic], doc_scores=[0.5, 0.5])
        self.assertGreater(scores[1], scores[0])

    def test_score_batch_matches_manual_scoring(self):
        scorer = HashingQDERScorer(max_query_tokens=8, max_doc_tokens=32)
        query = "climate policy"
        docs = ["climate policy debate", "unrelated text about sports"]
        batch_scores = scorer.score_batch(query, docs, doc_scores=[1.0, 1.0])
        manual_scores = [scorer._score_one(query, doc, 1.0) for doc in docs]
        self.assertEqual(batch_scores, manual_scores)

    def test_doc_score_scales_result(self):
        scorer = HashingQDERScorer(max_query_tokens=8, max_doc_tokens=32)
        query = "climate policy"
        doc = "climate policy debate today"
        low = scorer._score_one(query, doc, doc_score=0.1)
        high = scorer._score_one(query, doc, doc_score=1.0)
        self.assertGreater(high, low)

    def test_entities_enabled_changes_score(self):
        query = "Barack Obama policy speech"
        doc = "Barack Obama gave a policy speech in Washington."
        no_entities = HashingQDERScorer(entity_provider=NullEntityProvider())
        with_entities = HashingQDERScorer(entity_provider=HeuristicEntityProvider())
        score_without = no_entities._score_one(query, doc, 0.5)
        score_with = with_entities._score_one(query, doc, 0.5)
        self.assertNotEqual(score_without, score_with)


class TestQDERReranker(unittest.TestCase):
    def _candidates_equal_bm25(self):
        # Equal bm25_score => QDER's own doc-score scaling factor (derived
        # from bm25_score, see §2.4) normalizes to a neutral 0.5 for both, so
        # these tests isolate the text channel's own semantic discrimination.
        return [
            Candidate(
                docid="off",
                score=0.0,
                rank=1,
                text="A recipe for banana bread requires flour, sugar, and ripe bananas.",
                features={"bm25_score": 1.0},
            ),
            Candidate(
                docid="on",
                score=0.0,
                rank=2,
                text="Congestion pricing provides dedicated transit funding with strong accountability.",
                features={"bm25_score": 1.0},
            ),
        ]

    def _candidates_differing_bm25(self):
        # bm25_score deliberately favors the lexically off-topic doc, so
        # lambda=1 (pure bm25) and lambda=0 (pure qder) disagree on ranking.
        candidates = self._candidates_equal_bm25()
        candidates[0].features["bm25_score"] = 5.0
        candidates[1].features["bm25_score"] = 1.0
        return candidates

    def test_pure_qder_ranks_relevant_doc_first(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding accountability")
        reranker = QDERReranker(HashingQDERScorer(), hybrid_lambda=None)
        reranked = reranker.rerank(topic, self._candidates_equal_bm25(), k=2)
        self.assertEqual(reranked[0].docid, "on")
        self.assertIn("qder_score", reranked[0].features)

    def test_hybrid_lambda_one_follows_bm25_order(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding accountability")
        reranker = QDERReranker(HashingQDERScorer(), hybrid_lambda=1.0)
        reranked = reranker.rerank(topic, self._candidates_differing_bm25(), k=2)
        # bm25_score is highest for "off" (5.0 vs 1.0), so lambda=1 (pure bm25) ranks it first.
        self.assertEqual(reranked[0].docid, "off")

    def test_hybrid_lambda_zero_follows_qder_order(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding accountability")
        reranker = QDERReranker(HashingQDERScorer(), hybrid_lambda=0.0)
        reranked = reranker.rerank(topic, self._candidates_equal_bm25(), k=2)
        self.assertEqual(reranked[0].docid, "on")

    def test_candidates_without_text_sort_last(self):
        topic = Topic(topic_id="t1", text="anything")
        candidates = [
            Candidate(docid="no_text", score=0.0, rank=1, text=None, features={"bm25_score": 1.0}),
            Candidate(docid="has_text", score=0.0, rank=2, text="anything at all", features={"bm25_score": 1.0}),
        ]
        reranker = QDERReranker(HashingQDERScorer())
        reranked = reranker.rerank(topic, candidates, k=2)
        self.assertEqual(reranked[-1].docid, "no_text")

    def test_empty_input(self):
        reranker = QDERReranker(HashingQDERScorer())
        self.assertEqual(reranker.rerank(Topic("t1", "q"), [], k=5), [])

    def test_truncates_to_k(self):
        topic = Topic(topic_id="t1", text="query text")
        candidates = [
            Candidate(docid=str(i), score=0.0, rank=i, text=f"document number {i}", features={"bm25_score": float(i)})
            for i in range(1, 11)
        ]
        reranker = QDERReranker(HashingQDERScorer(), hybrid_lambda=None)
        reranked = reranker.rerank(topic, candidates, k=4)
        self.assertEqual(len(reranked), 4)
        self.assertEqual([c.rank for c in reranked], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
