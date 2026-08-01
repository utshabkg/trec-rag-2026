import unittest

from trecrag.retrieval.biencoder import BiencoderRetriever, HashingEncoder
from trecrag.schema import Candidate, Topic


class TestHashingEncoder(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        encoder = HashingEncoder(dims=64)
        vec1 = encoder.encode(["hello world"])[0]
        vec2 = encoder.encode(["hello world"])[0]
        self.assertTrue((vec1 == vec2).all())
        self.assertAlmostEqual(float((vec1**2).sum()) ** 0.5, 1.0, places=5)

    def test_identical_text_has_similarity_one(self):
        encoder = HashingEncoder(dims=64)
        vecs = encoder.encode(["climate policy debate", "climate policy debate"])
        self.assertAlmostEqual(float(vecs[0] @ vecs[1]), 1.0, places=5)


class TestBiencoderRetriever(unittest.TestCase):
    def test_rescore_favors_lexical_overlap_with_query(self):
        topic = Topic(topic_id="t1", text="congestion pricing transit funding")
        pool = [
            Candidate(docid="off_topic", score=10.0, rank=1, text="unrelated cooking recipe content"),
            Candidate(
                docid="on_topic",
                score=5.0,
                rank=2,
                text="congestion pricing transit funding accountability",
            ),
        ]
        biencoder = BiencoderRetriever(HashingEncoder(dims=256))
        reranked = biencoder.rescore(topic, pool, k=2)
        self.assertEqual(reranked[0].docid, "on_topic")
        self.assertIn("bien_score", reranked[0].features)
        self.assertEqual(reranked[0].rank, 1)

    def test_drops_candidates_without_text(self):
        topic = Topic(topic_id="t1", text="query text")
        pool = [Candidate(docid="no_text", score=1.0, rank=1, text=None)]
        biencoder = BiencoderRetriever(HashingEncoder(dims=32))
        self.assertEqual(biencoder.rescore(topic, pool, k=5), [])

    def test_retrieve_without_pool_retriever_raises(self):
        biencoder = BiencoderRetriever(HashingEncoder(dims=32))
        with self.assertRaises(ValueError):
            biencoder.retrieve(Topic(topic_id="t1", text="q"), 10)


if __name__ == "__main__":
    unittest.main()
