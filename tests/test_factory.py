import unittest

from trecrag.factory import _validate_output_depth, build_qder
from trecrag.pipeline import StageDepths
from trecrag.rerank.qder import HashingQDERScorer
from trecrag.rerank.qder.entities import HeuristicEntityProvider, NullEntityProvider


def _depths(**overrides):
    defaults = dict(colbert_top_k=100, qder_top_k=100, ce_top_k=100, llm_listwise_top_k=100)
    defaults.update(overrides)
    return StageDepths(**defaults)


class TestValidateOutputDepth(unittest.TestCase):
    def test_noop_when_depth_not_configured(self):
        _validate_output_depth(_depths(), None)  # must not raise

    def test_passes_when_all_stages_sufficient(self):
        _validate_output_depth(_depths(), 100)  # must not raise

    def test_raises_when_llm_listwise_top_k_too_small(self):
        with self.assertRaises(ValueError):
            _validate_output_depth(_depths(llm_listwise_top_k=20), 100)

    def test_raises_when_cross_encoder_top_k_too_small(self):
        with self.assertRaises(ValueError):
            _validate_output_depth(_depths(ce_top_k=50), 100)

    def test_raises_when_qder_top_k_too_small(self):
        with self.assertRaises(ValueError):
            _validate_output_depth(_depths(qder_top_k=80), 100)

    def test_raises_when_colbert_top_k_too_small(self):
        with self.assertRaises(ValueError):
            _validate_output_depth(_depths(colbert_top_k=50), 100)


class TestBuildQDER(unittest.TestCase):
    def test_raises_when_use_entities_true_and_entity_provider_missing(self):
        with self.assertRaises(ValueError):
            build_qder({"use_entities": True})

    def test_raises_when_use_entities_true_and_entity_provider_unknown(self):
        with self.assertRaises(ValueError):
            build_qder({"use_entities": True, "entity_provider": "bogus"})

    def test_uses_heuristic_provider_when_configured(self):
        reranker = build_qder({"use_entities": True, "entity_provider": "heuristic"})
        self.assertIsInstance(reranker.scorer, HashingQDERScorer)
        self.assertIsInstance(reranker.scorer.entity_provider, HeuristicEntityProvider)

    def test_defaults_to_null_provider_when_use_entities_false(self):
        reranker = build_qder({})
        self.assertIsInstance(reranker.scorer.entity_provider, NullEntityProvider)


if __name__ == "__main__":
    unittest.main()
