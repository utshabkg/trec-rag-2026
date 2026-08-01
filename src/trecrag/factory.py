"""Builds a `RagPipeline` from a loaded `configs/pipeline.yaml` dict.

This is the only module that reads config `backend:` switches and env-var
names and turns them into concrete objects; every other module works purely
in terms of the `Retriever` / `Reranker` / `Generator` interfaces.
"""

from __future__ import annotations

from typing import Any

from trecrag.chunking import PassageSelector
from trecrag.clients.llm_client import AnthropicClient, IdentityLLMClient, LLMClient, OpenAICompatibleClient
from trecrag.clients.pyserini_client import PyseriniClient
from trecrag.config import get_env, require_env
from trecrag.generation.generator import ExtractiveGenerator, Generator, LLMGenerator
from trecrag.pipeline import GenerationSettings, RagPipeline, StageDepths
from trecrag.rerank.colbert import ColBERTReranker, HashingTokenEncoder, TransformerTokenEncoder
from trecrag.rerank.fusion import FeatureFusionReranker
from trecrag.rerank.cross_encoder import (
    CrossEncoderReranker,
    LexicalOverlapScorer,
    SentenceTransformersCrossEncoderScorer,
)
from trecrag.rerank.llm_listwise import LLMListwiseReranker
from trecrag.rerank.ltr import LTRReranker
from trecrag.rerank.qder import (
    EntityEmbeddingProvider,
    HashingQDERScorer,
    HeuristicEntityProvider,
    NullEntityProvider,
    QDERReranker,
    QDERScorer,
    TransformerQDERScorer,
)
from trecrag.retrieval.biencoder import BiencoderRetriever, Encoder, HashingEncoder, SentenceTransformerEncoder
from trecrag.retrieval.bm25 import BM25Retriever


def build_llm_client(
    backend: str,
    llm_config: dict[str, Any],
    model_env_override: str | None = None,
) -> LLMClient:
    if backend == "identity":
        return IdentityLLMClient()

    api_key = require_env(llm_config["api_key_env"], hint="LLM API key")
    model = require_env(model_env_override or llm_config["model_env"], hint="LLM model name")
    timeout_seconds = float(llm_config.get("timeout_seconds", 60))

    if backend == "openai":
        base_url = require_env(llm_config["base_url_env"], hint="LLM base URL")
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_attempts=int(llm_config.get("max_attempts", 3)),
            retry_backoff=float(llm_config.get("retry_backoff_seconds", 5.0)),
        )
    if backend == "anthropic":
        return AnthropicClient(api_key=api_key, model=model, timeout_seconds=timeout_seconds)
    raise ValueError(f"unknown LLM backend {backend!r}")


def build_pyserini_client(config: dict[str, Any]) -> PyseriniClient:
    pyserini_config = config["pyserini"]
    return PyseriniClient(
        base_url=pyserini_config["base_url"],
        index=pyserini_config["index"],
        token=get_env(pyserini_config.get("token_env")),
        timeout_seconds=float(pyserini_config.get("timeout_seconds", 20)),
        max_retries=int(pyserini_config.get("max_retries", 3)),
        cache_dir=pyserini_config.get("cache_dir"),
        backoff_factor=float(pyserini_config.get("backoff_factor", 0.5)),
        rate_limit_pause_seconds=float(
            pyserini_config.get("rate_limit_pause_seconds", 60.0)
        ),
    )


def build_passage_selector(cfg: dict[str, Any]) -> PassageSelector | None:
    if not cfg.get("enabled", False):
        return None
    return PassageSelector(
        window_chars=int(cfg.get("window_chars", 2000)),
        stride_chars=int(cfg.get("stride_chars", 1000)),
        windows_per_doc=int(cfg.get("windows_per_doc", 2)),
        min_doc_chars=int(cfg.get("min_doc_chars", 2000)),
    )


def build_biencoder(biencoder_cfg: dict[str, Any]) -> BiencoderRetriever | None:
    if not biencoder_cfg.get("enabled", True):
        return None
    encoder: Encoder
    if biencoder_cfg.get("backend", "hashing") == "sentence_transformer":
        encoder = SentenceTransformerEncoder(
            biencoder_cfg["model_name"],
            device=biencoder_cfg.get("device"),
            batch_size=int(biencoder_cfg.get("batch_size", 128)),
            query_prefix=biencoder_cfg.get("query_prefix", ""),
            passage_prefix=biencoder_cfg.get("passage_prefix", ""),
        )
    else:
        encoder = HashingEncoder(dims=int(biencoder_cfg.get("dims", 768)))
    return BiencoderRetriever(encoder, pool_size=int(biencoder_cfg.get("pool_size", 1000)))


def build_ltr(ltr_cfg: dict[str, Any]) -> LTRReranker | None:
    if not ltr_cfg.get("enabled", True):
        return None
    return LTRReranker(
        feature_names=ltr_cfg["feature_names"],
        weights=ltr_cfg.get("weights"),
        model_path=ltr_cfg.get("model_path"),
    )


def build_colbert(colbert_cfg: dict[str, Any]) -> ColBERTReranker | None:
    if not colbert_cfg.get("enabled", True):
        return None
    if colbert_cfg.get("backend", "hashing") == "transformer":
        token_encoder = TransformerTokenEncoder(
            colbert_cfg["model_name"],
            device=colbert_cfg.get("device"),
            batch_size=int(colbert_cfg.get("batch_size", 32)),
        )
    else:
        token_encoder = HashingTokenEncoder()
    return ColBERTReranker(
        token_encoder,
        max_query_tokens=int(colbert_cfg.get("max_query_tokens", 32)),
        max_doc_tokens=int(colbert_cfg.get("max_doc_tokens", 180)),
    )


def build_qder(qder_cfg: dict[str, Any]) -> QDERReranker | None:
    if not qder_cfg.get("enabled", True):
        return None

    use_entities = bool(qder_cfg.get("use_entities", False))
    entity_provider_name = qder_cfg.get("entity_provider", "none")
    entity_provider: EntityEmbeddingProvider
    if use_entities:
        if entity_provider_name == "heuristic":
            entity_provider = HeuristicEntityProvider()
        else:
            raise ValueError(
                f"stages.qder.use_entities is true but stages.qder.entity_provider "
                f"is {entity_provider_name!r} — this would build QDERModel with an "
                "active entity channel that only ever receives zero placeholder "
                "vectors. Set entity_provider: heuristic, or use_entities: false."
            )
    else:
        entity_provider = NullEntityProvider()

    scorer: QDERScorer
    if qder_cfg.get("backend", "hashing") == "transformer":
        scorer = TransformerQDERScorer(
            model_name=qder_cfg["model_name"],
            use_entities=use_entities,
            entity_provider=entity_provider,
            score_method=qder_cfg.get("score_method", "bilinear"),
            model_path=qder_cfg.get("model_path"),
            max_query_tokens=int(qder_cfg.get("max_query_tokens", 32)),
            max_doc_tokens=int(qder_cfg.get("max_doc_tokens", 256)),
        )
    else:
        scorer = HashingQDERScorer(
            entity_provider=entity_provider,
            max_query_tokens=int(qder_cfg.get("max_query_tokens", 32)),
            max_doc_tokens=int(qder_cfg.get("max_doc_tokens", 256)),
        )

    hybrid_lambda = qder_cfg.get("hybrid_lambda", 0.5)
    return QDERReranker(
        scorer,
        bm25_feature_name=qder_cfg.get("bm25_feature_name", "bm25_score"),
        hybrid_lambda=None if hybrid_lambda is None else float(hybrid_lambda),
    )


def build_cross_encoder(ce_cfg: dict[str, Any]) -> CrossEncoderReranker | None:
    if not ce_cfg.get("enabled", True):
        return None
    if ce_cfg.get("backend", "lexical") == "sentence_transformer":
        scorer = SentenceTransformersCrossEncoderScorer(
            ce_cfg["model_name"],
            device=ce_cfg.get("device"),
            batch_size=int(ce_cfg.get("batch_size", 64)),
        )
    else:
        scorer = LexicalOverlapScorer()
    return CrossEncoderReranker(scorer)


def build_fusion(fusion_cfg: dict[str, Any]) -> FeatureFusionReranker | None:
    if not fusion_cfg.get("enabled", False):
        return None
    return FeatureFusionReranker(
        feature_weights=fusion_cfg.get("feature_weights", {}),
        k_rrf=int(fusion_cfg.get("k_rrf", 60)),
        strict=bool(fusion_cfg.get("strict", True)),
    )


def build_llm_listwise(listwise_cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> LLMListwiseReranker | None:
    if not listwise_cfg.get("enabled", True):
        return None
    llm = build_llm_client(
        listwise_cfg.get("backend", "identity"), llm_cfg, model_env_override=listwise_cfg.get("model_env")
    )
    return LLMListwiseReranker(
        llm,
        window_size=int(listwise_cfg.get("window_size", 20)),
        step=int(listwise_cfg.get("step", 10)),
        max_passage_chars=int(listwise_cfg.get("max_passage_chars", 300)),
    )


def build_generator(generation_cfg: dict[str, Any], llm_cfg: dict[str, Any]) -> Generator:
    if generation_cfg.get("backend", "extractive") == "llm":
        llm = build_llm_client(generation_cfg.get("llm_backend", "openai"), llm_cfg)
        return LLMGenerator(llm)
    return ExtractiveGenerator()


def _validate_output_depth(depths: StageDepths, output_depth: int | None) -> None:
    """`retrieval_output.depth` is only ever honoured if every stage between the
    cascade's end and the final rerank still carries at least that many
    candidates — `RagPipeline.retrieve()` truncates to each stage's `top_k` in
    sequence, so a smaller downstream `top_k` silently caps the submitted
    depth regardless of what this config value says. Fail at build time
    instead of silently shipping a shorter run."""
    if output_depth is None:
        return
    bottlenecks = {
        "stages.colbert.top_k": depths.colbert_top_k,
        "stages.qder.top_k": depths.qder_top_k,
        "stages.cross_encoder.top_k": depths.ce_top_k,
        "stages.llm_listwise.top_k": depths.llm_listwise_top_k,
    }
    too_small = {name: v for name, v in bottlenecks.items() if v < output_depth}
    if too_small:
        raise ValueError(
            f"retrieval_output.depth ({output_depth}) exceeds these stage top_k "
            f"values, so RagPipeline.retrieve() can never return that many "
            f"candidates: {too_small}. Raise each to at least {output_depth}, "
            "or lower retrieval_output.depth."
        )


def build_pipeline_from_config(config: dict[str, Any]) -> RagPipeline:
    stages = config["stages"]
    bm25_cfg = stages.get("bm25", {})
    hybrid_cfg = stages.get("hybrid", {})
    ltr_cfg = stages.get("ltr", {})
    colbert_cfg = stages.get("colbert", {})
    qder_cfg = stages.get("qder", {})
    ce_cfg = stages.get("cross_encoder", {})
    fusion_cfg = stages.get("fusion", {})
    listwise_cfg = stages.get("llm_listwise", {})
    generation_cfg = config["generation"]
    llm_cfg = config["llm"]

    bm25 = BM25Retriever(build_pyserini_client(config))

    depths = StageDepths(
        bm25_hits=int(bm25_cfg.get("hits", 1000)),
        hybrid_top_k=int(hybrid_cfg.get("top_k", 500)),
        rrf_k=int(hybrid_cfg.get("rrf_k", 60)),
        ltr_top_k=int(ltr_cfg.get("top_k", 200)),
        colbert_top_k=int(colbert_cfg.get("top_k", 100)),
        qder_top_k=int(qder_cfg.get("top_k", 80)),
        ce_top_k=int(ce_cfg.get("top_k", 50)),
        fusion_top_k=int(fusion_cfg.get("top_k", 50)),
        llm_listwise_top_k=int(listwise_cfg.get("top_k", 20)),
    )
    generation = GenerationSettings(
        evidence_k=int(generation_cfg.get("evidence_k", 10)),
        max_words=int(generation_cfg.get("max_words", 1024)),
        max_citations_per_sentence=int(generation_cfg.get("max_citations_per_sentence", 3)),
    )

    output_depth_cfg = config.get("retrieval_output", {}).get("depth")
    _validate_output_depth(depths, int(output_depth_cfg) if output_depth_cfg is not None else None)

    return RagPipeline(
        bm25=bm25,
        depths=depths,
        generation=generation,
        generator=build_generator(generation_cfg, llm_cfg),
        passage_selector=build_passage_selector(stages.get("passage_selection", {})),
        biencoder=build_biencoder(stages.get("biencoder", {})),
        ltr=build_ltr(ltr_cfg),
        colbert=build_colbert(colbert_cfg),
        qder=build_qder(qder_cfg),
        cross_encoder=build_cross_encoder(ce_cfg),
        fusion=build_fusion(fusion_cfg),
        llm_listwise=build_llm_listwise(listwise_cfg, llm_cfg),
    )
