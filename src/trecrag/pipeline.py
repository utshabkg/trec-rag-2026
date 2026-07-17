"""Orchestrates the full cascade:

    (BM25 + Biencoder) -> Hybrid RRF fusion -> LTR -> ColBERT -> CE
        -> LLM listwise -> final ranked list -> Generator

Every stage after BM25 is optional (`None` skips it, simply truncating to
that stage's configured depth) so partial pipelines can be assembled and
tested independently of the full cascade.
"""

from __future__ import annotations

from dataclasses import dataclass

from trecrag.generation.generator import Generator
from trecrag.rerank.base import Reranker
from trecrag.retrieval.base import Retriever
from trecrag.retrieval.biencoder import BiencoderRetriever
from trecrag.retrieval.hybrid import reciprocal_rank_fusion
from trecrag.schema import Candidate, RagAnswer, Topic


@dataclass
class StageDepths:
    bm25_hits: int = 1000
    hybrid_top_k: int = 500
    ltr_top_k: int = 200
    colbert_top_k: int = 100
    ce_top_k: int = 50
    llm_listwise_top_k: int = 20
    rrf_k: int = 60


@dataclass
class GenerationSettings:
    evidence_k: int = 10
    max_words: int = 1024
    max_citations_per_sentence: int = 3


class RagPipeline:
    def __init__(
        self,
        bm25: Retriever,
        depths: StageDepths,
        generation: GenerationSettings,
        generator: Generator,
        biencoder: BiencoderRetriever | None = None,
        ltr: Reranker | None = None,
        colbert: Reranker | None = None,
        cross_encoder: Reranker | None = None,
        llm_listwise: Reranker | None = None,
    ):
        self.bm25 = bm25
        self.biencoder = biencoder
        self.ltr = ltr
        self.colbert = colbert
        self.cross_encoder = cross_encoder
        self.llm_listwise = llm_listwise
        self.generator = generator
        self.depths = depths
        self.generation = generation

    def retrieve(self, topic: Topic) -> list[Candidate]:
        bm25_candidates = self.bm25.retrieve(topic, self.depths.bm25_hits)

        ranked_lists = [bm25_candidates]
        if self.biencoder is not None:
            biencoder_candidates = self.biencoder.rescore(
                topic, bm25_candidates, self.depths.bm25_hits
            )
            if biencoder_candidates:
                ranked_lists.append(biencoder_candidates)

        candidates = reciprocal_rank_fusion(
            ranked_lists, k_const=self.depths.rrf_k, top_k=self.depths.hybrid_top_k
        )

        if self.ltr is not None:
            candidates = self.ltr.rerank(topic, candidates, self.depths.ltr_top_k)
        else:
            candidates = candidates[: self.depths.ltr_top_k]

        if self.colbert is not None:
            candidates = self.colbert.rerank(topic, candidates, self.depths.colbert_top_k)
        else:
            candidates = candidates[: self.depths.colbert_top_k]

        if self.cross_encoder is not None:
            candidates = self.cross_encoder.rerank(topic, candidates, self.depths.ce_top_k)
        else:
            candidates = candidates[: self.depths.ce_top_k]

        if self.llm_listwise is not None:
            candidates = self.llm_listwise.rerank(topic, candidates, self.depths.llm_listwise_top_k)
        else:
            candidates = candidates[: self.depths.llm_listwise_top_k]

        return candidates

    def run_topic(self, topic: Topic) -> tuple[list[Candidate], RagAnswer]:
        final_ranked = self.retrieve(topic)
        evidence = final_ranked[: self.generation.evidence_k]
        rag_answer = self.generator.generate(
            topic,
            evidence,
            max_words=self.generation.max_words,
            max_citations=self.generation.max_citations_per_sentence,
        )
        return final_ranked, rag_answer
