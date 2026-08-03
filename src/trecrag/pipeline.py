"""Orchestrates the full cascade:

    (BM25 + Biencoder) -> Hybrid RRF fusion -> LTR -> ColBERT -> CE
        -> LLM listwise -> final ranked list -> Generator

Every stage after BM25 is optional (`None` skips it, simply truncating to
that stage's configured depth) so partial pipelines can be assembled and
tested independently of the full cascade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from trecrag.chunking import PassageSelector
from trecrag.diagnostics import StageDiagnostics
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
        passage_selector: PassageSelector | None = None,
        diagnostics: StageDiagnostics | None = None,
    ):
        self.diagnostics = diagnostics or StageDiagnostics(None)
        self.bm25 = bm25
        self.passage_selector = passage_selector
        self.biencoder = biencoder
        self.ltr = ltr
        self.colbert = colbert
        self.cross_encoder = cross_encoder
        self.llm_listwise = llm_listwise
        self.generator = generator
        self.depths = depths
        self.generation = generation

    def retrieve(self, topic: Topic) -> list[Candidate]:
        # Every stage is timed so the latency/quality tradeoff can be reported
        # per stage rather than as one opaque wall-clock number.
        started = time.perf_counter()
        bm25_candidates = self.bm25.retrieve(topic, self.depths.bm25_hits)
        self.diagnostics.record(
            topic.topic_id, "bm25", bm25_candidates, bm25_candidates,
            elapsed_s=time.perf_counter() - started,
        )

        # ClimbMix returns whole documents (median ~43k chars). Narrow each to its
        # query-relevant span once, here, so every later stage scores and cites
        # that span instead of independently truncating to the document's opening.
        if self.passage_selector is not None:
            started = time.perf_counter()
            before = bm25_candidates
            bm25_candidates = self.passage_selector.apply(topic.text, bm25_candidates)
            self.diagnostics.record(
                topic.topic_id, "passage_selection", before, bm25_candidates,
                elapsed_s=time.perf_counter() - started,
            )

        ranked_lists = [bm25_candidates]
        if self.biencoder is not None:
            started = time.perf_counter()
            biencoder_candidates = self.biencoder.rescore(
                topic, bm25_candidates, self.depths.bm25_hits
            )
            self.diagnostics.record(
                topic.topic_id, "biencoder", bm25_candidates, biencoder_candidates,
                elapsed_s=time.perf_counter() - started,
            )
            if biencoder_candidates:
                ranked_lists.append(biencoder_candidates)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            ranked_lists, k_const=self.depths.rrf_k, top_k=self.depths.hybrid_top_k
        )
        self.diagnostics.record(
            topic.topic_id, "rrf", bm25_candidates, fused,
            elapsed_s=time.perf_counter() - started,
        )
        candidates = fused

        for name, reranker, depth in (
            ("ltr", self.ltr, self.depths.ltr_top_k),
            ("colbert", self.colbert, self.depths.colbert_top_k),
            ("cross_encoder", self.cross_encoder, self.depths.ce_top_k),
            ("llm_listwise", self.llm_listwise, self.depths.llm_listwise_top_k),
        ):
            started = time.perf_counter()
            before = candidates
            if reranker is not None:
                candidates = reranker.rerank(topic, candidates, depth)
            else:
                # A disabled stage still truncates to its configured depth, which
                # is easy to forget and silently caps everything downstream.
                candidates = candidates[:depth]
            self.diagnostics.record(
                topic.topic_id, name, before, candidates,
                elapsed_s=time.perf_counter() - started,
            )

        return candidates

    def run_topic(self, topic: Topic) -> tuple[list[Candidate], RagAnswer]:
        final_ranked = self.retrieve(topic)
        evidence = final_ranked[: self.generation.evidence_k]
        started = time.perf_counter()
        rag_answer = self.generator.generate(
            topic,
            evidence,
            max_words=self.generation.max_words,
            max_citations=self.generation.max_citations_per_sentence,
        )
        self.diagnostics.record(
            topic.topic_id, "generation", evidence, evidence,
            elapsed_s=time.perf_counter() - started,
        )
        return final_ranked, rag_answer
