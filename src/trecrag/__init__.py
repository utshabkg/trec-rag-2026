"""TREC RAG 2026 submission pipeline.

Stage order: (BM25 + Biencoder) -> Hybrid fusion -> LTR -> ColBERT -> CE -> LLM listwise -> Generator.
"""

__version__ = "0.1.0"
