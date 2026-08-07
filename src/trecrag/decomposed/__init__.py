"""Per-AIN retrieval: fan the cascade out over atomic information needs.

The `scripts/decompose/` preprocessors turn each composite narrative into
atomic information needs and reformulated queries. This package is what finally
consumes that output: it runs the existing cascade once per need, fuses the
per-need runs (plus the composite run) into one pool, and hands that pool to the
existing generator alongside the original composite question.
"""

from trecrag.decomposed.evidence import effective_evidence_k, select_evidence
from trecrag.decomposed.fusion import AinRun, fuse_ain_runs, sub_question_feature
from trecrag.decomposed.loader import (
    DecompositionFormatError,
    attach,
    load_decomposed_topics,
)
from trecrag.decomposed.orchestrator import DecomposedPipeline, FanoutSettings
from trecrag.decomposed.schema import AtomicNeed, DecomposedTopic, SubQuestion

__all__ = [
    "AinRun",
    "AtomicNeed",
    "DecomposedPipeline",
    "DecomposedTopic",
    "DecompositionFormatError",
    "FanoutSettings",
    "SubQuestion",
    "attach",
    "effective_evidence_k",
    "fuse_ain_runs",
    "load_decomposed_topics",
    "select_evidence",
    "sub_question_feature",
]
