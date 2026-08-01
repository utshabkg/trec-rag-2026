#!/usr/bin/env python3
"""Build QDER training/dev data (JSONL) for the vendored SIGIR2025-QDER
doc-ranking trainer at `third_party/qder_training/train.py`.

For each topic, reconstructs the same candidate pool QDER would see in the
live pipeline -- BM25 -> biencoder -> RRF -> LTR -> ColBERT, using whichever
of those stages `--config` has enabled -- and labels each candidate against a
qrels file, writing the vendored trainer's expected per-line JSON schema:
`query`, `query_id`, `query_ent_emb`, `doc_id`, `doc`, `doc_score`,
`doc_ent_emb`, `label`. Entity embeddings are always written as empty lists:
this project has no real entity linker for ClimbMix
(`stages.qder.use_entities: false` in `configs/pipeline.yaml`), so whatever
checkpoint this data trains must also be loaded with `use_entities: false` --
training with entities on and serving with them off will fail to load
(the model's scoring-layer shape depends on `use_entities`).

By default (`--label-mode qrels-only`), only candidates with an explicit
qrels judgment are used -- matching `scripts/train_ltr.py`'s own convention
of dropping unjudged candidates rather than guessing at their label.
`--label-mode llm-judge` instead sends a bounded number of unjudged,
pool-ranked candidates per topic to an LLM relevance judge (UMBRELA-style 0-3
grading, reusing trecrag's existing OpenAI-compatible client) so training
data isn't capped at the sparse official qrels pool. Point it at a local
vLLM server serving whatever model you have (e.g. Qwen2.5-72B-Instruct,
gemma-3-27b-it): vLLM's `/v1/chat/completions` endpoint is OpenAI-compatible,
so no new client code is needed -- only `configs/pipeline.yaml`'s existing
`llm:` section, pointed at it via `.env.local`.

Example (qrels-only, no LLM):
    python scripts/prepare_qder_data.py \\
        --topics data/dev/topics/rag25-topics-dev.tsv \\
        --qrels data/dev/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \\
        --output-train data/qder_train.jsonl --output-dev data/qder_dev.jsonl

Example (with an LLM judge backed by a local vLLM server):
    # .env.local:
    #   LLM_BASE_URL=http://localhost:8000/v1
    #   LLM_API_KEY=not-needed
    #   QDER_JUDGE_MODEL=Qwen2.5-72B-Instruct
    python scripts/prepare_qder_data.py \\
        --topics data/dev/topics/rag25-topics-dev.tsv \\
        --qrels data/dev/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \\
        --label-mode llm-judge --llm-judge-pool-size 20 \\
        --output-train data/qder_train.jsonl --output-dev data/qder_dev.jsonl

Then train with the vendored trainer (see third_party/qder_training/README.md):
    python third_party/qder_training/train.py \\
        --train data/qder_train.jsonl --dev data/qder_dev.jsonl \\
        --qrels data/dev/rag25-dev-umbrela-qrels/rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels \\
        --save-dir output/qder --text-enc bert --score-method bilinear --use-cuda
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trecrag.config import load_dotenv_local, load_pipeline_config
from trecrag.data.topics import load_topics
from trecrag.factory import build_biencoder, build_colbert, build_llm_client, build_ltr, build_pyserini_client
from trecrag.logging_utils import configure_logging, get_logger
from trecrag.retrieval.bm25 import BM25Retriever
from trecrag.retrieval.hybrid import reciprocal_rank_fusion
from trecrag.rerank.qder.reranker import _minmax_normalize
from trecrag.schema import Candidate, Topic

logger = get_logger("prepare_qder_data")

_GRADE_RE = re.compile(r"[0-3]")

_JUDGE_PROMPT = """You are a search relevance judge. Rate how relevant the passage is to the query on a 0-3 scale:
0 = not relevant at all
1 = marginally relevant (tangentially related, does not answer the query)
2 = relevant (answers part of the query)
3 = highly relevant (directly and fully answers the query)

Query: {query}

Passage: {passage}

Respond with only a single digit: 0, 1, 2, or 3."""


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            topic_id, _iteration, docid, grade = parts
            qrels[topic_id][docid] = int(grade)
    return qrels


def split_topics(topics: list[Topic], dev_fraction: float, seed: int) -> tuple[list[Topic], list[Topic]]:
    shuffled = list(topics)
    random.Random(seed).shuffle(shuffled)
    n_dev = max(1, round(len(shuffled) * dev_fraction)) if shuffled else 0
    return shuffled[n_dev:], shuffled[:n_dev]


class JudgeCache:
    """Disk-backed (topic_id, docid) -> grade cache, appended-to as it grows
    so an interrupted run doesn't re-pay for LLM judgments already made."""

    def __init__(self, path: Path):
        self.path = path
        self._cache: dict[tuple[str, str], int] = {}
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    self._cache[(row["topic_id"], row["doc_id"])] = row["grade"]

    def get(self, topic_id: str, doc_id: str) -> int | None:
        return self._cache.get((topic_id, doc_id))

    def put(self, topic_id: str, doc_id: str, grade: int) -> None:
        self._cache[(topic_id, doc_id)] = grade
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"topic_id": topic_id, "doc_id": doc_id, "grade": grade}) + "\n")


def judge_relevance(llm, query: str, passage: str) -> int | None:
    prompt = _JUDGE_PROMPT.format(query=query, passage=passage[:2000])
    response = llm.complete(prompt, max_tokens=8)
    match = _GRADE_RE.search(response)
    return int(match.group(0)) if match else None


def build_candidate_pool(
    topic: Topic,
    bm25: BM25Retriever,
    biencoder,
    ltr,
    colbert,
    depths: dict[str, int],
) -> list[Candidate]:
    """Reproduce the exact candidate pool QDER sees in the live pipeline:
    BM25 -> biencoder -> RRF -> LTR -> ColBERT (skipping whichever optional
    stage `--config` has disabled).

    This mirrors `scripts/train_ltr.py`'s own inline cascade rather than
    reusing `RagPipeline.retrieve()`: `RagPipeline` re-applies every later
    stage's `top_k` as a truncation even when that stage is disabled (`None`
    just means "pass through, then truncate"), which would silently shrink
    this pool a second and third time via the cross-encoder's and listwise
    reranker's configured depths -- stages this script never runs.
    """
    candidates = bm25.retrieve(topic, depths["bm25_hits"])

    ranked_lists = [candidates]
    if biencoder is not None:
        biencoder_candidates = biencoder.rescore(topic, candidates, depths["bm25_hits"])
        if biencoder_candidates:
            ranked_lists.append(biencoder_candidates)

    fused = reciprocal_rank_fusion(ranked_lists, k_const=depths["rrf_k"], top_k=depths["hybrid_top_k"])

    if ltr is not None:
        fused = ltr.rerank(topic, fused, depths["ltr_top_k"])
    else:
        fused = fused[: depths["ltr_top_k"]]

    if colbert is not None:
        fused = colbert.rerank(topic, fused, depths["colbert_top_k"])
    else:
        fused = fused[: depths["colbert_top_k"]]

    return fused[: depths["pool_size"]]


def label_topic(
    topic: Topic,
    pool: list[Candidate],
    topic_qrels: dict[str, int],
    threshold: int,
    label_mode: str,
    llm_judge_pool_size: int,
    llm,
    judge_cache: JudgeCache | None,
    balance: bool,
    ensure_positive_coverage: bool,
    pyserini_client,
) -> tuple[list[Candidate], list[int]]:
    positives = [c for c in pool if topic_qrels.get(c.docid, -1) >= threshold]
    negatives = [c for c in pool if c.docid in topic_qrels and topic_qrels[c.docid] < threshold]
    unjudged = [c for c in pool if c.docid not in topic_qrels]

    labels: dict[str, int] = {c.docid: 1 for c in positives}
    labels.update({c.docid: 0 for c in negatives})
    labeled_candidates: list[Candidate] = list(positives) + list(negatives)

    # Pool order is retrieval-rank order, so truncating this list (below,
    # for --balance) keeps the highest-ranked -- i.e. hardest -- unjudged
    # candidates first, rather than an arbitrary subset.
    if label_mode == "llm-judge" and llm is not None:
        for candidate in unjudged[:llm_judge_pool_size]:
            if not candidate.text:
                continue
            grade = judge_cache.get(topic.topic_id, candidate.docid) if judge_cache else None
            if grade is None:
                grade = judge_relevance(llm, topic.text, candidate.text)
                if grade is None:
                    logger.warning(
                        "topic %s: unparsable LLM judgment for %s, skipping", topic.topic_id, candidate.docid
                    )
                    continue
                if judge_cache:
                    judge_cache.put(topic.topic_id, candidate.docid, grade)
            labels[candidate.docid] = 1 if grade >= threshold else 0
            labeled_candidates.append(candidate)

    if ensure_positive_coverage and 1 not in labels.values():
        for docid, grade in topic_qrels.items():
            if grade < threshold or docid in labels:
                continue
            text = pyserini_client.fetch_doc(docid)
            if not text:
                logger.warning(
                    "topic %s: qrels-positive %s not retrievable, skipping backfill", topic.topic_id, docid
                )
                continue
            # Never found by our own retrieval pool, so there is no real
            # bm25_score for it. 0.0 is the honest neutral stand-in -- the
            # same value min-max normalization assigns the pool's weakest
            # member -- not a claim that it actually scored that low.
            backfilled = Candidate(docid=docid, score=0.0, rank=len(pool) + 1, text=text, features={"bm25_score": 0.0})
            labeled_candidates.append(backfilled)
            labels[docid] = 1
            logger.info("topic %s: backfilled missing qrels-positive %s", topic.topic_id, docid)
            break  # one guaranteed positive is enough to make the topic usable

    if balance:
        pos = [c for c in labeled_candidates if labels[c.docid] == 1]
        neg = [c for c in labeled_candidates if labels[c.docid] == 0]
        n = min(len(pos), len(neg))
        labeled_candidates = pos[:n] + neg[:n]

    label_list = [labels[c.docid] for c in labeled_candidates]
    return labeled_candidates, label_list


def write_examples(
    path: Path,
    topic: Topic,
    pool: list[Candidate],
    labeled_candidates: list[Candidate],
    label_list: list[int],
    max_doc_chars: int,
) -> int:
    # Normalized over the *whole* pool, not just the labeled subset -- this
    # must match `QDERReranker`'s own `_minmax_normalize(bm25_like)` call at
    # inference time, which scales relative to everything the reranker sees,
    # not just the examples that happen to carry a training label.
    pool_bm25 = [c.features.get("bm25_score", 0.0) for c in pool]
    normalized_by_docid = dict(zip((c.docid for c in pool), _minmax_normalize(pool_bm25))) if pool_bm25 else {}

    with open(path, "a", encoding="utf-8") as fh:
        for candidate, label in zip(labeled_candidates, label_list):
            line = {
                "query": topic.text,
                "query_id": topic.topic_id,
                "query_ent_emb": [],
                "doc_id": candidate.docid,
                "doc": (candidate.text or "")[:max_doc_chars],
                "doc_score": normalized_by_docid.get(candidate.docid, 0.0),
                "doc_ent_emb": [],
                "label": label,
            }
            fh.write(json.dumps(line) + "\n")
    return len(labeled_candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics", required=True, help="Topics TSV to build data from.")
    parser.add_argument("--qrels", required=True, help="Qrels for --topics (TREC format).")
    parser.add_argument("--dev-topics", default=None, help="Separate dev-split topics TSV; if omitted, split --topics.")
    parser.add_argument("--dev-qrels", default=None, help="Qrels for --dev-topics; defaults to --qrels.")
    parser.add_argument("--dev-fraction", type=float, default=0.1, help="Used only when --dev-topics is omitted.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--pool-size", type=int, default=None, help="Candidate pool depth per topic; default = stages.colbert.top_k."
    )
    parser.add_argument("--relevance-threshold", type=int, default=1, help="qrels/LLM grade >= this becomes label 1.")
    parser.add_argument("--label-mode", choices=["qrels-only", "llm-judge"], default="qrels-only")
    parser.add_argument(
        "--llm-judge-pool-size", type=int, default=20, help="Unjudged candidates per topic sent to the LLM judge."
    )
    parser.add_argument(
        "--llm-judge-backend",
        choices=["openai", "anthropic"],
        default="openai",
        help="'openai' is what a vLLM OpenAI-compatible server speaks.",
    )
    parser.add_argument(
        "--llm-judge-model-env",
        default="QDER_JUDGE_MODEL",
        help="Env var holding the judge model name (e.g. Qwen2.5-72B-Instruct, gemma-3-27b-it).",
    )
    parser.add_argument(
        "--llm-judge-cache", default=None, help="Path to the judge cache JSONL; default derived from --output-train."
    )
    parser.add_argument("--balance", action="store_true", help="Equalize positive/negative counts per topic.")
    parser.add_argument(
        "--no-ensure-positive-coverage",
        action="store_true",
        help="Don't backfill a qrels-positive document our own pool missed.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=4000,
        help="Chars of document text written; the tokenizer truncates further at train time regardless.",
    )
    parser.add_argument("--limit-topics", type=int, default=None, help="Process only the first N topics (smoke test).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-dev", required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    load_dotenv_local()

    config = load_pipeline_config(args.config)
    stages = config["stages"]

    pyserini_client = build_pyserini_client(config)
    bm25 = BM25Retriever(pyserini_client)
    biencoder = build_biencoder(stages.get("biencoder", {}))
    ltr = build_ltr(stages.get("ltr", {}))
    colbert = build_colbert(stages.get("colbert", {}))

    depths = {
        "bm25_hits": int(stages["bm25"]["hits"]),
        "hybrid_top_k": int(stages["hybrid"]["top_k"]),
        "rrf_k": int(stages["hybrid"]["rrf_k"]),
        "ltr_top_k": int(stages.get("ltr", {}).get("top_k", 200)),
        "colbert_top_k": int(stages.get("colbert", {}).get("top_k", 100)),
    }
    depths["pool_size"] = int(args.pool_size or depths["colbert_top_k"])

    llm = None
    judge_cache = None
    if args.label_mode == "llm-judge":
        llm = build_llm_client(args.llm_judge_backend, config["llm"], model_env_override=args.llm_judge_model_env)
        cache_path = (
            Path(args.llm_judge_cache)
            if args.llm_judge_cache
            else Path(args.output_train).with_suffix(".judge_cache.jsonl")
        )
        judge_cache = JudgeCache(cache_path)

    all_topics = load_topics(args.topics)
    if args.limit_topics:
        all_topics = all_topics[: args.limit_topics]
    qrels = load_qrels(args.qrels)

    if args.dev_topics:
        train_topics = all_topics
        dev_topics = load_topics(args.dev_topics)
        dev_qrels = load_qrels(args.dev_qrels) if args.dev_qrels else qrels
    else:
        train_topics, dev_topics = split_topics(all_topics, args.dev_fraction, args.seed)
        dev_qrels = qrels

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_dev).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_train).write_text("")
    Path(args.output_dev).write_text("")

    for split_name, topics, split_qrels, out_path in (
        ("train", train_topics, qrels, Path(args.output_train)),
        ("dev", dev_topics, dev_qrels, Path(args.output_dev)),
    ):
        topics_written = 0
        topics_skipped = 0
        examples_written = 0
        for topic in topics:
            topic_qrels = split_qrels.get(topic.topic_id, {})
            if not topic_qrels:
                logger.warning("%s: topic %s has no qrels, skipping", split_name, topic.topic_id)
                topics_skipped += 1
                continue

            pool = build_candidate_pool(topic, bm25, biencoder, ltr, colbert, depths)
            labeled_candidates, label_list = label_topic(
                topic,
                pool,
                topic_qrels,
                args.relevance_threshold,
                args.label_mode,
                args.llm_judge_pool_size,
                llm,
                judge_cache,
                args.balance,
                not args.no_ensure_positive_coverage,
                pyserini_client,
            )

            if len(labeled_candidates) < 2 or 1 not in label_list or 0 not in label_list:
                logger.warning(
                    "%s: topic %s has %d usable examples (labels=%s), skipping",
                    split_name,
                    topic.topic_id,
                    len(labeled_candidates),
                    sorted(set(label_list)),
                )
                topics_skipped += 1
                continue

            examples_written += write_examples(out_path, topic, pool, labeled_candidates, label_list, args.max_doc_chars)
            topics_written += 1
            logger.info(
                "%s: topic %s -> %d examples (%d positive)",
                split_name,
                topic.topic_id,
                len(labeled_candidates),
                sum(label_list),
            )

        logger.info(
            "%s: %d topics written, %d skipped, %d total examples -> %s",
            split_name,
            topics_written,
            topics_skipped,
            examples_written,
            out_path,
        )


if __name__ == "__main__":
    main()
