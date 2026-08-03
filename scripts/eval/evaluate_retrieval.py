#!/usr/bin/env python3
"""Score TREC run files against the RAG25 development qrels.

Until now every ranking decision in this project has been made by inspection --
"this looks better" -- because no dev signal was wired up. This turns that into
numbers: nDCG, recall and MAP over the 22 RAG25 development narratives, whose
ClimbMix relevance judgments ship in `trec-rag-data`.

    python scripts/eval/evaluate_retrieval.py run_a.trec run_b.trec

Compares any number of runs side by side, so a config change (cascade width,
passage selection, decomposition) can be judged on measurement rather than
argument.

IMPORTANT CAVEATS, both from the track's own development-data reference:

  - These are *projected* UMBRELA judgments produced by a model, not NIST
    assessments. They are development diagnostics. A run that wins here has not
    been shown to win on the official test set.
  - Three assessor variants ship (ministral / qwen3.5 / codex-gpt5.5). They do
    not agree. `--qrels all` scores against every one, because a difference that
    only appears under a single assessor is not a result. The track reference
    says to pick one consistently or report each separately; this defaults to
    reporting all three.

Also note the dev narratives are RAG25 topics with numeric ids (14, 31, 58...),
NOT the 2026 test narratives (`rag2026-*`). A run over the test topics cannot be
scored here -- no test qrels have been released. To evaluate a system you must
first run it over `topics/rag25-topics-dev.tsv`.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

DEV_ROOT = Path("trec-rag-data/trec-rag-2026/development-data")
QRELS_DIR = DEV_ROOT / "rag25-dev-umbrela-qrels"
DEV_TOPICS = DEV_ROOT / "topics/rag25-topics-dev.tsv"

# UMBRELA grades run 0-3 (0 irrelevant, 3 perfectly relevant). Binary metrics
# (recall, MAP) need a cut: >=2 means "helpful", which is the conventional
# UMBRELA positive threshold. nDCG uses the graded values directly.
DEFAULT_REL_THRESHOLD = 2

METRICS = {
    "ndcg_cut.10",
    "ndcg_cut.20",
    "ndcg_cut.100",
    "recall.100",
    "recall.1000",
    "map",
}
DISPLAY = ["ndcg_cut_10", "ndcg_cut_20", "ndcg_cut_100", "recall_100", "recall_1000", "map"]


def load_qrels(path: Path, rel_threshold: int) -> tuple[dict, dict]:
    """Return (graded, binary) qrels dicts in pytrec_eval's nested format."""
    graded: dict[str, dict[str, int]] = collections.defaultdict(dict)
    binary: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 4:
            if line.strip():
                print(f"  warn: {path.name}:{line_no}: expected 4 fields", file=sys.stderr)
            continue
        topic, _, docid, grade = parts
        try:
            g = int(grade)
        except ValueError:
            continue
        graded[topic][docid] = g
        binary[topic][docid] = 1 if g >= rel_threshold else 0
    return dict(graded), dict(binary)


def load_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 6:
            continue
        topic, _, docid, _, score, _ = parts
        try:
            run[topic][docid] = float(score)
        except ValueError:
            continue
    return dict(run)


def evaluate(run: dict, graded: dict, binary: dict) -> tuple[dict[str, float], dict]:
    """Graded metrics off graded qrels, binary metrics off thresholded qrels.

    Kept separate deliberately: feeding graded labels to recall/MAP silently
    counts grade-1 ("related but unhelpful") documents as hits and inflates them.
    """
    import pytrec_eval

    graded_eval = pytrec_eval.RelevanceEvaluator(graded, {"ndcg_cut.10", "ndcg_cut.20", "ndcg_cut.100"})
    binary_eval = pytrec_eval.RelevanceEvaluator(binary, {"recall.100", "recall.1000", "map"})

    per_topic: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for source in (graded_eval.evaluate(run), binary_eval.evaluate(run)):
        for topic, scores in source.items():
            per_topic[topic].update(scores)

    if not per_topic:
        return {}, {}  # must stay a 2-tuple; the caller unpacks unconditionally
    return {
        metric: statistics.mean(scores.get(metric, 0.0) for scores in per_topic.values())
        for metric in DISPLAY
    }, per_topic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="TREC run file(s) to score")
    parser.add_argument("--qrels", default="all",
                        help="'all' (default), or a path/substring selecting one qrels variant")
    parser.add_argument("--qrels-dir", default=str(QRELS_DIR))
    parser.add_argument("--rel-threshold", type=int, default=DEFAULT_REL_THRESHOLD,
                        help="minimum UMBRELA grade counted as relevant for recall/MAP")
    parser.add_argument("--per-topic", action="store_true", help="also print per-topic nDCG@10")
    parser.add_argument(
        "--condensed", action="store_true",
        help="drop unjudged documents before scoring (Sakai's condensed lists). Use this "
             "whenever comparing a reranker against the system the judgment pool was built "
             "from -- otherwise the reranker is penalised for surfacing unjudged documents.",
    )
    parser.add_argument(
        "--judged-rate", action="store_true",
        help="report what fraction of each run's top-k was actually judged",
    )
    args = parser.parse_args()

    qrels_files = sorted(Path(args.qrels_dir).glob("*.qrels"))
    if args.qrels != "all":
        candidate = Path(args.qrels)
        qrels_files = [candidate] if candidate.is_file() else [
            f for f in qrels_files if args.qrels in f.name
        ]
    if not qrels_files:
        raise SystemExit(f"no qrels found (looked in {args.qrels_dir})")

    runs = {}
    for spec in args.runs:
        path = Path(spec)
        if not path.is_file():
            raise SystemExit(f"run file not found: {path}")
        runs[path.name] = load_run(path)

    for qrels_path in qrels_files:
        graded, binary = load_qrels(qrels_path, args.rel_threshold)
        judged_topics = set(graded)
        variant = qrels_path.name.replace("rag25-climbmix-umbrela-", "").replace(".qrels", "")
        print(f"\n=== qrels: {variant}  ({len(judged_topics)} judged topics, "
              f"relevant = grade >= {args.rel_threshold}) ===")
        header = f"{'run':42s} " + " ".join(f"{m.replace('ndcg_cut','nDCG@').replace('recall_','R@'):>10s}" for m in DISPLAY)
        print(header)
        print("-" * len(header))
        for name, run in runs.items():
            overlap = judged_topics & set(run)
            if not overlap:
                # Almost always means a test-topic run scored against dev qrels.
                print(f"{name[:42]:42s}  NO OVERLAP with judged topics "
                      f"(run has {len(run)} topics, e.g. {sorted(run)[:2]})")
                continue
            scoped = {t: run[t] for t in overlap}
            if args.judged_rate:
                rates = []
                for topic, docs in scoped.items():
                    top = sorted(docs.items(), key=lambda kv: -kv[1])[:10]
                    if top:
                        rates.append(sum(1 for d, _ in top if d in graded[topic]) / len(top))
                print(f"{name[:42]:42s}  judged@10 = {statistics.mean(rates):.1%}")
            if args.condensed:
                # Unjudged documents score 0, so a reranker that promotes documents
                # the pool never saw is punished for the pool's shape rather than
                # its own ranking. Removing them compares systems only on documents
                # the assessors actually looked at.
                scoped = {
                    t: {d: s for d, s in docs.items() if d in graded.get(t, {})}
                    for t, docs in scoped.items()
                }
                scoped = {t: d for t, d in scoped.items() if d}
            scores, per_topic = evaluate(scoped, graded, binary)
            row = f"{name[:42]:42s} " + " ".join(f"{scores.get(m, 0.0):10.4f}" for m in DISPLAY)
            if len(overlap) < len(judged_topics):
                row += f"   [{len(overlap)}/{len(judged_topics)} topics]"
            print(row)
            if args.per_topic:
                for topic in sorted(per_topic, key=lambda t: per_topic[t].get("ndcg_cut_10", 0)):
                    print(f"      {topic:>6s}  nDCG@10={per_topic[topic].get('ndcg_cut_10', 0):.4f}")

    # Measured on these qrels: ~900 documents per topic are graded >=2, so R@100
    # cannot exceed ~0.11 even for a perfect ranking. Reading it as "we only found
    # 11% of the relevant documents" would be wrong -- it is a ceiling, not a
    # failure. nDCG@10/@20 and MAP are the metrics that discriminate here.
    rel_per_topic = statistics.mean(
        sum(1 for g in docs.values() if g >= args.rel_threshold) for docs in graded.values()
    )
    print(f"\nNote: ~{rel_per_topic:.0f} documents/topic are graded >= {args.rel_threshold},")
    print(f"so R@100 is capped near {100 / rel_per_topic:.2f} even for a perfect run.")
    print("Prefer nDCG@10/@20 and MAP when comparing systems.")
    print("\nDevelopment diagnostics only: these are model-projected UMBRELA judgments,")
    print("not NIST assessments. Treat a win here as a signal, not a result --")
    print("and check it holds across all three assessor variants.")


if __name__ == "__main__":
    main()
