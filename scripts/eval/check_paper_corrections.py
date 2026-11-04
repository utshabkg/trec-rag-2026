#!/usr/bin/env python3
"""Recompute the numbers behind the proposed corrections to the paper.

Reads the run files, dev qrels and run logs on this branch, plus the organizers'
retrieval results folder, which is not in the repository:

    python scripts/eval/check_paper_corrections.py --organizer-dir PATH/TO/retrievalresults

Dev scores here use condensed lists, like `evaluate_retrieval.py --condensed`:
unjudged documents are dropped before scoring. That scorer reports the same
deltas, but not the judged counts or the equal depth comparison printed here.

Equal depth means both lists are cut to the smaller judged count for that topic
before scoring, so neither list is penalised just for holding fewer judged
documents. It compares order among judged documents only, so it cannot say
whether the unjudged documents that replaced them are relevant.

P values are two-sided paired t-tests over topics, with no multiple-comparison
correction. They need scipy and print as n/a without it.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None

ROOT = Path(__file__).resolve().parents[2]
RUN_FILE = "r_output_trec_rag_2026.tsv"

SUBMITTED = {
    "bruteforce": ROOT / "submissions/retrieval/bruteforce-cascade" / RUN_FILE,
    "decomposed": ROOT / "submissions/retrieval/decomposed-cascade" / RUN_FILE,
}
OFFICIAL_RUNS = {
    "bruteforce": "bruteforce-cascade-r",
    "decomposed": "decomposed-cascade-r",
}

QRELS_DIR = ROOT / "data/dev/rag25-dev-umbrela-qrels"
JUDGES = {
    "codex": "rag25-climbmix-umbrela-codex-gpt5.5-medium-reasoning-v1.qrels",
    "ministral": "rag25-climbmix-umbrela-ministral-3-14b-instruct-2512-v2.qrels",
    "qwen3.5": "rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels",
}
DEV_RUNS = {
    "composite": ROOT / "output_dev_assubmitted" / RUN_FILE,
    "hits=100": ROOT / "output_dev_decomposed" / RUN_FILE,
    "hits=1000": ROOT / "output_dev_decomposed_hits1000" / RUN_FILE,
}
# hits=100 is the decomposed arm as submitted, so the first pair is decomposition
# against the baseline and the last pair is the widening experiment.
COMPARISONS = [
    ("hits=100", "composite"),
    ("hits=1000", "composite"),
    ("hits=1000", "hits=100"),
]

# In the order they ran: the repair run re-did 12 narratives, so its entry wins.
POOL_LOGS = [ROOT / "logs/decomposed_full.log", ROOT / "logs/decomposed_repair.log"]
NEEDS_FILE = ROOT / "output_decomp/preprocessed_queries_step3.jsonl"
POOL_CAP = 100
COMPOSITE_DOCS = 50
DOCS_PER_NEED = 5

RELEVANT_GRADE = 2


def load_run(path):
    """topic -> docids in rank order."""
    ranked = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            topic, _, docid, rank, *_ = line.split()
            ranked[topic].append((int(rank), docid))
    return {topic: [docid for _, docid in sorted(rows)] for topic, rows in ranked.items()}


def load_qrels(path):
    """topic -> {docid: grade}."""
    qrels = defaultdict(dict)
    with open(path) as handle:
        for line in handle:
            topic, _, docid, grade = line.split()
            qrels[topic][docid] = int(grade)
    return qrels


def condense(docids, judged):
    return [docid for docid in docids if docid in judged]


def ndcg(condensed, judged, depth):
    gain = sum(
        max(judged[docid], 0) / math.log2(rank + 2)
        for rank, docid in enumerate(condensed[:depth])
    )
    best = sorted((max(grade, 0) for grade in judged.values()), reverse=True)[:depth]
    ideal = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(best))
    return gain / ideal if ideal else 0.0


def relevant_total(judged):
    return sum(grade >= RELEVANT_GRADE for grade in judged.values())


def average_precision(condensed, judged):
    hits = 0
    total = 0.0
    for rank, docid in enumerate(condensed, 1):
        if judged[docid] >= RELEVANT_GRADE:
            hits += 1
            total += hits / rank
    relevant = relevant_total(judged)
    return total / relevant if relevant else 0.0


def recall_at(condensed, judged, depth):
    relevant = relevant_total(judged)
    found = sum(judged[docid] >= RELEVANT_GRADE for docid in condensed[:depth])
    return found / relevant if relevant else 0.0


def judged_precision(condensed, judged):
    """Share of judged documents at or above the relevance grade."""
    if not condensed:
        return 0.0
    return sum(judged[docid] >= RELEVANT_GRADE for docid in condensed) / len(condensed)


METRICS = {
    "nDCG@10": lambda c, j: ndcg(c, j, 10),
    "nDCG@20": lambda c, j: ndcg(c, j, 20),
    "nDCG@100": lambda c, j: ndcg(c, j, 100),
    "R@100": lambda c, j: recall_at(c, j, 100),
    "MAP": average_precision,
}


def pearson(xs, ys):
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    norm_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    norm_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if not norm_x or not norm_y:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (norm_x * norm_y)


def paired_p(deltas):
    if ttest_rel is None:
        return "n/a"
    p_value = ttest_rel(deltas, [0.0] * len(deltas)).pvalue
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def cell(deltas):
    return f"{statistics.mean(deltas):+.4f} (p {paired_p(deltas)})"


def check_submitted(organizer_dir):
    """Item 3: how much of the composite run reaches the submitted decomposed list."""
    matches = list(organizer_dir.rglob("retrieval-document-level-scores.tsv"))
    if not matches:
        sys.exit(f"No retrieval-document-level-scores.tsv under {organizer_dir}")
    with open(matches[0]) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    scores = {(row["run"], row["narrative-id"]): row for row in rows}
    scored = sorted({row["narrative-id"] for row in rows if row["narrative-id"] != "all"})

    def hits(arm, narrative):
        row = scores[(OFFICIAL_RUNS[arm], narrative)]
        return round(float(row["set_P"]) * int(row["num_ret"]))

    runs = {arm: load_run(path) for arm, path in SUBMITTED.items()}
    overlap = {
        narrative: len(set(runs["bruteforce"][narrative]) & set(runs["decomposed"][narrative]))
        for narrative in scored
    }
    # Shared relevant hits cannot exceed the list overlap or either run's own hits.
    missing = sum(
        hits("bruteforce", n) - min(hits("bruteforce", n), hits("decomposed", n), overlap[n])
        for n in scored
    )
    print(f"== Item 3, submitted lists ({len(scored)} scored narratives)")
    print(f"bruteforce documents also in the decomposed list: {statistics.mean(overlap.values()):.2f} per narrative")
    print(f"official relevant hits: bruteforce {sum(hits('bruteforce', n) for n in scored)}, decomposed {sum(hits('decomposed', n) for n in scored)}")
    print(f"bruteforce relevant hits missing from the decomposed list: at least {missing}")
    print(f"narratives where decomposed has more hits: {sum(hits('decomposed', n) > hits('bruteforce', n) for n in scored)}, fewer: {sum(hits('decomposed', n) < hits('bruteforce', n) for n in scored)}")
    print()
    return scored


def check_pool(scored):
    """Item 3: on how many scored narratives the pool reaches its cap."""
    pool = {}
    for path in POOL_LOGS:
        with open(path) as handle:
            for line in handle:
                match = re.search(r"Topic (\S+): \d+/\d+ needs retrieved, pool=(\d+)", line)
                if match:
                    pool[match.group(1)] = int(match.group(2))

    needs = {}
    with open(NEEDS_FILE) as handle:
        for line in handle:
            record = json.loads(line)
            needs[record["query_id"]] = sum(len(item.get("ain_array", [])) for item in record.get("ain", []))

    at_cap = under_cap = 0
    derived = []
    undecided = []
    for narrative in scored:
        if narrative in pool:
            at_cap += pool[narrative] >= POOL_CAP
            under_cap += pool[narrative] < POOL_CAP
        elif COMPOSITE_DOCS + DOCS_PER_NEED * needs[narrative] < POOL_CAP:
            # Not in these logs, but too few needs for the pool to reach the cap.
            under_cap += 1
            derived.append(f"{narrative} ({needs[narrative]} needs)")
        else:
            undecided.append(narrative)
    print("== Item 3, pool of 100")
    print(f"scored narratives at the cap: {at_cap}, under it: {under_cap}")
    if derived:
        print(f"under the cap by need count, not logged here: {', '.join(derived)}")
    if undecided:
        print(f"undecided: {', '.join(undecided)}")
    print()


def check_dev():
    """Items 3 and 4: dev lists under condensed scoring."""
    runs = {name: load_run(path) for name, path in DEV_RUNS.items()}
    qrels = {name: load_qrels(QRELS_DIR / file) for name, file in JUDGES.items()}
    topics = sorted(runs["composite"])

    print(f"== Item 3, dev lists ({len(topics)} topics)")
    survive = [len(set(runs["composite"][t]) & set(runs["hits=100"][t])) for t in topics]
    print(f"composite documents also in the decomposed list: {statistics.mean(survive):.2f} of 50")
    for judge, judged in qrels.items():
        known = gone = 0
        for topic in topics:
            kept = set(runs["hits=100"][topic])
            for docid in runs["composite"][topic]:
                if judged[topic].get(docid, -1) >= RELEVANT_GRADE:
                    known += 1
                    gone += docid not in kept
        print(f"{judge}: composite known-relevant documents gone: {gone} of {known} ({100 * gone / known:.1f}%)")
    print()

    print("== Item 4, judged documents per list")
    # The judged sets are the same across the three judges, so one is enough.
    reference = next(iter(qrels.values()))
    judged_count = {
        name: {t: len(condense(runs[name][t], reference[t])) for t in topics} for name in runs
    }
    for name in runs:
        counts = judged_count[name].values()
        print(f"{name}: mean {statistics.mean(counts):.2f}, topics under 20: {sum(c < 20 for c in counts)} of {len(topics)}")
    print()

    print("== Item 4, condensed deltas (delta, p, r with change in judged count)")
    for left, right in COMPARISONS:
        print(f"-- {left} vs {right}")
        judged_change = [judged_count[left][t] - judged_count[right][t] for t in topics]
        for metric, score in METRICS.items():
            parts = []
            for judge, judged in qrels.items():
                deltas = [
                    score(condense(runs[left][t], judged[t]), judged[t])
                    - score(condense(runs[right][t], judged[t]), judged[t])
                    for t in topics
                ]
                parts.append(f"{cell(deltas)} r {pearson(judged_change, deltas):+.2f}")
            print(f"{metric:9}" + " | ".join(parts))
    print(f"columns: {', '.join(qrels)}")
    print()

    print("== Item 4, both lists cut to the smaller judged count per topic")
    for left, right in COMPARISONS:
        print(f"-- {left} vs {right}")
        ndcg_parts, precision_parts = [], []
        for judged in qrels.values():
            ndcg_deltas, precision_deltas = [], []
            for topic in topics:
                a = condense(runs[left][topic], judged[topic])
                b = condense(runs[right][topic], judged[topic])
                depth = min(len(a), len(b))
                ndcg_deltas.append(ndcg(a, judged[topic], depth) - ndcg(b, judged[topic], depth))
                precision_deltas.append(
                    judged_precision(a[:depth], judged[topic]) - judged_precision(b[:depth], judged[topic])
                )
            ndcg_parts.append(cell(ndcg_deltas))
            precision_parts.append(cell(precision_deltas))
        print("nDCG      " + " | ".join(ndcg_parts))
        print("precision " + " | ".join(precision_parts))
    print(f"columns: {', '.join(qrels)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--organizer-dir",
        type=Path,
        required=True,
        help="folder holding retrieval-document-level-scores.tsv, at any depth",
    )
    args = parser.parse_args()
    scored = check_submitted(args.organizer_dir)
    check_pool(scored)
    check_dev()


if __name__ == "__main__":
    main()
