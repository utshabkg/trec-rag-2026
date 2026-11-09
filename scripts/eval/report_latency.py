#!/usr/bin/env python3
"""Turn a run's stage diagnostics into a paper-ready latency breakdown.

Reads the `stages_*.jsonl` written by `run_pipeline.py` and reports per-stage
wall clock: mean and median per narrative, share of total, and the throughput
implication. Pairs with `evaluate_retrieval.py` so a stage's cost can be set
against what it buys -- the tradeoff argument only works if both halves are
measured on the same runs.

    python scripts/eval/report_latency.py logs/stages_devcascade_*.jsonl
    python scripts/eval/report_latency.py logs/stages_*.jsonl --markdown

THE CACHE CAVEAT, which decides whether these numbers mean anything.

Retrieval responses are cached on disk. A cached run reads ~60MB from local disk
where a cold run makes a rate-limited HTTPS request for the same payload, so the
`bm25` row can differ by an order of magnitude between two runs of identical
code. Any latency figure that goes in a paper has to say which it was. Pass
`--cold-api-seconds` with a measured cold cost (see `scripts/check_api.py`, which
reports it) to project what an uncached deployment would cost; the report then
prints both, clearly labelled.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import platform
import shutil
import statistics
import subprocess
from pathlib import Path

# Stage order as the pipeline executes it, so the table reads top to bottom.
STAGE_ORDER = [
    "bm25", "passage_selection", "biencoder", "rrf",
    "ltr", "colbert", "cross_encoder", "llm_listwise", "generation",
]

KIND = {
    "bm25": "network (Pyserini API)",
    "passage_selection": "CPU",
    "biencoder": "GPU",
    "rrf": "CPU",
    "ltr": "CPU",
    "colbert": "GPU",
    "cross_encoder": "GPU",
    "llm_listwise": "LLM",
    "generation": "LLM",
}


def hardware_summary() -> list[str]:
    """Captured at report time, so the paper can state the platform concretely."""
    lines = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True, timeout=15
        ).strip().splitlines()
        for i, row in enumerate(out):
            name, mem, drv = (x.strip() for x in row.split(","))
            lines.append(f"GPU {i}: {name}, {mem}, driver {drv}")
    except (OSError, subprocess.SubprocessError):
        lines.append("GPU: not detected")

    try:
        cpu = next(
            l.split(":", 1)[1].strip()
            for l in Path("/proc/cpuinfo").read_text().splitlines()
            if l.startswith("model name")
        )
        cores = sum(1 for l in Path("/proc/cpuinfo").read_text().splitlines()
                    if l.startswith("processor"))
        lines.append(f"CPU: {cpu} ({cores} logical cores)")
    except (OSError, StopIteration):
        lines.append(f"CPU: {platform.processor() or 'unknown'}")

    try:
        mem_kb = next(
            int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
            if l.startswith("MemTotal")
        )
        lines.append(f"RAM: {mem_kb / 1024**2:.0f} GB")
    except (OSError, StopIteration):
        pass

    lines.append(f"OS: {platform.system()} {platform.release()}, Python {platform.python_version()}")
    return lines


def load_stages(paths: list[str]) -> dict[str, dict[str, list[float]]]:
    by_run: dict[str, dict[str, list[float]]] = {}
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            run = Path(path).stem.replace("stages_", "")
            timings: dict[str, list[float]] = collections.defaultdict(list)
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("elapsed_s") is not None:
                    timings[rec["stage"]].append(rec["elapsed_s"])
            if timings:
                by_run[run] = timings
    return by_run


def report(run: str, timings: dict[str, list[float]], cold_api: float | None, markdown: bool) -> None:
    stages = [s for s in STAGE_ORDER if s in timings] + [
        s for s in timings if s not in STAGE_ORDER
    ]
    n_topics = max(len(v) for v in timings.values())
    per_topic_total = sum(statistics.mean(timings[s]) for s in stages)

    print(f"\n### {run}   ({n_topics} narratives)")
    if markdown:
        print("\n| stage | kind | mean s/narrative | median | share | total s |")
        print("|---|---|---|---|---|---|")
    else:
        print(f"\n{'stage':20s} {'kind':22s} {'mean s':>9s} {'median':>8s} {'share':>7s} {'total s':>9s}")
        print("-" * 80)

    for stage in stages:
        values = timings[stage]
        mean = statistics.mean(values)
        median = statistics.median(values)
        share = mean / per_topic_total if per_topic_total else 0.0
        total = sum(values)
        kind = KIND.get(stage, "-")
        if markdown:
            print(f"| {stage} | {kind} | {mean:.2f} | {median:.2f} | {share:.1%} | {total:.0f} |")
        else:
            print(f"{stage:20s} {kind:22s} {mean:9.2f} {median:8.2f} {share:6.1%} {total:9.0f}")

    if markdown:
        print(f"| **total** | | **{per_topic_total:.2f}** | | 100% | "
              f"**{per_topic_total * n_topics:.0f}** |")
    else:
        print("-" * 80)
        print(f"{'TOTAL':20s} {'':22s} {per_topic_total:9.2f} {'':8s} {100:5.1f}% "
              f"{per_topic_total * n_topics:9.0f}")

    print(f"\n  throughput: {3600 / per_topic_total:.0f} narratives/hour "
          f"({per_topic_total * 119 / 60:.0f} min for the 119-narrative test set)")

    if "bm25" in timings:
        cached = statistics.mean(timings["bm25"])
        print(f"  NOTE: the bm25 row is {cached:.2f}s/narrative as measured. If this run used the")
        print("        response cache, that is a local disk read, NOT live API latency.")
        if cold_api:
            projected = per_topic_total - cached + cold_api
            print(f"        With a measured cold API cost of {cold_api:.2f}s, an uncached run "
                  f"would be ~{projected:.2f}s/narrative")
            print(f"        ({3600 / projected:.0f} narratives/hour, "
                  f"{projected * 119 / 60:.0f} min for the test set).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage_logs", nargs="+", help="logs/stages_*.jsonl (globs accepted)")
    parser.add_argument("--cold-api-seconds", type=float, default=None,
                        help="measured cold Pyserini latency per narrative, to project an uncached run")
    parser.add_argument("--markdown", action="store_true", help="emit markdown tables for the paper")
    args = parser.parse_args()

    by_run = load_stages(args.stage_logs)
    if not by_run:
        raise SystemExit("no timed stage records found (runs before timing was added have none)")

    print("## Hardware")
    for line in hardware_summary():
        print(f"  {line}")
    total, _, free = shutil.disk_usage("/")
    print(f"  Disk: {free // 1024**3} GB free of {total // 1024**3} GB")

    for run, timings in by_run.items():
        report(run, timings, args.cold_api_seconds, args.markdown)

    print("\nSingle-process, one narrative at a time (--workers 1): the neural stages")
    print("share one model instance each, so these are per-narrative serial latencies,")
    print("not a throughput-optimised deployment.")


if __name__ == "__main__":
    main()
