#!/usr/bin/env python3
"""Score generated answers against the RAG25 development nuggets.

The RAG task is judged on coverage -- did the answer contain the atomic facts a
good answer should contain -- and the 2025 overview found that coverage, not
citation accuracy, is the open problem. This measures it: for each development
narrative, an LLM judge decides which gold nuggets the answer actually covers.
That is the AutoNuggetizer idea the organizers use, run locally so we can iterate
without waiting for official judgments.

    python scripts/eval/evaluate_nuggets.py output_dev/rag_output_trec_rag_2026.jsonl

Reports vital-nugget coverage (the headline), all-nugget coverage, and a strict
score that only counts full support.

DESIGN NOTES

  - One judge call per narrative, not per nugget. The narratives carry 29-90
    nuggets each; per-nugget calls would be ~1250 requests for 22 topics. The
    judge sees the whole answer once and labels every nugget against it, which is
    both faster and better-informed.
  - Labels are support / partial / not_support, matching the track's vocabulary.
    Partial is reported separately rather than folded into either side, because
    collapsing it is exactly how a coverage number gets quietly inflated.
  - The parse is deterministic and all-or-nothing per narrative, with one retry.
    A narrative whose judgment cannot be parsed is reported as unscored rather
    than counted as zero coverage -- scoring a parse failure as a miss would
    understate the system and, worse, do so invisibly.

CAVEAT: this is a model judging a model. Nuggets are development diagnostics, not
NIST assessments, and the judge is a local model rather than the organizers'.
Use it to compare OUR runs against each other, where the judge is held constant.
Do not read an absolute number here as a predicted official score.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from trecrag.clients.llm_client import OpenAICompatibleClient  # noqa: E402
from trecrag.config import load_dotenv_local  # noqa: E402

NUGGETS_PATH = (
    "trec-rag-data/trec-rag-2026/development-data/rag25-dev-nuggets/rag25-dev-nuggets.jsonl"
)

JUDGE_PROMPT = """\
You are checking which facts an answer covers. You judge only what the answer \
says; you never use outside knowledge.

QUESTION:
{narrative}

ANSWER TO JUDGE:
{answer}

FACTS TO CHECK (one per line, numbered):
{nuggets}

For every numbered fact, decide whether the ANSWER conveys it:
  support  - the answer clearly states this fact
  partial  - the answer gestures at it, or states part of it, or hedges it
  no       - the answer does not convey it

Judge meaning, not wording: a fact stated in different words is still support. A \
fact the answer merely mentions as a topic without asserting is partial, not \
support.

Output exactly one line per fact, in order, and nothing else:
1: support
2: no
3: partial

Output {count} lines, numbered 1 to {count}.
"""

_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.\)]\s*(support|partial|no|not_support)\b", re.I | re.M)


def parse_judgments(raw: str, count: int) -> dict[int, str] | None:
    """Map nugget index -> label, or None if the reply is not a usable judgment.

    All-or-nothing on coverage rather than content: a reply that labels only a
    handful of nuggets is a truncated or confused generation, and back-filling
    the rest as "no" would silently invent misses.
    """
    labels: dict[int, str] = {}
    for match in _LINE_RE.finditer(raw or ""):
        idx = int(match.group(1))
        label = match.group(2).lower()
        if label == "not_support":
            label = "no"
        if 1 <= idx <= count and idx not in labels:
            labels[idx] = label
    if len(labels) < count * 0.8:  # judge must cover most of what it was asked
        return None
    return labels


def score_topic(labels: dict[int, str], nuggets: list[dict]) -> dict[str, float]:
    vital = [i for i, n in enumerate(nuggets, 1) if n.get("importance") == "vital"]
    every = list(range(1, len(nuggets) + 1))

    def frac(indices, accepted):
        if not indices:
            return None
        return sum(1 for i in indices if labels.get(i, "no") in accepted) / len(indices)

    return {
        "vital_coverage": frac(vital, {"support", "partial"}),
        "vital_strict": frac(vital, {"support"}),
        "all_coverage": frac(every, {"support", "partial"}),
        "all_strict": frac(every, {"support"}),
        "n_vital": len(vital),
        "n_nuggets": len(every),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", nargs="+", help="RAG JSONL file(s) to score")
    parser.add_argument("--nuggets", default=NUGGETS_PATH)
    parser.add_argument("--model", default=None, help="judge model (default: $LLM_MODEL)")
    parser.add_argument("--base-url", default=None, help="default: $LLM_BASE_URL")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N narratives")
    parser.add_argument("--per-topic", action="store_true")
    args = parser.parse_args()

    load_dotenv_local()
    import os

    base_url = args.base_url or os.environ.get("LLM_BASE_URL")
    model = args.model or os.environ.get("LLM_MODEL")
    if not base_url or not model:
        raise SystemExit("set LLM_BASE_URL and LLM_MODEL (see .env.local), or pass --base-url/--model")
    judge = OpenAICompatibleClient(
        base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "x"),
        model=model, timeout_seconds=900, max_attempts=2,
    )

    nuggets_by_qid = {
        rec["qid"]: rec["nuggets"]
        for rec in (json.loads(l) for l in open(args.nuggets, encoding="utf-8"))
    }
    print(f"loaded nuggets for {len(nuggets_by_qid)} development narratives")
    print(f"judge: {model}\n")

    for spec in args.answers:
        records = [json.loads(l) for l in open(spec, encoding="utf-8")]
        scored, unscored = [], []
        for rec in records[: args.limit] if args.limit else records:
            qid = rec["metadata"]["narrative_id"]
            if qid not in nuggets_by_qid:
                continue
            nuggets = nuggets_by_qid[qid]
            answer = " ".join(s["text"] for s in rec["answer"])
            listing = "\n".join(f"{i}. {n['text']}" for i, n in enumerate(nuggets, 1))
            prompt = JUDGE_PROMPT.format(
                narrative=rec["metadata"]["narrative"], answer=answer,
                nuggets=listing, count=len(nuggets),
            )
            labels = None
            for _ in range(2):  # one retry; a flaky judge reply is not a system failure
                try:
                    labels = parse_judgments(judge.complete(prompt, max_tokens=4096), len(nuggets))
                except Exception as exc:
                    print(f"  {qid}: judge call failed ({type(exc).__name__})", file=sys.stderr)
                if labels:
                    break
            if not labels:
                unscored.append(qid)
                continue
            row = score_topic(labels, nuggets)
            row["qid"] = qid
            row["words"] = len(answer.split())
            scored.append(row)
            if args.per_topic:
                print(f"  {qid:>6s}  vital {row['vital_coverage']:.3f} "
                      f"(strict {row['vital_strict']:.3f})  all {row['all_coverage']:.3f}  "
                      f"{row['n_vital']}/{row['n_nuggets']} nuggets  {row['words']}w")

        print(f"=== {Path(spec).name} ===")
        if not scored:
            print("  no narratives scored -- do the answer ids match the development nuggets?")
            print(f"  (nugget qids look like {sorted(nuggets_by_qid)[:3]})")
            continue
        for metric in ("vital_coverage", "vital_strict", "all_coverage", "all_strict"):
            values = [r[metric] for r in scored if r[metric] is not None]
            print(f"  {metric:16s} {statistics.mean(values):.4f}")
        print(f"  {'narratives':16s} {len(scored)} scored"
              + (f", {len(unscored)} UNSCORED (judge unparsable): {unscored}" if unscored else ""))
        print(f"  {'mean words':16s} {statistics.mean(r['words'] for r in scored):.0f}")

    print("\nA local model judging a local model, over development nuggets.")
    print("Comparable across OUR runs (same judge); not a predicted official score.")


if __name__ == "__main__":
    main()
