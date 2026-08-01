#!/usr/bin/env python3
"""First-contact smoke test for the Pyserini REST API.

Run this the moment a PYSERINI_API_TOKEN arrives, BEFORE the full pipeline run.
Everything upstream of this point was validated against a mock server, so this
is where real-corpus assumptions get checked. It answers, with evidence:

  1. Is the token valid?
  2. What shape is `doc`?  -- the single biggest silent-failure risk: any
     candidate whose text cannot be extracted is dropped from the biencoder,
     the cross-encoder AND generation, so an unexpected shape empties the
     cascade without raising anything.
  3. Does hits=1000 actually return 1000?
  4. Do docids match the ClimbMix contract (shard_NNNNN_N)?
  5. How slow is a deep request, and how big is the cached response?

Prints a PASS/WARN/FAIL line per check and exits non-zero on any FAIL.
Never prints the token.

    python scripts/check_api.py --config configs/pipeline.gpu.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trecrag.clients.pyserini_client import (
    PyseriniAuthError,
    PyseriniClient,
    PyseriniRequestError,
    _extract_text,
)
from trecrag.config import get_env, load_dotenv_local, load_pipeline_config

DOCID_RE = re.compile(r"^shard_\d{5}_\d+$")

failures = 0
warnings = 0


def report(status: str, check: str, detail: str) -> None:
    global failures, warnings
    if status == "FAIL":
        failures += 1
    elif status == "WARN":
        warnings += 1
    print(f"[{status:4s}] {check}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.gpu.yaml")
    parser.add_argument("--query", default="Albert Einstein", help="health-check query")
    parser.add_argument("--hits", type=int, default=1000, help="depth to probe")
    args = parser.parse_args()

    load_dotenv_local()
    config = load_pipeline_config(args.config)
    pyserini = config["pyserini"]

    token = get_env(pyserini.get("token_env"))
    if not token:
        report("FAIL", "token", f"{pyserini.get('token_env')} is not set (check .env.local)")
        raise SystemExit(1)
    report("PASS", "token", "present (value not shown)")

    # Deliberately no cache: this must exercise the live service.
    client = PyseriniClient(
        base_url=pyserini["base_url"],
        index=pyserini["index"],
        token=token,
        timeout_seconds=float(pyserini.get("timeout_seconds", 60)),
        max_retries=int(pyserini.get("max_retries", 3)),
        cache_dir=None,
    )

    if client.health():
        report("PASS", "health", f"{pyserini['base_url']} reachable")
    else:
        report("FAIL", "health", f"{pyserini['base_url']} did not return 200")
        raise SystemExit(1)

    started = time.time()
    try:
        candidates = client.search(args.query, hits=args.hits)
    except PyseriniAuthError:
        report("FAIL", "auth", "token rejected (missing/expired/invalid)")
        raise SystemExit(1)
    except PyseriniRequestError as exc:
        report("FAIL", "search", str(exc)[:200])
        raise SystemExit(1)
    elapsed = time.time() - started

    report("PASS", "auth", "authenticated search succeeded")
    report(
        "PASS" if candidates else "FAIL",
        "results",
        f"{len(candidates)} candidates in {elapsed:.1f}s",
    )
    if not candidates:
        raise SystemExit(1)

    # Depth: the pipeline asks for 1000 per narrative. Silently getting fewer
    # would shrink the whole cascade's recall without any error.
    if len(candidates) < args.hits:
        report(
            "WARN",
            "depth",
            f"asked hits={args.hits}, got {len(candidates)} -- the API appears to cap "
            f"depth. Lower stages.bm25.hits to {len(candidates)} so the config is honest.",
        )
    else:
        report("PASS", "depth", f"hits={args.hits} fully honoured")

    # The big one: can we actually get text out of every candidate?
    with_text = sum(1 for c in candidates if c.text)
    pct = 100.0 * with_text / len(candidates)
    if with_text == len(candidates):
        report("PASS", "doc-text", f"all {len(candidates)} candidates yielded text")
    elif pct >= 95:
        report("WARN", "doc-text", f"only {with_text}/{len(candidates)} ({pct:.1f}%) yielded text")
    else:
        report(
            "FAIL",
            "doc-text",
            f"only {with_text}/{len(candidates)} ({pct:.1f}%) yielded text -- _extract_text "
            f"cannot read this doc shape. The biencoder, cross-encoder and generator all "
            f"skip textless candidates, so the cascade would silently empty out.",
        )

    # Show the raw shape so a mismatch is diagnosable rather than mysterious.
    raw = client.session.get(
        f"{client.base_url}/v1/{client.index}/search",
        params={"query": args.query, "hits": 1},
        headers={"Authorization": f"Bearer {token}"},
        timeout=client.timeout_seconds,
    ).json()
    sample_doc = (raw.get("candidates") or [{}])[0].get("doc")
    report("INFO", "doc-shape", f"type={type(sample_doc).__name__}")
    if isinstance(sample_doc, dict):
        report("INFO", "doc-keys", f"{sorted(sample_doc)[:10]}")
    extracted = _extract_text(sample_doc)
    report(
        "PASS" if extracted else "FAIL",
        "doc-extract",
        f"{len(extracted)} chars extracted" if extracted else "_extract_text returned None",
    )

    bad_ids = [c.docid for c in candidates if not DOCID_RE.match(c.docid)][:5]
    if bad_ids:
        report("WARN", "docid-format", f"unexpected docid shape, e.g. {bad_ids}")
    else:
        report("PASS", "docid-format", "all docids match shard_NNNNN_N")

    ranks = [c.rank for c in candidates]
    report(
        "PASS" if ranks == sorted(ranks) else "WARN",
        "rank-order",
        "ranks ascending" if ranks == sorted(ranks) else "ranks NOT ascending as returned",
    )

    payload_mb = len(json.dumps(raw).encode()) * len(candidates) / 1e6
    report("INFO", "cache-size", f"~{payload_mb:.0f} MB/narrative, ~{payload_mb * 119:.0f} MB for 119")
    report("INFO", "runtime", f"~{elapsed * 119 / 60:.1f} min of BM25 alone for 119 narratives")

    print()
    if failures:
        print(f"{failures} FAIL(s), {warnings} WARN(s) -- do not start the full run yet.")
        raise SystemExit(1)
    print(f"All checks passed ({warnings} warning(s)). Safe to start the full run.")


if __name__ == "__main__":
    main()
