# ClimbMix: individual documents that kill the response stream

**For:** TREC RAG 2026 Slack — thread reply / new message to @Nour Jedidi, @Lingwei Gu
**From:** iris-mst (Missouri S&T)

---

Following the rate-limit thread — we've hit something that looks **separate from
the 429 / load-shedding issue**, and it has a one-request reproducer.

## Minimal reproducer

```bash
curl -sS -K .curlrc.pyserini-rest \
  "http://api.castorini.uwaterloo.ca/v1/climbmix-400b/doc/shard_02798_23484"
```

Returns HTTP 200 and then drops the connection part-way through the body:

```
ChunkedEncodingError: Connection broken:
ConnectionResetError(104, 'Connection reset by peer')
```

Single document, single request, no search, no concurrency. Adding
`max_doc_length` makes the same fetch succeed, so the document is retrievable —
it's writing its **full** contents that kills the stream. Our guess would be an
encoding edge case in the stored text (lone surrogate / invalid UTF-8), but
you'll be able to see that directly.

## Why we don't think this is the 429 / load-shedding behaviour

- It's a **single** request, so there's no QPS or volume to shed.
- The failure is a mid-body connection reset after a 200, not a 429.
- It's **stable across hours and load conditions**: we measured identical
  failure thresholds in two sessions several hours apart, one when the endpoint
  was answering in ~2s and one when the same request took ~22s.
- A **smaller** request fails while a **larger** one succeeds — `rag2026-70` at
  `hits=25` (~4.5 MB) fails, while `rag2026-72` at `hits=500` (27 MB) succeeds.
  Volume-based shedding can't produce that ordering.

## How it surfaces in a search pipeline

Any `search` deep enough to include the document dies. On the official test
narratives:

| narrative | succeeds | fails | offending rank |
|---|---|---|---|
| `rag2026-70` | `hits` ≤ 13 | `hits` ≥ 14 | rank 14 = `shard_02798_23484` |
| `rag2026-72` | `hits` ≤ 500 | `hits` ≥ 800 | somewhere in ranks 501–800 |

Two narratives out of 119 in our runs, so there are probably a handful of these
documents in the collection rather than just one.

## Why it's worth fixing before the deadline

The failure arrives as a transport error *after* a 200, so status-code-based
retry (urllib3's `Retry`, most HTTP client defaults) never sees it. And because
it's deterministic, retrying never recovers it — a pipeline that doesn't
special-case this simply loses those narratives from its submission, quite
possibly without anyone noticing until after judging.

Our workaround is to walk `hits` down until the offending document falls outside
the window, which costs retrieval depth on the affected narratives (we retrieve
10 and 500 candidates instead of 1000 on those two).

Happy to send full traces or test a fix. And thanks for the QPS guidance above —
we've been running a single in-flight request with exponential backoff, and
haven't seen a 429.

— iris-mst
