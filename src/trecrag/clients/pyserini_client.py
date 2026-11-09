"""Client for the Pyserini REST API (ClimbMix search + document fetch).

Endpoint contract, per the `pyserini-rest-api` skill:
    GET /                          health check, unauthenticated
    GET /v1/{index}/search?query=...&hits=...
    GET /v1/{index}/doc/{docid}

`doc` in responses may be a string, object, array, number, boolean, or null;
`_extract_text` normalizes it to plain text (or None).

Token handling: the token is read from an environment variable (never a CLI
argument or a tracked file) and is never logged. Request an access token by
emailing get-pyserini@googlegroups.com.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from trecrag.logging_utils import get_logger

logger = get_logger("pyserini_client")

# urllib3's Retry keys off HTTP status codes, so it only covers failures that
# produce a response. ClimbMix search payloads run to ~60 MB, and a connection
# dropped part-way through the body raises at the read layer instead -- observed
# once in 73 topics on a real run. Retry those explicitly.
_TRANSPORT_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    json.JSONDecodeError,  # truncated body that still parsed as a complete response
)


class PyseriniAuthError(RuntimeError):
    pass


class PyseriniRequestError(RuntimeError):
    pass


@dataclass
class SearchCandidate:
    docid: str
    rank: int
    score: float
    text: str | None


def _extract_text(doc: Any) -> str | None:
    if doc is None:
        return None
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        for key in ("text", "contents", "content", "body"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
        return None
    # Numbers, booleans, arrays: not usable as passage text.
    return None


class PyseriniClient:
    def __init__(
        self,
        base_url: str,
        index: str,
        token: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
        cache_dir: str | Path | None = None,
        backoff_factor: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.index = index
        self._token = token
        self.timeout_seconds = timeout_seconds
        self.session = session or self._build_session(max_retries, backoff_factor)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Latency accounting. A cached run reads ~60MB off disk instead of the
        # network, so reporting wall clock without this split would present disk
        # reads as API latency and understate the real cost of the system by an
        # order of magnitude.
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "api_seconds": 0.0,
            "cache_seconds": 0.0,
        }

    def _get_json(
        self, url: str, params: dict[str, Any] | None = None, max_attempts: int | None = None
    ) -> Any:
        """GET and decode JSON, retrying transport-level failures with backoff.

        Reading the body is where large ClimbMix responses fail, and that happens
        after the status line has already been accepted -- so this sits outside
        the adapter's status-code retry rather than duplicating it.
        """
        attempts = self.max_retries + 1 if max_attempts is None else max(1, max_attempts)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self.session.get(
                    url, params=params, headers=self._headers(), timeout=self.timeout_seconds
                )
                self._raise_for_status(response)
                return response.json()
            except PyseriniAuthError:
                raise  # a bad token will not fix itself; fail fast
            except _TRANSPORT_ERRORS as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                delay = self.backoff_factor * (2**attempt)
                logger.warning(
                    "transport error (%s), retrying in %.1fs [%d/%d]",
                    type(exc).__name__, delay, attempt + 1, self.max_retries,
                )
                time.sleep(delay)
        raise PyseriniRequestError(
            f"Pyserini API request failed after {self.max_retries} retries: "
            f"{type(last_error).__name__}: {last_error}"
        )

    def _cache_path(self, kind: str, key: str) -> Path | None:
        """The API is shared infrastructure and results are deterministic, so a
        re-run (or a resume after a crash) should never re-issue a request it has
        already paid for. Keyed on index+query+hits, so changing depth re-fetches."""
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(f"{self.index}|{kind}|{key}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{kind}_{digest}.json"

    @staticmethod
    def _build_session(max_retries: int, backoff_factor: float = 0.5) -> requests.Session:
        session = requests.Session()
        # The API is rate-limited shared infrastructure. urllib3 honours a
        # Retry-After header when the service sends one and falls back to
        # exponential backoff otherwise, per the skill's pacing rules.
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _headers(self) -> dict[str, str]:
        if not self._token:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def health(self) -> bool:
        response = self.session.get(f"{self.base_url}/", timeout=self.timeout_seconds)
        return response.status_code == 200

    def search_adaptive(
        self, query: str, hits: int, min_hits: int = 10
    ) -> tuple[list[SearchCandidate], int]:
        """Search at `hits`, falling back to shallower depths on a hard failure.

        Some ClimbMix documents crash the API's response stream: the request dies
        mid-body with a connection reset, deterministically, for any depth deep
        enough to include the offending document. Measured on the real service:
        rag2026-70 succeeds at hits<=10 and fails at >=25; rag2026-72 succeeds at
        <=500 and fails at >=800. `parse=false` does not avoid it.

        Retrying cannot fix a deterministic failure, so instead we walk the depth
        down until the bad document falls outside the requested window. A
        narrative retrieved shallowly is far better than one missing from the
        submission entirely. Returns the candidates and the depth actually used,
        so the caller can record what happened rather than silently degrade.
        """
        try:
            return self.search(query, hits=hits), hits
        except PyseriniRequestError:
            logger.warning(
                "search failed at hits=%d after retries; walking depth down "
                "(a document in this result set breaks the API's response stream)",
                hits,
            )

        # Halve repeatedly, but always finish on min_hits itself: plain halving
        # steps straight past the floor (1000 -> ... -> 15 -> 7), and 7 < floor
        # ends the walk without ever trying 10 -- which is exactly the depth that
        # works for rag2026-70.
        ladder: list[int] = []
        depth = hits // 2
        while depth > min_hits:
            ladder.append(depth)
            depth //= 2
        ladder.append(min_hits)

        for depth in ladder:
            try:
                # One attempt per depth: failures here are deterministic, so the
                # patient backoff already spent at full depth would only stall.
                candidates = self.search(query, hits=depth, max_attempts=1)
                logger.warning(
                    "search recovered at hits=%d (requested %d); this narrative is "
                    "retrieved shallower than the rest of the run",
                    depth, hits,
                )
                return candidates, depth
            except (PyseriniRequestError, *_TRANSPORT_ERRORS):
                continue
        raise PyseriniRequestError(
            f"search failed at every depth from {hits} down to {min_hits}"
        )

    def search(
        self, query: str, hits: int = 10, max_attempts: int | None = None
    ) -> list[SearchCandidate]:
        cache_path = self._cache_path("search", f"{query}|{hits}")
        payload: Any = None
        if cache_path and cache_path.is_file():
            started = time.perf_counter()
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                self.stats["cache_hits"] += 1
                self.stats["cache_seconds"] += time.perf_counter() - started
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload is None:
            started = time.perf_counter()
            payload = self._get_json(
                f"{self.base_url}/v1/{self.index}/search",
                params={"query": query, "hits": hits},
                max_attempts=max_attempts,
            )
            self.stats["cache_misses"] += 1
            self.stats["api_seconds"] += time.perf_counter() - started
            if cache_path:
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(cache_path)

        candidates = []
        for entry in payload.get("candidates", []):
            candidates.append(
                SearchCandidate(
                    docid=entry["docid"],
                    rank=entry.get("rank", len(candidates) + 1),
                    score=float(entry.get("score", 0.0)),
                    text=_extract_text(entry.get("doc")),
                )
            )
        return candidates

    def fetch_doc(self, docid: str) -> str | None:
        payload = self._get_json(f"{self.base_url}/v1/{self.index}/doc/{docid}")
        return _extract_text(payload.get("doc"))

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.status_code == 200:
            return
        if response.status_code in (401, 403):
            raise PyseriniAuthError(
                "Pyserini API rejected the request (missing/expired/invalid token). "
                "Request a token from get-pyserini@googlegroups.com."
            )
        raise PyseriniRequestError(
            f"Pyserini API request failed: HTTP {response.status_code} {response.text[:200]!r}"
        )
