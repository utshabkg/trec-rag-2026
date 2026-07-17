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

from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.index = index
        self._token = token
        self.timeout_seconds = timeout_seconds
        self.session = session or self._build_session(max_retries)

    @staticmethod
    def _build_session(max_retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
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

    def search(self, query: str, hits: int = 10) -> list[SearchCandidate]:
        response = self.session.get(
            f"{self.base_url}/v1/{self.index}/search",
            params={"query": query, "hits": hits},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response)
        payload = response.json()
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
        response = self.session.get(
            f"{self.base_url}/v1/{self.index}/doc/{docid}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        self._raise_for_status(response)
        payload = response.json()
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
