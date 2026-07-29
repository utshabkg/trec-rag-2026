"""Pyserini client transport-resilience tests.

A full run makes 119 requests whose bodies are ~60 MB each. One of them dropped
mid-body on a real run (`ChunkedEncodingError`), which the adapter's status-code
Retry cannot see because the status line had already been accepted. Losing a
narrative to a transient reset is expensive and easy to miss, so the retry path
is pinned here.
"""

import json
import unittest

import requests

from trecrag.clients.pyserini_client import (
    PyseriniAuthError,
    PyseriniClient,
    PyseriniRequestError,
)


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


_OK = {"candidates": [{"docid": "shard_00001_2", "rank": 1, "score": 1.0, "doc": "text"}]}


class _ScriptedSession:
    """Raises/returns whatever the script says, one entry per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0) if self.script else _Response(_OK)
        if isinstance(item, Exception):
            raise item
        return item


def _client(session, max_retries=3):
    return PyseriniClient(
        base_url="http://example.invalid",
        index="climbmix-400b",
        token="dummy",
        max_retries=max_retries,
        backoff_factor=0.0,  # keep the test fast; backoff logic is unrelated
        session=session,
    )


class TestTransportRetries(unittest.TestCase):
    def test_recovers_from_a_body_read_reset(self):
        session = _ScriptedSession(
            [
                requests.exceptions.ChunkedEncodingError("Connection broken"),
                requests.exceptions.ChunkedEncodingError("Connection broken"),
                _Response(_OK),
            ]
        )
        candidates = _client(session).search("q", hits=1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(session.calls, 3)

    def test_recovers_from_connection_error_and_timeout(self):
        for error in (
            requests.exceptions.ConnectionError("reset"),
            requests.exceptions.Timeout("slow"),
        ):
            session = _ScriptedSession([error, _Response(_OK)])
            self.assertEqual(len(_client(session).search("q", hits=1)), 1)

    def test_gives_up_after_exhausting_retries(self):
        session = _ScriptedSession(
            [requests.exceptions.ChunkedEncodingError("broken")] * 10
        )
        with self.assertRaises(PyseriniRequestError):
            _client(session, max_retries=2).search("q", hits=1)
        self.assertEqual(session.calls, 3)  # initial attempt + 2 retries

    def test_auth_failure_is_not_retried(self):
        """A rejected token will be rejected again; burning retries on all 119
        narratives just delays a failure that needs a human."""
        session = _ScriptedSession([_Response({"error": "Unauthorized."}, 401)] * 5)
        with self.assertRaises(PyseriniAuthError):
            _client(session).search("q", hits=1)
        self.assertEqual(session.calls, 1)

    def test_successful_request_makes_exactly_one_call(self):
        session = _ScriptedSession([_Response(_OK)])
        _client(session).search("q", hits=1)
        self.assertEqual(session.calls, 1)


if __name__ == "__main__":
    unittest.main()
