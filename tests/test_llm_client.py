import unittest
from unittest.mock import MagicMock, patch

import requests

from trecrag.clients.llm_client import OpenAICompatibleClient


def _fake_response(content: str):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


class TestOpenAICompatibleClient(unittest.TestCase):
    def test_complete_posts_expected_payload_and_parses_response(self):
        client = OpenAICompatibleClient(base_url="https://example.test/v1", api_key="secret-key", model="my-model")
        with patch("trecrag.clients.llm_client.requests.post", return_value=_fake_response("hello")) as mock_post:
            result = client.complete("a prompt", max_tokens=42)

        self.assertEqual(result, "hello")
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer secret-key"})
        self.assertEqual(kwargs["json"]["model"], "my-model")
        self.assertEqual(kwargs["json"]["messages"], [{"role": "user", "content": "a prompt"}])
        self.assertEqual(kwargs["json"]["max_tokens"], 42)

    def test_strips_reasoning_block_from_response(self):
        client = OpenAICompatibleClient(base_url="https://example.test/v1", api_key="key", model="m")
        with patch(
            "trecrag.clients.llm_client.requests.post",
            return_value=_fake_response("<think>maybe 5 then 9</think>[2] > [1]"),
        ):
            result = client.complete("q")
        self.assertEqual(result, "[2] > [1]")

    def test_retries_on_request_exception_then_succeeds(self):
        client = OpenAICompatibleClient(
            base_url="https://example.test/v1", api_key="key", model="m", max_attempts=3, retry_backoff=0.0
        )
        side_effects = [requests.exceptions.ConnectionError("reset"), _fake_response("ok")]
        with patch("trecrag.clients.llm_client.requests.post", side_effect=side_effects), \
             patch("trecrag.clients.llm_client.time.sleep") as mock_sleep:
            result = client.complete("q")
        self.assertEqual(result, "ok")
        mock_sleep.assert_called_once()

    def test_retries_on_malformed_payload_then_succeeds(self):
        """A truncated/malformed response (missing expected keys) is as fatal to
        the topic as a dropped connection, so it must be retried too."""
        client = OpenAICompatibleClient(
            base_url="https://example.test/v1", api_key="key", model="m", max_attempts=3, retry_backoff=0.0
        )
        malformed = MagicMock()
        malformed.raise_for_status.return_value = None
        malformed.json.return_value = {"choices": [{}]}  # missing "message" -> KeyError on access
        side_effects = [malformed, _fake_response("ok")]
        with patch("trecrag.clients.llm_client.requests.post", side_effect=side_effects), \
             patch("trecrag.clients.llm_client.time.sleep"):
            result = client.complete("q")
        self.assertEqual(result, "ok")

    def test_raises_after_max_attempts_exhausted(self):
        client = OpenAICompatibleClient(
            base_url="https://example.test/v1", api_key="key", model="m", max_attempts=2, retry_backoff=0.0
        )
        with patch(
            "trecrag.clients.llm_client.requests.post",
            side_effect=requests.exceptions.Timeout("read timed out"),
        ), patch("trecrag.clients.llm_client.time.sleep"):
            with self.assertRaises(RuntimeError):
                client.complete("q")


if __name__ == "__main__":
    unittest.main()
