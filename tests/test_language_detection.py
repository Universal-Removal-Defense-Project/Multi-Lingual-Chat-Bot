"""Language detection in the backend (Issues #17, #36)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import backend


class FakeClient:
    """Stands in for the Groq client: records the request, returns a fixed reply."""

    def __init__(self, reply):
        self.reply = reply
        self.request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def fake_client():
    client = FakeClient("Chinese")
    with patch.object(backend, "get_client", lambda api_key=None: client):
        yield client


# --- Detection model settings ---

def test_gpt_oss_detection_keeps_reasoning_short_and_out_of_reply(fake_client, monkeypatch):
    monkeypatch.setattr(backend, "DETECT_MODEL", "openai/gpt-oss-20b")
    backend.detect_language("你好，我想问一下签证的问题", api_key="x")
    request = fake_client.request
    assert request["model"] == "openai/gpt-oss-20b"
    assert request["reasoning_effort"] == "low"
    assert request["include_reasoning"] is False
    # Reasoning tokens count against this budget; a handful would return nothing.
    assert request["max_completion_tokens"] >= 64


def test_other_detection_models_get_no_reasoning_options(fake_client, monkeypatch):
    monkeypatch.setattr(backend, "DETECT_MODEL", "some-provider/plain-model")
    backend.detect_language("你好，我想问一下签证的问题", api_key="x")
    assert "reasoning_effort" not in fake_client.request
    assert "include_reasoning" not in fake_client.request
