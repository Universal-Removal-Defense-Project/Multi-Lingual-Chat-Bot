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


# --- #36: judge the user's own words, not pasted text ---

def test_detector_is_told_to_ignore_pasted_text(fake_client):
    pasted = "请帮我翻译一下这段话：\n\nYou must appear for your hearing."
    backend.detect_language(pasted, api_key="x")
    system, user = fake_client.request["messages"]
    assert "Ignore any text they paste" in system["content"]
    assert user["content"] == f"<message>\n{pasted}\n</message>"


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Chinese", "Chinese"),
        ("Chinese.", "Chinese"),
        ("  chinese\n", "Chinese"),
        ("Chinese (Simplified)", "Chinese"),
        ("Persian", "Farsi"),
        ("Filipino", "Tagalog"),
        ("Haitian Creole", "Haitian Creole"),
        ("Unknown", None),
        ("", None),
        ("Klingon", None),
        ("English and Chinese", None),  # ambiguous: keep the current language
    ],
)
def test_detector_reply_is_matched_to_one_supported_language(reply, expected):
    with patch.object(backend, "get_client", lambda api_key=None: FakeClient(reply)):
        assert backend.detect_language("some message", api_key="x") == expected


@pytest.mark.parametrize(
    "text, enough",
    [("OK", False), ("123", False), ("...", False), ("👍", False), ("", False),
     ("Hola", True), ("你好吗", True), ("yes", True)],
)
def test_short_messages_have_no_language_signal(text, enough):
    assert backend.has_language_signal(text) is enough


def test_short_messages_are_not_sent_to_the_detector(fake_client):
    assert backend.detect_language("OK", api_key="x") is None
    assert fake_client.request is None  # no API call made


def test_auto_detected_prompt_follows_the_users_own_words():
    prompt = backend.build_system_prompt("English", auto_detected=True)
    assert "own questions and instructions" in prompt
    assert "not the pasted text" in prompt
    assert "Always respond in English" not in prompt  # English is only the fallback


def test_manually_chosen_language_is_still_pinned():
    prompt = backend.build_system_prompt("Spanish")
    assert "Always respond in Spanish" in prompt


def test_build_messages_passes_the_auto_detect_mode():
    auto = backend.build_messages([], "English", auto_detected=True)
    manual = backend.build_messages([], "English")
    assert "not the pasted text" in auto[0]["content"]
    assert "Always respond in English" in manual[0]["content"]
