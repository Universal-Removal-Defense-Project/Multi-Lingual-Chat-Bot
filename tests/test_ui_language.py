"""Language behaviour in the chat UI (Issues #36, #37, #39)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import backend

APP = str(Path(__file__).resolve().parent.parent / "ui.py")


def run_app():
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception, at.exception
    return at


def auto_detect_toggle(at):
    return next(t for t in at.toggle if "Auto-detect" in (t.label or ""))


def send(at, text):
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception, at.exception
    return at


@pytest.fixture
def replies(monkeypatch):
    """Use a fake model: record each request's system prompt, reply with "ok"."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    system_prompts = []

    def fake_stream(messages, **kwargs):
        system_prompts.append(messages[0]["content"])
        yield "ok"

    monkeypatch.setattr(backend, "stream_response", fake_stream)
    return system_prompts


# --- #36: the reply prompt follows the auto-detect setting ---

@pytest.mark.parametrize(
    "auto, expected",
    [(True, "not the pasted text"), (False, "Always respond in")],
)
def test_reply_prompt_matches_auto_detect_setting(replies, auto, expected):
    at = run_app()
    auto_detect_toggle(at).set_value(auto).run(timeout=30)
    send(at, "请帮我翻译一下这段话：You must appear for your hearing.")
    assert expected in replies[-1]
