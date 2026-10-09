"""Language behaviour in the chat UI (Issues #36, #37, #39)."""

from pathlib import Path
from types import SimpleNamespace

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


# --- #37: the sidebar shows the outcome of the latest detection ---

@pytest.fixture
def detector(monkeypatch):
    """Fake detect_language: returns (or raises) ``result`` and records calls."""
    fake = SimpleNamespace(result="Chinese", calls=[])

    def detect(text, *, api_key=None):
        fake.calls.append(text)
        if isinstance(fake.result, Exception):
            raise fake.result
        return fake.result

    monkeypatch.setattr(backend, "detect_language", detect)
    return fake


def app_with_auto_detect():
    at = run_app()
    auto_detect_toggle(at).set_value(True).run(timeout=30)
    return at


def sidebar_text(at):
    return " ".join(caption.value for caption in at.sidebar.caption)


def active(at):
    return at.session_state["conversations"][at.session_state["active_id"]]


def test_detected_language_shows_in_the_same_run(replies, detector):
    at = app_with_auto_detect()
    send(at, "你好，我想问一下签证的问题")
    assert "Last detected: Chinese" in sidebar_text(at)
    assert at.selectbox(key="lang_select").value == "中文"
    assert active(at)["language"] == "Chinese"


@pytest.mark.parametrize(
    "result, message",
    [
        (None, "Couldn't detect the language of your last message — kept English"),
        (backend.BackendError("down"), "Language detection is unavailable right now — kept English"),
    ],
)
def test_failed_detection_is_shown_not_hidden(replies, detector, result, message):
    detector.result = result
    at = app_with_auto_detect()
    send(at, "something ambiguous here")
    assert message in sidebar_text(at)
    assert active(at)["language"] == "English"


def test_short_messages_skip_detection_and_keep_language(replies, detector):
    at = app_with_auto_detect()
    send(at, "你好，我想问一下签证的问题")
    send(at, "OK")
    assert detector.calls == ["你好，我想问一下签证的问题"]
    assert active(at)["language"] == "Chinese"
    assert "Last detected: Chinese" in sidebar_text(at)


def test_choosing_a_language_by_hand_resets_the_status(replies, detector):
    at = app_with_auto_detect()
    send(at, "你好，我想问一下签证的问题")
    auto_detect_toggle(at).set_value(False).run(timeout=30)
    at.selectbox(key="lang_select").set_value("Español").run(timeout=30)
    auto_detect_toggle(at).set_value(True).run(timeout=30)
    assert "now: Spanish" in sidebar_text(at)
    assert "Last detected" not in sidebar_text(at)


def test_right_to_left_layout_switches_in_the_same_run(replies, detector):
    detector.result = "Arabic"
    at = app_with_auto_detect()
    send(at, "أحتاج مساعدة في طلب اللجوء")
    assert "direction: rtl" in " ".join(m.value for m in at.markdown)


def test_new_chat_title_shows_in_the_sidebar_at_once(replies):
    at = run_app()
    send(at, "How do I apply for asylum?")
    assert "How do I apply for asylum?" in [b.label for b in at.button]
