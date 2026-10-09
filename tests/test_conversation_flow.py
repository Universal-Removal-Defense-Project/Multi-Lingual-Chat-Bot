"""Multi-message conversations and renaming in the chat UI."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import backend

APP = str(Path(__file__).resolve().parent.parent / "ui.py")


@pytest.fixture
def app(monkeypatch):
    """The chat app with a fake model that answers "ok" and counts replies."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    replies = []

    def fake_stream(messages, **kwargs):
        replies.append(messages[-1]["content"])
        yield "ok"

    monkeypatch.setattr(backend, "stream_response", fake_stream)
    at = AppTest.from_file(APP).run(timeout=30)
    assert not at.exception, at.exception
    at.replies = replies
    return at


def send(at, text):
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception, at.exception


def active(at):
    return at.session_state["conversations"][at.session_state["active_id"]]


def test_every_message_in_a_new_chat_is_kept_and_answered(app):
    # Regression: the rename box used to revert the auto-title on the next run,
    # which dropped every second message (2, 4, 6, ...) without a reply.
    sent = [f"message {i}" for i in range(1, 7)]
    for text in sent:
        send(app, text)
    conversation = active(app)
    assert [m["content"] for m in conversation["messages"] if m["role"] == "user"] == sent
    assert app.replies == sent
    assert conversation["title"] == "message 1"


def test_manual_rename_shows_at_once_and_survives_new_messages(app):
    send(app, "How do I apply for asylum?")
    cid = app.session_state["active_id"]
    app.text_input(key=f"rename_{cid}").set_value("Housing case").run(timeout=30)
    assert "Housing case" in [b.label for b in app.button]
    send(app, "What documents do I need?")
    assert active(app)["title"] == "Housing case"
    assert len(app.replies) == 2


def test_clearing_the_rename_box_keeps_the_title(app):
    send(app, "How do I apply for asylum?")
    cid = app.session_state["active_id"]
    app.text_input(key=f"rename_{cid}").set_value("   ").run(timeout=30)
    assert active(app)["title"] == "How do I apply for asylum?"
    assert app.text_input(key=f"rename_{cid}").value == "How do I apply for asylum?"
