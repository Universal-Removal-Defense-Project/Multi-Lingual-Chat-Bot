"""Automated test suite (Milestone 5 / Issue #21).

These tests are network-free: the Groq client is mocked wherever a call would
happen, so they run in CI without an API key. App-level tests use Streamlit's
AppTest to execute ui.py headlessly.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

import backend
import storage
import styles

APP_PATH = Path(__file__).resolve().parents[1] / "ui.py"


def _run(at):
    at.run(timeout=30)
    assert not at.exception, at.exception
    return at


def _fake_client(reply_text):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply_text))]
        )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# --- Language model (M1, M4 / #16, #18) ---

def test_languages_count_and_alphabetical():
    assert len(backend.LANGUAGES) >= 25
    names = [lang["name"] for lang in backend.LANGUAGES]
    assert names == sorted(names)
    assert all(lang.get("code") and "rtl" in lang for lang in backend.LANGUAGES)


def test_is_rtl():
    assert backend.is_rtl("Arabic") and backend.is_rtl("Urdu")
    assert not backend.is_rtl("English")


def test_system_prompt_pins_language():
    assert "Spanish" in backend.build_system_prompt("Spanish")


def test_build_messages_prepends_system():
    msgs = backend.build_messages([{"role": "user", "content": "hi"}], "French")
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(backend.BackendError):
        backend.resolve_api_key(None)


# --- Detection & streaming (mocked, M4 #17 / M1 #2) ---

def test_detect_language(monkeypatch):
    with patch.object(backend, "get_client", lambda api_key=None: _fake_client("Arabic")):
        assert backend.detect_language("مرحبا", api_key="x") == "Arabic"
    with patch.object(backend, "get_client", lambda api_key=None: _fake_client("Klingon")):
        assert backend.detect_language("nuqneH", api_key="x") is None
    assert backend.detect_language("   ", api_key="x") is None


def test_stream_response_assembles_and_filters_empty():
    def create(**kwargs):
        for tok in ["Hel", "lo", None]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=tok))])
    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: create()))
    )
    with patch.object(backend, "get_client", lambda api_key=None: fake):
        out = "".join(backend.stream_response([{"role": "user", "content": "x"}], api_key="x"))
    assert out == "Hello"


# --- Styles (M3 #10, M4 #18) ---

def test_theme_and_rtl_css():
    assert "<style>" in styles.theme_css("dark")
    assert "<style>" in styles.theme_css("light")
    assert "direction: rtl" in styles.rtl_css(True)
    assert styles.rtl_css(False) == ""


# --- Storage (M2 #5) ---

def test_storage_roundtrip(tmp_path):
    path = tmp_path / "conversations.json"
    conv = storage.new_conversation("Spanish")
    conv["messages"].append({"role": "user", "content": "hola"})
    storage.save_conversations({conv["id"]: conv}, path)
    loaded = storage.load_conversations(path)
    assert loaded[conv["id"]]["language"] == "Spanish"
    assert loaded[conv["id"]]["messages"][0]["content"] == "hola"
    assert storage.load_conversations(tmp_path / "missing.json") == {}


# --- App smoke tests via AppTest (M1–M4) ---

def test_app_renders_with_one_conversation():
    at = _run(AppTest.from_file(APP_PATH))
    assert len(at.session_state["conversations"]) == 1
    assert len(at.selectbox[0].options) >= 25
    assert any("Auto-detect" in (t.label or "") for t in at.toggle)


def test_rename_reflected_in_sidebar_list():
    at = _run(AppTest.from_file(APP_PATH))
    cid = at.session_state["active_id"]
    at.text_input(key=f"rename_{cid}").set_value("Housing case")
    _run(at)
    assert "Housing case" in [b.label for b in at.button]


def test_language_switch_toggles_rtl():
    at = _run(AppTest.from_file(APP_PATH))
    at.selectbox[0].set_value("العربية")
    _run(at)
    assert "direction: rtl" in " ".join(m.value for m in at.markdown)
    at.selectbox[0].set_value("English")
    _run(at)
    assert "direction: rtl" not in " ".join(m.value for m in at.markdown)


def test_new_chat_and_delete_flow():
    at = _run(AppTest.from_file(APP_PATH))
    start = len(at.session_state["conversations"])
    for b in at.button:
        if b.label and "New chat" in b.label:
            b.click(); _run(at); break
    assert len(at.session_state["conversations"]) == start + 1
    for b in at.button:
        if b.label == "🗑":
            b.click(); _run(at); break
    target = at.session_state["pending_delete"]
    for b in at.button:
        if b.label == "Delete":
            b.click(); _run(at); break
    assert target not in at.session_state["conversations"]
