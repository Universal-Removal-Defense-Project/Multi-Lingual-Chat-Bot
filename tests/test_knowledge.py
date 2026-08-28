"""Tests for the RAG retrieval and knowledge base (Milestone 6).

Network-free and PDF-free: PDF extraction is patched so the store/chunk/search/
delete/reindex flow is tested deterministically without generating real PDFs.
"""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import backend
import knowledge
import rag

# Absolute path to the repo root so AppTest.from_file resolves regardless of the
# Streamlit version's relative-path behaviour.
APP_DIR = Path(__file__).resolve().parent.parent


# --- rag.py (#24) ---

def test_chunk_text():
    assert rag.chunk_text("") == []
    assert rag.chunk_text("one paragraph") == ["one paragraph"]
    chunks = rag.chunk_text("para one\n\npara two\n\n" + "x " * 500)
    assert len(chunks) >= 2


def test_tfidf_ranks_relevant_chunk_first():
    index = rag.TfidfIndex(
        ["the cat sat on the mat", "dogs run very fast", "a cat and a dog play"]
    )
    results = index.search("cat", k=2)
    assert results, "expected at least one hit"
    assert results[0][0] in (0, 2)  # a chunk mentioning 'cat'
    assert index.search("zzzzz") == []  # no overlap -> no hits


# --- knowledge.py (#23, #24, #26) ---

FAKE_PAGES = [
    (1, "Asylum applications must generally be filed within one year of arrival."),
    (2, "Employment authorization requires filing Form I-765 with USCIS."),
]


def test_add_search_reindex_delete(tmp_path):
    db = tmp_path / "knowledge.db"
    with patch.object(knowledge, "extract_pages", lambda data: FAKE_PAGES):
        result = knowledge.add_pdf("handbook.pdf", b"%PDF-fake", path=db)
    assert result["status"] == "ready" and result["num_chunks"] >= 2
    assert knowledge.has_documents(db)

    docs = knowledge.list_documents(db)
    assert len(docs) == 1 and docs[0]["filename"] == "handbook.pdf"

    hits = knowledge.search("asylum one year arrival", k=2, path=db)
    assert hits and hits[0]["filename"] == "handbook.pdf"
    assert "asylum" in hits[0]["text"].lower()
    assert hits[0]["page"] == 1

    # A query about the other topic retrieves the other page.
    perm_hits = knowledge.search("employment authorization form", k=1, path=db)
    assert perm_hits and perm_hits[0]["page"] == 2

    with patch.object(knowledge, "extract_pages", lambda data: FAKE_PAGES):
        reindexed = knowledge.reindex_document(docs[0]["id"], path=db)
    assert reindexed["status"] == "ready"

    knowledge.delete_document(docs[0]["id"], path=db)
    assert not knowledge.has_documents(db)
    assert knowledge.list_documents(db) == []


def test_failed_pdf_is_recorded_not_raised(tmp_path):
    db = tmp_path / "knowledge.db"
    with patch.object(knowledge, "extract_pages", lambda data: []):
        result = knowledge.add_pdf("scanned.pdf", b"x", path=db)
    assert result["status"] == "failed" and result["num_chunks"] == 0
    assert not knowledge.has_documents(db)


# --- backend RAG prompt (#25) ---

def test_build_messages_with_context_and_sanitises_history():
    history = [{"role": "user", "content": "q", "sources": [{"filename": "a.pdf"}]}]
    msgs = backend.build_messages(history, "English", context=["[a.pdf p.1] text"])
    assert any("Sources:" in m["content"] for m in msgs if m["role"] == "system")
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert set(user_msg.keys()) == {"role", "content"}  # extra 'sources' stripped


# --- Knowledge Base admin page (#26) ---

def test_kb_page_renders_when_empty():
    at = AppTest.from_file(str(APP_DIR / "pages" / "Knowledge_Base.py")).run(timeout=30)
    assert not at.exception, at.exception
    assert any("No documents yet" in (m.value or "") for m in at.info)


def test_chat_grounds_answer_in_sources(monkeypatch):
    """End-to-end #25: a question with a loaded doc gets sources attached."""
    with patch.object(knowledge, "extract_pages", lambda data: FAKE_PAGES):
        knowledge.add_pdf("handbook.pdf", b"x")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def fake_stream(messages, **kwargs):
        assert any("Sources:" in m["content"] for m in messages if m["role"] == "system")
        yield "Asylum must be filed within one year [handbook.pdf p.1]."

    monkeypatch.setattr(backend, "stream_response", fake_stream)

    at = AppTest.from_file(str(APP_DIR / "ui.py")).run(timeout=30)
    at.chat_input[0].set_value("what is the asylum deadline?").run(timeout=30)
    assert not at.exception, at.exception

    cid = at.session_state["active_id"]
    assistant = [m for m in at.session_state["conversations"][cid]["messages"]
                 if m["role"] == "assistant"][-1]
    assert assistant.get("sources"), "expected sources attached to the answer"
    assert assistant["sources"][0]["filename"] == "handbook.pdf"
