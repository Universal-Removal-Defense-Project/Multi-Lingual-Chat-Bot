"""Knowledge base: PDF documents and their chunks, stored in SQLite.

Milestone 6:
  #23 PDF upload + text extraction (with failed-PDF handling)
  #24 chunk storage + retrieval (ranking delegated to rag.py)
  #26 document management (list / delete / re-index / status)

No Streamlit dependency. SQLite is the structured-storage foundation that
Milestone 9 (intake submissions) will reuse. The original PDF bytes are stored
so a document can be re-indexed after chunking changes.
"""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

import rag

DB_PATH = Path(__file__).parent / "data" / "knowledge.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL,
            num_chunks INTEGER NOT NULL DEFAULT 0,
            data BLOB NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            page INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )"""
    )
    return conn


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that have extractable text (#23)."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def _index_document(conn: sqlite3.Connection, doc_id: int, data: bytes) -> tuple[str, int]:
    """Extract, chunk and store chunks for a document. Returns (status, num_chunks)."""
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    try:
        pages = extract_pages(data)
    except Exception:
        pages = []
    if not pages:
        return "failed", 0  # unreadable / no extractable text (#23 failed PDFs)
    count = 0
    for page_number, text in pages:
        for chunk in rag.chunk_text(text):
            conn.execute(
                "INSERT INTO chunks (doc_id, page, text) VALUES (?, ?, ?)",
                (doc_id, page_number, chunk),
            )
            count += 1
    return "ready", count


def add_pdf(filename: str, data: bytes, path: Path = DB_PATH) -> dict:
    """Store a PDF, extract and index it. Failed extraction is recorded, not raised."""
    conn = _connect(path)
    try:
        cur = conn.execute(
            "INSERT INTO documents (filename, uploaded_at, status, data) VALUES (?, ?, ?, ?)",
            (filename, _now(), "processing", data),
        )
        doc_id = cur.lastrowid
        status, count = _index_document(conn, doc_id, data)
        conn.execute(
            "UPDATE documents SET status = ?, num_chunks = ? WHERE id = ?",
            (status, count, doc_id),
        )
        conn.commit()
        return {"id": doc_id, "filename": filename, "status": status, "num_chunks": count}
    finally:
        conn.close()


def list_documents(path: Path = DB_PATH) -> list[dict]:
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT id, filename, uploaded_at, status, num_chunks FROM documents ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "filename": r[1], "uploaded_at": r[2], "status": r[3], "num_chunks": r[4]}
        for r in rows
    ]


def delete_document(doc_id: int, path: Path = DB_PATH) -> None:
    conn = _connect(path)
    try:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def reindex_document(doc_id: int, path: Path = DB_PATH) -> dict:
    """Re-extract and re-chunk a stored document from its saved PDF bytes (#26)."""
    conn = _connect(path)
    try:
        row = conn.execute("SELECT data FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise KeyError(f"No document with id {doc_id}")
        status, count = _index_document(conn, doc_id, row[0])
        conn.execute(
            "UPDATE documents SET status = ?, num_chunks = ? WHERE id = ?",
            (status, count, doc_id),
        )
        conn.commit()
        return {"id": doc_id, "status": status, "num_chunks": count}
    finally:
        conn.close()


def has_documents(path: Path = DB_PATH) -> bool:
    conn = _connect(path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    finally:
        conn.close()
    return bool(row[0])


def search(query: str, k: int = 4, path: Path = DB_PATH) -> list[dict]:
    """Return the top-k most relevant chunks with their source (#24, #25)."""
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT c.text, c.page, d.filename FROM chunks c JOIN documents d ON c.doc_id = d.id"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return []
    index = rag.TfidfIndex([r[0] for r in rows])
    results = index.search(query, k)
    return [
        {"text": rows[i][0], "page": rows[i][1], "filename": rows[i][2], "score": score}
        for i, score in results
    ]
