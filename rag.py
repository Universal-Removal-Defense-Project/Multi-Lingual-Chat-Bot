"""Lightweight, dependency-free retrieval (Milestone 6 / Issue #24).

Pure logic: no Streamlit, no database, no network. Text is split into chunks and
ranked against a query with TF-IDF cosine similarity. This is intentionally
lightweight (offline, testable, no extra API key). The interface — ``chunk_text``
and ``TfidfIndex`` — is what the rest of the app depends on, so the ranking can
later be swapped for semantic embeddings without touching callers.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Split text into chunks of roughly ``max_chars``, on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 1 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    # Fall back to a single chunk if there were no blank-line paragraphs.
    return chunks or ([text.strip()] if text.strip() else [])


class TfidfIndex:
    """In-memory TF-IDF index over a list of chunk texts."""

    def __init__(self, docs: list[str]):
        self.docs = docs
        tokens = [tokenize(d) for d in docs]
        df: Counter = Counter()
        for toks in tokens:
            df.update(set(toks))
        n = len(docs) or 1
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = [self._vector(toks) for toks in tokens]
        self.norms = [math.sqrt(sum(w * w for w in v.values())) for v in self.vectors]

    def _vector(self, toks: list[str]) -> dict[str, float]:
        return {t: c * self.idf.get(t, 0.0) for t, c in Counter(toks).items()}

    def search(self, query: str, k: int = 4) -> list[tuple[int, float]]:
        """Return up to ``k`` (doc_index, score) pairs with score > 0, best first."""
        q = self._vector(tokenize(query))
        q_norm = math.sqrt(sum(w * w for w in q.values()))
        if q_norm == 0:
            return []
        scored = []
        for i, (vec, norm) in enumerate(zip(self.vectors, self.norms)):
            if norm == 0:
                continue
            dot = sum(w * vec.get(t, 0.0) for t, w in q.items())
            score = dot / (q_norm * norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
