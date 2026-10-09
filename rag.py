"""Lightweight, dependency-free retrieval (Milestone 6 / Issue #24).

Pure logic: no Streamlit, no database, no network. Text is split into chunks and
ranked against a query with TF-IDF cosine similarity. This is intentionally
lightweight (offline, testable, no extra API key). The interface — ``chunk_text``
and ``TfidfIndex`` — is what the rest of the app depends on, so the ranking can
later be swapped for semantic embeddings without touching callers.

Tokenization works for every supported language using only the standard
library (#35):
  - A word is a run of letters, digits and combining marks in any script, so
    accented words ("expulsión") and Indic words ("हिन्दी") stay whole.
  - Scripts written without spaces between words (Chinese, Japanese, Korean,
    Thai, ...) are split into overlapping two-character pieces.
  - Words with diacritics are also indexed without them, so "expulsion" still
    finds "expulsión" and unvowelled Arabic finds vowelled Arabic.
  - Text is NFKC-normalised and case-folded, and digits in any script become
    ASCII, so a form number like I-589 matches however it is written.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache


def _code_point_ranges(predicate, limit: int = 0x20000):
    """Yield (first, last) ranges of code points below ``limit`` matching predicate.

    Planes 0–1 hold every combining mark and decimal digit used in text.
    """
    start = None
    for cp in range(limit + 1):
        if cp < limit and predicate(chr(cp)):
            if start is None:
                start = cp
        elif start is not None:
            yield start, cp - 1
            start = None


def _is_combining_mark(ch: str) -> bool:
    # Variation selectors are marks too, but they only style emoji and symbols.
    return unicodedata.category(ch)[0] == "M" and not "︀" <= ch <= "️"


# Combining marks (Indic vowel signs, Arabic harakat, ...) as a regex class body.
_MARKS = "".join(f"\\U{a:08x}-\\U{b:08x}" for a, b in _code_point_ranges(_is_combining_mark))

# Scripts written without spaces between words; their runs are split into bigrams.
_NO_SPACE = (
    "฀-໿"          # Thai, Lao
    "က-႟"          # Myanmar
    "ᄀ-ᇿ"          # Hangul Jamo
    "ក-៿"          # Khmer
    "々-〇"          # 々 〆 〇
    "぀-ヿ"          # Hiragana, Katakana
    "㄰-㆏"          # Hangul Compatibility Jamo
    "ㇰ-ㇿ"          # Katakana Phonetic Extensions
    "㐀-䶿"          # CJK Unified Ideographs Extension A
    "一-鿿"          # CJK Unified Ideographs
    "가-힯"          # Hangul Syllables
    "豈-﫿"          # CJK Compatibility Ideographs
    "\U00020000-\U0003ffff"  # CJK Unified Ideographs Extension B and later
)

# A word starts with a letter or digit, then continues with letters, digits or marks.
_WORD_RE = re.compile(f"[^\\W_](?:[^\\W_]|[{_MARKS}])*")
_SCRIPT_RUN_RE = re.compile(f"[{_NO_SPACE}]+|[^{_NO_SPACE}]+")
_NO_SPACE_RE = re.compile(f"[{_NO_SPACE}]")
_CHARACTER_RE = re.compile(f".[{_MARKS}]*", re.DOTALL)  # a letter plus its marks

# Applied after NFKC: digits in any script become ASCII ("٥٨٩" -> "589"), and
# invisible characters that can sit inside words are removed (soft hyphens in
# PDFs, zero-width joiners in Persian, Arabic tatweel).
_TRANSLATE = {
    cp: str(unicodedata.decimal(chr(cp)))
    for first, last in _code_point_ranges(lambda ch: unicodedata.category(ch) == "Nd")
    for cp in range(first, last + 1)
}
_TRANSLATE.update(dict.fromkeys(map(ord, "­ـ‌‍⁠﻿")))


def tokenize(text: str) -> list[str]:
    """Split text into search tokens for any supported language (see module doc)."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).casefold()
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        if word.isascii():  # fast path: nothing to split or strip
            tokens.append(word)
            continue
        for run in _SCRIPT_RUN_RE.findall(word):
            if _NO_SPACE_RE.match(run):
                tokens.extend(_bigrams(run))
            else:
                tokens.append(run)
                bare = _strip_diacritics(run)
                if bare and bare != run:
                    tokens.append(bare)
    return tokens


def _bigrams(run: str) -> list[str]:
    """Overlapping two-character pieces of a run, keeping marks with their letter."""
    chars = _CHARACTER_RE.findall(run)
    if len(chars) == 1:
        return chars
    return [a + b for a, b in zip(chars, chars[1:])]


@lru_cache(maxsize=65536)
def _strip_diacritics(word: str) -> str:
    """Drop diacritics ("expulsión" -> "expulsion"); Indic vowel signs stay."""
    if word.isascii():
        return word
    decomposed = unicodedata.normalize("NFD", word)
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", bare)


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
