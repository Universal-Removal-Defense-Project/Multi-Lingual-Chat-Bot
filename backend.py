"""Groq chat backend: prompt construction and chat completion.

This module has no Streamlit dependency. The UI lives in ui.py and persistence
in storage.py. ``stream_response`` yields response text incrementally so the UI
can render replies as they are generated.
"""

from __future__ import annotations

import os
from typing import Iterable, Iterator

from groq import Groq

# Default model: Groq Llama 3.3 70B. Override with the GROQ_MODEL env var.
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
# Small, fast model used only for language detection (Issue #17).
DETECT_MODEL = os.environ.get("GROQ_DETECT_MODEL", "llama-3.1-8b-instant")

# Supported languages (Issue #16). Each entry carries:
#   label — native name shown in the dropdown
#   name  — English name the model is instructed to reply in
#   code  — ISO 639-1 code
#   rtl   — True for right-to-left scripts (Issue #18)
# Ordered alphabetically by English name so the dropdown is organised.
LANGUAGES: list[dict] = [
    {"label": "العربية", "name": "Arabic", "code": "ar", "rtl": True},
    {"label": "বাংলা", "name": "Bengali", "code": "bn", "rtl": False},
    {"label": "中文", "name": "Chinese", "code": "zh", "rtl": False},
    {"label": "Nederlands", "name": "Dutch", "code": "nl", "rtl": False},
    {"label": "English", "name": "English", "code": "en", "rtl": False},
    {"label": "فارسی", "name": "Farsi", "code": "fa", "rtl": True},
    {"label": "Français", "name": "French", "code": "fr", "rtl": False},
    {"label": "Deutsch", "name": "German", "code": "de", "rtl": False},
    {"label": "ગુજરાતી", "name": "Gujarati", "code": "gu", "rtl": False},
    {"label": "Kreyòl Ayisyen", "name": "Haitian Creole", "code": "ht", "rtl": False},
    {"label": "עברית", "name": "Hebrew", "code": "he", "rtl": True},
    {"label": "हिन्दी", "name": "Hindi", "code": "hi", "rtl": False},
    {"label": "Italiano", "name": "Italian", "code": "it", "rtl": False},
    {"label": "日本語", "name": "Japanese", "code": "ja", "rtl": False},
    {"label": "한국어", "name": "Korean", "code": "ko", "rtl": False},
    {"label": "پښتو", "name": "Pashto", "code": "ps", "rtl": True},
    {"label": "Polski", "name": "Polish", "code": "pl", "rtl": False},
    {"label": "Português", "name": "Portuguese", "code": "pt", "rtl": False},
    {"label": "ਪੰਜਾਬੀ", "name": "Punjabi", "code": "pa", "rtl": False},
    {"label": "Русский", "name": "Russian", "code": "ru", "rtl": False},
    {"label": "Soomaali", "name": "Somali", "code": "so", "rtl": False},
    {"label": "Español", "name": "Spanish", "code": "es", "rtl": False},
    {"label": "Kiswahili", "name": "Swahili", "code": "sw", "rtl": False},
    {"label": "Tagalog", "name": "Tagalog", "code": "tl", "rtl": False},
    {"label": "தமிழ்", "name": "Tamil", "code": "ta", "rtl": False},
    {"label": "Türkçe", "name": "Turkish", "code": "tr", "rtl": False},
    {"label": "Українська", "name": "Ukrainian", "code": "uk", "rtl": False},
    {"label": "اردو", "name": "Urdu", "code": "ur", "rtl": True},
    {"label": "Tiếng Việt", "name": "Vietnamese", "code": "vi", "rtl": False},
]

# Derived lookups. SUPPORTED_LANGUAGES keeps the {label -> name} shape the UI
# already uses (dropdown labels -> instruction language), in the ordered list above.
SUPPORTED_LANGUAGES: dict[str, str] = {lang["label"]: lang["name"] for lang in LANGUAGES}
LANGUAGE_BY_NAME: dict[str, dict] = {lang["name"]: lang for lang in LANGUAGES}

DEFAULT_LANGUAGE = "English"


def is_rtl(language_name: str) -> bool:
    """Return True if the given language uses a right-to-left script (#18)."""
    return LANGUAGE_BY_NAME.get(language_name, {}).get("rtl", False)


class BackendError(RuntimeError):
    """Configuration or API error that the UI can surface to the user."""


def resolve_api_key(api_key: str | None = None) -> str:
    """Return the API key from the explicit argument or GROQ_API_KEY env var.

    Raises BackendError if neither is set.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise BackendError(
            "No Groq API key found. Set GROQ_API_KEY (env var) or add it to "
            ".streamlit/secrets.toml. Get a free key at https://console.groq.com/keys"
        )
    return key


def get_client(api_key: str | None = None) -> Groq:
    """Create a Groq client for the given (or environment) API key."""
    return Groq(api_key=resolve_api_key(api_key))


def build_system_prompt(language: str = DEFAULT_LANGUAGE) -> str:
    """Build the system prompt that fixes the assistant's response language."""
    return (
        "You are the Universal Multi-Lingual Access Assistant, built for the "
        "Universal Removal Defense Project (URDP). Your mission is to make "
        "information accessible across languages so that language is never a "
        "barrier to services or to justice.\n\n"
        f"Always respond in {language}, regardless of the language the user "
        "writes in, unless the user explicitly asks for a different language. "
        "Be clear, warm, and concise. If you are unsure, say so honestly."
    )


def build_context_prompt(context_blocks: list[str]) -> str:
    """Build a system message that grounds answers in retrieved sources (#25)."""
    joined = "\n\n".join(context_blocks)
    return (
        "Answer using only the sources below. After each fact you use, cite its "
        "source in brackets exactly as given, e.g. [handbook.pdf p.3]. If the "
        "answer is not in the sources, say you don't know based on the provided "
        "documents.\n\nSources:\n" + joined
    )


def build_messages(
    history: Iterable[dict],
    language: str = DEFAULT_LANGUAGE,
    context: list[str] | None = None,
) -> list[dict]:
    """Build the message list for the model.

    Prepends the language-aware system prompt and, when ``context`` is given,
    a retrieval-augmented sources prompt (#25). History is sanitised to only
    ``role``/``content`` so extra fields (e.g. stored ``sources``) are not sent.
    """
    messages = [{"role": "system", "content": build_system_prompt(language)}]
    if context:
        messages.append({"role": "system", "content": build_context_prompt(context)})
    messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    return messages


def stream_response(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> Iterator[str]:
    """Yield the assistant's reply incrementally as text deltas.

    Suitable for Streamlit's ``st.write_stream``.
    """
    client = get_client(api_key)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
    except Exception as exc:  # network / auth / model errors
        raise BackendError(f"Groq request failed: {exc}") from exc

    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def get_response(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> str:
    """Return the full reply as a string (non-streaming convenience wrapper)."""
    return "".join(stream_response(messages, model=model, api_key=api_key))


def detect_language(text: str, *, api_key: str | None = None) -> str | None:
    """Detect the language of ``text`` and return a supported language name (#17).

    Uses the fast detection model and constrains the answer to the supported
    languages. Returns None if the language cannot be matched, so the caller can
    keep the current language rather than guess.
    """
    if not text.strip():
        return None
    known = ", ".join(lang["name"] for lang in LANGUAGES)
    client = get_client(api_key)
    try:
        completion = client.chat.completions.create(
            model=DETECT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identify the language of the user's message. Reply with "
                        "ONLY the English name of the language, chosen from this "
                        f"list when it matches: {known}. Reply 'Unknown' if unsure."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=5,
        )
    except Exception as exc:
        raise BackendError(f"Language detection failed: {exc}") from exc

    name = (completion.choices[0].message.content or "").strip()
    return name if name in LANGUAGE_BY_NAME else None
