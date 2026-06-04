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

# Maps the label shown in the language dropdown to the language the model is
# instructed to reply in.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "English",
    "Español": "Spanish",
    "中文": "Chinese",
    "Français": "French",
    "Deutsch": "German",
    "العربية": "Arabic",
    "日本語": "Japanese",
    "हिन्दी": "Hindi",
    "Português": "Portuguese",
}

DEFAULT_LANGUAGE = "English"


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


def build_messages(history: Iterable[dict], language: str = DEFAULT_LANGUAGE) -> list[dict]:
    """Prepend the language-aware system prompt to the conversation history.

    ``history`` is a list of ``{"role": "user"|"assistant", "content": str}``
    message dicts.
    """
    return [{"role": "system", "content": build_system_prompt(language)}, *history]


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
