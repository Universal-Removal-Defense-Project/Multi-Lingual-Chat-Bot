"""Conversation data model and JSON persistence.

This module has no Streamlit dependency. ``new_conversation`` defines the
canonical conversation structure used across the UI and persistence layers;
``load_conversations`` / ``save_conversations`` persist the ``{id -> conversation}``
map to disk.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# On-disk location of the persisted conversation store (gitignored).
STORE_PATH = Path(__file__).parent / "data" / "conversations.json"

# On-disk location of persisted UI settings, e.g. the dark-mode preference.
SETTINGS_PATH = Path(__file__).parent / "data" / "settings.json"

# Title used until the first user message provides a better one.
DEFAULT_TITLE = "New chat"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_conversation(language: str = "English") -> dict:
    """Create an empty conversation.

    This is the canonical conversation structure; other modules should not
    construct conversation dicts directly.
    """
    return {
        "id": uuid.uuid4().hex,
        "title": DEFAULT_TITLE,
        "created_at": _now_iso(),
        "language": language,
        "messages": [],  # list of {"role": "user"|"assistant", "content": str}
    }


def derive_title(conversation: dict, max_len: int = 40) -> str:
    """Derive a short title from the first user message."""
    for msg in conversation["messages"]:
        if msg["role"] == "user" and msg["content"].strip():
            text = msg["content"].strip().replace("\n", " ")
            return text[:max_len] + ("…" if len(text) > max_len else "")
    return DEFAULT_TITLE


# --- Persistence ---

def load_conversations(path: Path = STORE_PATH) -> dict[str, dict]:
    """Load the ``{id -> conversation}`` map from disk, or an empty dict."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable store: start clean rather than crash.
        return {}


def save_conversations(conversations: dict[str, dict], path: Path = STORE_PATH) -> None:
    """Persist the ``{id -> conversation}`` map to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(conversations, fh, ensure_ascii=False, indent=2)


def load_dark_mode(path: Path = SETTINGS_PATH) -> bool:
    """Load the persisted dark-mode preference; defaults to light (False)."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("dark_mode", False))
    except (json.JSONDecodeError, OSError):
        return False


def save_dark_mode(dark_mode: bool, path: Path = SETTINGS_PATH) -> None:
    """Persist the dark-mode preference so it survives refreshes/restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"dark_mode": dark_mode}, fh)
