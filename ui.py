"""Streamlit chat UI for the URDP Multi-Lingual Access Assistant.

This is the only module that imports Streamlit. AI logic lives in backend.py
and the conversation data model in storage.py. Conversations are keyed by id in
session_state and persisted to disk, so history survives page refreshes and the
user can keep, switch between, rename, and delete multiple conversations.

Run with: streamlit run ui.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import backend
import storage
import styles

# Load a local .env (if present) so GROQ_API_KEY etc. are available via os.environ.
load_dotenv()

st.set_page_config(
    page_title="URDP Multi-Lingual Assistant",
    page_icon="🌐",
    layout="centered",
)

# Optional brand logo. Drop a file here to have it shown automatically.
LOGO_PATH = "assets/logo.png"


def render_brand() -> None:
    """Render the URDP brand mark: logo image if present, else a text mark."""
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown("### 🌐 URDP")
    st.caption("Multi-Lingual Assistant · Language should never be a barrier.")


def get_api_key() -> str | None:
    """Return the Groq API key from Streamlit secrets or the GROQ_API_KEY env var.

    Returns None if unset so the UI can show a setup message instead of crashing.
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        # No secrets file configured; fall back to the environment variable.
        pass
    return os.environ.get("GROQ_API_KEY")


# --- Session state and persistence ---

def persist() -> None:
    """Write the current conversations to disk."""
    storage.save_conversations(st.session_state.conversations)


def most_recent_id() -> str:
    """Return the id of the most recently created conversation."""
    return max(
        st.session_state.conversations,
        key=lambda cid: st.session_state.conversations[cid]["created_at"],
    )


def init_state() -> None:
    """Load persisted conversations and select an active one."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = storage.load_conversations()
    if not st.session_state.conversations:
        conv = storage.new_conversation(language=backend.DEFAULT_LANGUAGE)
        st.session_state.conversations[conv["id"]] = conv
        persist()
    if (
        "active_id" not in st.session_state
        or st.session_state.active_id not in st.session_state.conversations
    ):
        set_active(most_recent_id())
    st.session_state.setdefault("dark_mode", styles.DEFAULT_THEME == "dark")
    st.session_state.setdefault("auto_detect", False)


def active_conversation() -> dict:
    return st.session_state.conversations[st.session_state.active_id]


def label_for_language(name: str) -> str:
    """Return the dropdown label for a language name (falls back to English)."""
    return next(
        (lbl for lbl, n in backend.SUPPORTED_LANGUAGES.items() if n == name),
        "English",
    )


def set_active(cid: str) -> None:
    """Switch the active conversation. The language dropdown re-syncs on the next
    render (see render_sidebar), because a widget key cannot be written after the
    widget is instantiated."""
    st.session_state.active_id = cid


def start_new_chat() -> None:
    """Create a new conversation and make it active, keeping existing chats."""
    conv = storage.new_conversation(language=backend.DEFAULT_LANGUAGE)
    st.session_state.conversations[conv["id"]] = conv
    set_active(conv["id"])
    persist()


def delete_conversation(cid: str) -> None:
    """Delete a conversation, ensuring one active conversation always remains."""
    st.session_state.conversations.pop(cid, None)
    if not st.session_state.conversations:
        start_new_chat()
        return
    if st.session_state.active_id == cid:
        set_active(most_recent_id())
    persist()


# --- Sidebar ---

def _on_language_change() -> None:
    """Apply a manual language pick from the dropdown to the active conversation."""
    active_conversation()["language"] = backend.SUPPORTED_LANGUAGES[
        st.session_state.lang_select
    ]
    persist()


def render_sidebar(conversation: dict) -> None:
    with st.sidebar:
        render_brand()

        if st.button("➕ New chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        # Auto-detect language from the user's message (Issue #17). When off,
        # the dropdown is the manual override.
        auto = st.toggle(
            "🌐 Auto-detect language",
            key="auto_detect",
            help="Detect the language from your message and reply in it.",
        )

        # Sync the dropdown to the active conversation's language before the
        # widget is created (writing a widget key after creation is disallowed).
        st.session_state.lang_select = label_for_language(conversation["language"])
        st.selectbox(
            "Response language",
            list(backend.SUPPORTED_LANGUAGES.keys()),
            key="lang_select",
            on_change=_on_language_change,
            disabled=auto,
            help="The assistant will reply in this language.",
        )
        if auto:
            st.caption(f"Detected: {conversation['language']}")

        st.divider()
        _render_conversation_list()
        _render_delete_confirmation()
        _render_rename(conversation)

        st.divider()
        st.toggle("🌙 Dark mode", key="dark_mode")


def _render_conversation_list() -> None:
    """List conversations (newest first); clicking one switches to it."""
    st.caption("Conversations")
    conversations = st.session_state.conversations
    ordered = sorted(
        conversations,
        key=lambda cid: conversations[cid]["created_at"],
        reverse=True,
    )
    for cid in ordered:
        title = conversations[cid]["title"]
        is_active = cid == st.session_state.active_id
        row, del_col = st.columns([0.82, 0.18])
        if row.button(
            title,
            key=f"open_{cid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            set_active(cid)
            st.rerun()
        if del_col.button("🗑", key=f"del_{cid}", help="Delete this conversation"):
            st.session_state.pending_delete = cid
            st.rerun()


def _render_delete_confirmation() -> None:
    """Confirm before deleting, so chats are not removed accidentally."""
    cid = st.session_state.get("pending_delete")
    if not cid or cid not in st.session_state.conversations:
        st.session_state.pop("pending_delete", None)
        return
    title = st.session_state.conversations[cid]["title"]
    st.warning(f"Delete “{title}”? This cannot be undone.")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Delete", key="confirm_delete", type="primary", use_container_width=True):
        delete_conversation(cid)
        st.session_state.pop("pending_delete", None)
        st.rerun()
    if cancel_col.button("Cancel", key="cancel_delete", use_container_width=True):
        st.session_state.pop("pending_delete", None)
        st.rerun()


def _render_rename(conversation: dict) -> None:
    """Rename the active conversation."""
    with st.expander("Rename current chat"):
        new_title = st.text_input(
            "Title",
            value=conversation["title"],
            key=f"rename_{conversation['id']}",
            label_visibility="collapsed",
        )
        cleaned = new_title.strip()
        if cleaned and cleaned != conversation["title"]:
            conversation["title"] = cleaned
            persist()
            # Rerun so the sidebar list (rendered above) shows the new title now.
            st.rerun()


# --- Main chat panel ---

def render_history(conversation: dict) -> None:
    for msg in conversation["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_input(conversation: dict, api_key: str | None) -> None:
    prompt = st.chat_input("Type your message…")
    if not prompt:
        return

    conversation["messages"].append({"role": "user", "content": prompt})
    # Auto-title a new chat from its first message.
    if conversation["title"] == storage.DEFAULT_TITLE:
        conversation["title"] = storage.derive_title(conversation)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Auto-detect the response language from the message (Issue #17).
    if st.session_state.get("auto_detect") and api_key:
        try:
            detected = backend.detect_language(prompt, api_key=api_key)
            if detected:
                conversation["language"] = detected
        except backend.BackendError:
            pass  # detection is best-effort; keep the current language

    with st.chat_message("assistant"):
        if not api_key:
            reply = (
                "⚠️ No Groq API key configured. Add `GROQ_API_KEY` to "
                "`.streamlit/secrets.toml` or set it as an environment variable, "
                "then reload. Get a free key at https://console.groq.com/keys"
            )
            st.warning(reply)
        else:
            messages = backend.build_messages(
                conversation["messages"], conversation["language"]
            )
            try:
                # st.write_stream renders the generator incrementally and
                # returns the full reply text.
                reply = st.write_stream(
                    backend.stream_response(messages, api_key=api_key)
                )
            except backend.BackendError as exc:
                reply = f"⚠️ {exc}"
                st.error(reply)
            except Exception as exc:  # catch-all so a failure doesn't crash the app
                reply = f"⚠️ Unexpected error: {exc}"
                st.error(reply)

    conversation["messages"].append({"role": "assistant", "content": reply})
    persist()


def main() -> None:
    init_state()
    conversation = active_conversation()

    render_sidebar(conversation)
    # Inject theme CSS after the sidebar so the toggle applies in the same run.
    theme = "dark" if st.session_state.dark_mode else "light"
    st.markdown(styles.theme_css(theme), unsafe_allow_html=True)
    # Flip to right-to-left when the active language needs it (Issue #18).
    st.markdown(
        styles.rtl_css(backend.is_rtl(conversation["language"])),
        unsafe_allow_html=True,
    )

    st.title("🌐 URDP Multi-Lingual Assistant")
    st.caption("Ask anything — I'll reply in your selected language.")

    render_history(conversation)
    handle_input(conversation, get_api_key())


if __name__ == "__main__":
    main()
