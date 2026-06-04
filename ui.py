"""Streamlit chat UI for the URDP Multi-Lingual Access Assistant.

This is the only module that imports Streamlit. AI logic lives in backend.py
and the conversation data model in storage.py. Conversations are keyed by id in
session_state, so the app supports multiple conversations.

Run with: streamlit run ui.py
"""

from __future__ import annotations

import os

import streamlit as st

import backend
import storage

st.set_page_config(
    page_title="URDP Multi-Lingual Assistant",
    page_icon="🌐",
    layout="centered",
)


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


def init_state() -> None:
    """Initialise session state with an active conversation."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "active_id" not in st.session_state or (
        st.session_state.active_id not in st.session_state.conversations
    ):
        conv = storage.new_conversation(language=backend.DEFAULT_LANGUAGE)
        st.session_state.conversations[conv["id"]] = conv
        st.session_state.active_id = conv["id"]


def active_conversation() -> dict:
    return st.session_state.conversations[st.session_state.active_id]


def render_sidebar(conversation: dict) -> None:
    """Render the sidebar with the response-language selector."""
    with st.sidebar:
        st.markdown("### 🌐 URDP Assistant")
        st.caption("Language should never be a barrier.")

        labels = list(backend.SUPPORTED_LANGUAGES.keys())
        # Keep the dropdown in sync with the language stored on the conversation.
        current_label = next(
            (lbl for lbl, lang in backend.SUPPORTED_LANGUAGES.items()
             if lang == conversation["language"]),
            "English",
        )
        chosen = st.selectbox(
            "Response language",
            labels,
            index=labels.index(current_label),
            help="The assistant will reply in this language.",
        )
        conversation["language"] = backend.SUPPORTED_LANGUAGES[chosen]


def render_history(conversation: dict) -> None:
    for msg in conversation["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def handle_input(conversation: dict, api_key: str | None) -> None:
    prompt = st.chat_input("Type your message…")
    if not prompt:
        return

    conversation["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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


def main() -> None:
    init_state()
    conversation = active_conversation()

    render_sidebar(conversation)

    st.title("🌐 URDP Multi-Lingual Assistant")
    st.caption("Ask anything — I'll reply in your selected language.")

    render_history(conversation)
    handle_input(conversation, get_api_key())


if __name__ == "__main__":
    main()
