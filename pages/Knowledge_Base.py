"""Knowledge Base admin page (Milestone 6 / Issue #26).

A Streamlit multipage page (auto-listed in the sidebar nav). Lets staff upload
PDFs, see processing status, re-index, and delete. All document logic lives in
knowledge.py; this file is UI only.
"""

import streamlit as st

import knowledge
import storage
import styles

st.set_page_config(page_title="URDP · Knowledge Base", page_icon="📚", layout="centered")

# Match the main app's theme (same persisted preference as ui.py's sidebar toggle).
theme = "dark" if st.session_state.get("dark_mode", storage.load_dark_mode()) else "light"
st.markdown(styles.theme_css(theme), unsafe_allow_html=True)

st.title("📚 Knowledge Base")
st.caption("Upload documents the assistant can cite when answering.")

# --- Upload (#23) ---
with st.form("upload_form", clear_on_submit=True):
    files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )
    submitted = st.form_submit_button("Add to knowledge base")

if submitted and files:
    for uploaded in files:
        try:
            result = knowledge.add_pdf(uploaded.name, uploaded.getvalue())
        except Exception as exc:  # never let one bad file crash the page
            st.error(f"{uploaded.name}: could not process ({exc})")
            continue
        if result["status"] == "ready":
            st.success(f"{uploaded.name}: indexed {result['num_chunks']} chunks")
        else:
            st.warning(f"{uploaded.name}: no extractable text (scanned or empty PDF?)")

st.divider()

# --- Document list + management (#26) ---
documents = knowledge.list_documents()
if not documents:
    st.info("No documents yet. Upload a PDF above to get started.")
else:
    st.caption(f"{len(documents)} document(s)")
    for doc in documents:
        badge = "✅ ready" if doc["status"] == "ready" else "⚠️ " + doc["status"]
        name_col, meta_col, reindex_col, del_col = st.columns([0.5, 0.2, 0.15, 0.15])
        name_col.markdown(f"**{doc['filename']}**")
        meta_col.caption(f"{badge} · {doc['num_chunks']} chunks")
        if reindex_col.button("Re-index", key=f"reindex_{doc['id']}"):
            knowledge.reindex_document(doc["id"])
            st.rerun()
        if del_col.button("Delete", key=f"delete_{doc['id']}"):
            knowledge.delete_document(doc["id"])
            st.rerun()
