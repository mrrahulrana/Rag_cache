import streamlit as st

from rag_cache.context import get_context
from rag_cache.loaders import SUPPORTED_EXTENSIONS, IngestError
from rag_cache.ui import render_sidebar

ctx = get_context()
user = st.session_state.get("user")
if not user:
    st.warning("Log in as an admin to upload documents.")
    st.stop()
if user.get("role") != "admin":
    st.error("Only an admin can upload documents.")
    st.stop()

render_sidebar(ctx)
st.title("Documents")
st.caption(
    "Upload a new file, or upload the same file name again to create the next version. "
    "Unchanged passages keep their embeddings. Only new or edited passages are embedded with "
    f"{ctx.settings.embed_model}."
)

accepted = sorted({extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS})
uploads = st.file_uploader(
    "PDF, Word, Excel, PowerPoint, text, HTML, or images",
    type=accepted,
    accept_multiple_files=True,
)
if st.button("Upload and embed", type="primary", disabled=not uploads):
    progress = st.progress(0, text="Starting")
    for uploaded in uploads:
        def on_progress(done: int, total: int, message: str, name: str = uploaded.name) -> None:
            fraction = 0 if total <= 0 else min(done / total, 1)
            progress.progress(fraction, text=f"{name}: {message}")

        try:
            if not ctx.ollama_ok:
                raise RuntimeError(ctx.ollama_error or "Ollama is not available.")
            result = ctx.ingestor.ingest_bytes(
                uploaded.name,
                uploaded.getvalue(),
                user["id"],
                on_progress,
            )
        except (IngestError, RuntimeError, OSError) as exc:
            st.error(f"{uploaded.name}: {exc}")
            continue
        if result.status == "unchanged":
            st.info(result.message)
        else:
            st.success(result.message)
    progress.progress(1.0, text="Finished")

documents = ctx.db.list_documents()
st.subheader("Library")
st.caption(f"Corpus version {ctx.db.corpus_version()} · {ctx.db.active_chunk_count()} active chunks")
if documents:
    st.dataframe(
        [
            {
                "Name": row["name"],
                "Version": row["version"],
                "Status": row["status"],
                "Chunks": row["chunk_count"],
                "Newly embedded": row["embedded_count"],
                "Reused embeddings": row["reused_count"],
                "Uploaded by": row.get("uploaded_by") or "",
                "Uploaded": row["created_at"],
            }
            for row in documents
        ],
        use_container_width=True,
        hide_index=True,
    )
    active = [row for row in documents if row["status"] == "active"]
    if active:
        choice = st.selectbox(
            "Inspect chunk metadata",
            options=active,
            format_func=lambda row: f"{row['name']} · v{row['version']}",
        )
        previews = ctx.db.list_chunk_previews(choice["id"])
        st.dataframe(
            [
                {
                    "Chunk": row["chunk_index"],
                    "Section": row.get("section") or "",
                    "Page": row.get("page_number") if row.get("page_number") is not None else "",
                    "Metadata": row.get("metadata") or {},
                    "Preview": row.get("preview") or "",
                }
                for row in previews
            ],
            use_container_width=True,
            hide_index=True,
        )
else:
    st.write("No documents uploaded yet.")
