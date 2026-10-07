import streamlit as st

from rag_cache.context import get_context
from rag_cache.ui import render_sidebar, render_sources, render_trace

ctx = get_context()
user = st.session_state.get("user")
if not user:
    st.warning("Log in to open your chats.")
    st.stop()

render_sidebar(ctx)
st.title("Document chat")
st.caption(
    "Answers come only from uploaded documents. "
    f"A passage is used when its similarity is above {ctx.settings.similarity_threshold:.2f}. "
    f"Long chats are summarized after {ctx.settings.chat_summary_limit} messages, and the full transcript stays in PostgreSQL."
)

if ctx.db.active_chunk_count() == 0:
    st.warning("No documents are indexed yet. An admin needs to upload files before this chat can answer from them.")

sessions = ctx.db.list_chat_sessions(user["id"])
current_id = st.session_state.get("session_id")
if current_id and not any(session["id"] == current_id for session in sessions):
    current_id = None
    st.session_state.pop("session_id", None)

st.sidebar.divider()
st.sidebar.subheader("Chats")
if st.sidebar.button("New chat", type="primary", key="new-chat"):
    st.session_state.pop("session_id", None)
    st.rerun()

for session in sessions:
    label = session["title"] or "New chat"
    if len(label) > 48:
        label = label[:48] + "…"
    selected = session["id"] == current_id
    if st.sidebar.button(label, key=f"session-{session['id']}", type="primary" if selected else "secondary"):
        st.session_state["session_id"] = session["id"]
        st.rerun()

current = ctx.db.get_chat_session(current_id, user["id"]) if current_id else None
if current and current.get("summary"):
    st.info("Long-term memory summary")
    st.write(current["summary"])

if st.session_state.pop("just_summarized", False):
    st.success("Older messages were summarized into long-term memory. The transcript below is unchanged.")

messages = ctx.db.list_chat_messages(current["id"]) if current else []
for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])
            render_trace(message.get("cache_trace") or {})

prompt = st.chat_input("Ask about the uploaded documents")
if prompt:
    try:
        result = ctx.memory.converse(
            user_id=user["id"],
            session_id=current["id"] if current else None,
            question=prompt,
        )
    except Exception as exc:
        st.error(str(exc))
    else:
        st.session_state["session_id"] = result["session_id"]
        st.session_state["just_summarized"] = result["summarized"]
        st.rerun()
