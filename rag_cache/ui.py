from __future__ import annotations

import streamlit as st

TRACE_LABELS = (
    ("exact_answer", "Exact answer"),
    ("query_embedding", "Query embedding"),
    ("semantic_answer", "Semantic answer"),
    ("retrieval", "Retrieval"),
    ("llm", "LLM response"),
)


def render_sidebar(ctx) -> None:
    user = st.session_state.get("user")
    if not user:
        return
    st.sidebar.markdown(f"**{user['first_name']} {user['last_name']}**")
    st.sidebar.caption(f"{user['username']} · {user['role']}")
    if st.sidebar.button("Log out", key="logout"):
        st.session_state.pop("user", None)
        st.session_state.pop("session_id", None)
        st.rerun()
    with st.sidebar.expander("Services"):
        st.write("PostgreSQL: connected")
        if ctx.redis_ok:
            st.write("Redis: connected")
        else:
            st.write("Redis: cache disabled")
            if ctx.redis_error:
                st.caption(ctx.redis_error)
        if ctx.ollama_ok:
            st.write(f"Embeddings: {ctx.settings.embed_model}")
            st.write(f"Chat: {ctx.settings.chat_model}")
        else:
            st.write("Ollama: unavailable")
            st.caption(ctx.ollama_error or "")
    if not ctx.db.has_admin():
        st.sidebar.warning("No admin account yet. Set ADMIN_PASSWORD in .env to something other than change-me, then restart.")


def render_sources(sources: list[dict] | None) -> None:
    if not sources:
        return
    from rag_cache.sources import format_source

    st.markdown("**Sources**")
    for source in sources:
        st.caption(format_source(source))
        snippet = source.get("snippet")
        if snippet:
            st.caption(snippet)


def render_trace(trace: dict | None) -> None:
    if not trace:
        return
    with st.expander("Cache path"):
        for key, label in TRACE_LABELS:
            state = trace.get(key, "skipped")
            st.write(f"{label}: {state}")
