import streamlit as st

from rag_cache.context import get_context

st.set_page_config(page_title="RAG Cache", layout="wide")

try:
    get_context()
except Exception as exc:
    st.error("RAG Cache could not start.")
    st.write(str(exc))
    st.info("Copy .env.example to .env, set DATABASE_URL to your local PostgreSQL 15 instance, then restart Streamlit.")
    st.stop()

user = st.session_state.get("user")
if not user:
    navigation = st.navigation([st.Page("ui_pages/login.py", title="Account", default=True)])
else:
    pages = [st.Page("ui_pages/chat.py", title="Chat", default=True)]
    if user.get("role") == "admin":
        pages.append(st.Page("ui_pages/admin.py", title="Documents"))
    navigation = st.navigation(pages)

navigation.run()
