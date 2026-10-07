import streamlit as st

from rag_cache.auth import AuthError, authenticate, register_user
from rag_cache.context import get_context

ctx = get_context()
st.title("RAG Cache")
st.caption("Sign in to continue a document chat, or register a new account. Uploads are handled by an admin.")

login_tab, register_tab = st.tabs(["Log in", "Register"])

with login_tab:
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
    if submitted:
        user = authenticate(ctx.db, username, password)
        if user is None:
            st.error("Unknown username or password.")
        else:
            st.session_state["user"] = user
            st.session_state.pop("session_id", None)
            st.rerun()

with register_tab:
    with st.form("register"):
        first_name = st.text_input("First name")
        last_name = st.text_input("Last name")
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        registered = st.form_submit_button("Create account", type="primary")
    if registered:
        try:
            user = register_user(
                ctx.db,
                first_name=first_name,
                last_name=last_name,
                username=new_username,
                password=new_password,
                confirm_password=confirm_password,
            )
        except AuthError as exc:
            st.error(str(exc))
        else:
            st.session_state["user"] = user
            st.session_state.pop("session_id", None)
            st.rerun()
