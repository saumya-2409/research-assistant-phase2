"""
auth.py
=======
Streamlit login/signup UI and session management.
Call render_auth_gate() at the top of main.py before any content.
If not logged in, shows login/signup form and stops execution.
If logged in, returns silently and the rest of main.py runs.
"""

import streamlit as st
from database import create_user, login_user, init_database


def _init_session():
    """Ensure auth keys exist in session state."""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_mode' not in st.session_state:
        st.session_state.auth_mode = 'login'


def render_auth_gate():
    """
    If user is not logged in, show the auth form and stop the page.
    If user is logged in, return immediately.
    """
    _init_session()

    # Already logged in — nothing to do
    if st.session_state.user is not None:
        return

    # ── Auth page layout ─────────────────────────────────────────────
    st.markdown("""
    <div style="max-width:420px;margin:60px auto 0;">
        <div style="text-align:center;margin-bottom:32px;">
            <div style="font-size:36px;margin-bottom:8px;">🔬</div>
            <div style="font-size:22px;font-weight:600;color:#1A1744;">
                AI Research Assistant
            </div>
            <div style="font-size:14px;color:#9B97C4;margin-top:4px;">
                Real papers. Organised. Ready to use.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Centre the form
    _, col, _ = st.columns([1, 2, 1])
    with col:
        # Tab switcher
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            _render_login()

        with tab_signup:
            _render_signup()

    # Stop the rest of main.py from running
    st.stop()


def _render_login():
    st.markdown("##### Welcome back")
    with st.form("login_form", clear_on_submit=False):
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submit   = st.form_submit_button("Sign In", type="primary", use_container_width=True)

    if submit:
        if not email or not password:
            st.error("Please fill in both fields.")
            return
        result = login_user(email, password)
        if result['success']:
            st.session_state.user = result['user']
            st.success(f"Welcome back, {result['user']['name']}!")
            st.rerun()
        else:
            st.error(result['error'])


def _render_signup():
    st.markdown("##### Create your account")
    with st.form("signup_form", clear_on_submit=False):
        name     = st.text_input("Full Name", placeholder="Saumya Garg")
        email    = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password",
                                  help="Minimum 6 characters")
        confirm  = st.text_input("Confirm Password", type="password")
        submit   = st.form_submit_button("Create Account", type="primary",
                                          use_container_width=True)

    if submit:
        if not name or not email or not password:
            st.error("All fields are required.")
            return
        if password != confirm:
            st.error("Passwords do not match.")
            return
        result = create_user(email, name, password)
        if result['success']:
            st.session_state.user = {
                'id':    result['user_id'],
                'name':  result['name'],
                'email': result['email'],
            }
            st.success(f"Account created! Welcome, {result['name']}.")
            st.rerun()
        else:
            st.error(result['error'])


def render_user_menu():
    """
    Renders a small user menu in the sidebar.
    Shows name, email, and logout button.
    """
    user = st.session_state.get('user')
    if not user:
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style="padding:8px 0;">
        <div style="font-size:13px;font-weight:600;color:#1A1744;">
            {user['name']}
        </div>
        <div style="font-size:11px;color:#9B97C4;">{user['email']}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Sign Out", use_container_width=True):
        st.session_state.user         = None
        st.session_state.papers_data  = []
        st.session_state.clusters     = {}
        st.session_state.full_text_papers  = []
        st.session_state.suggested_papers  = []
        st.rerun()