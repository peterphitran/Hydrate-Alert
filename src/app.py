import hashlib
import json
import time

import streamlit as st
from streamlit_option_menu import option_menu

import pages.data_analysis as data_analysis
import pages.data_upload as data_upload
import pages.help as help
import pages.home as home
import pages.landing as landing_page
from pages.style import google_button_style

try:
    from google_auth import GoogleAuth, is_google_auth_configured
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_TOKEN_TTL_SECONDS = 86400  # 24h
DEMO_USERNAME = "admin"
DEMO_PASSWORD = "password"

NAV_OPTIONS = ["Home", "Upload Data", "Data Analysis", "Help"]
NAV_ICONS = ["‎", "‎ ", "‎ ", "‎ "]

NAV_STYLES = {
    "container": {
        "padding": "0 !important",
        "background-color": "#f0f0f0",
        "border-radius": "10px",
    },
    "nav-link": {
        "text-align": "center !important",
        "font-family": "'Nunito', sans-serif !important",
        "font-size": "1rem",
        "color": "#000000",
        "font-weight": "normal !important",
        "margin": "0",
        "padding": "10px 20px",
        "background-color": "#f0f0f0",
        "border-radius": "8px",
    },
    "nav-link-selected": {
        "text-align": "center !important",
        "font-family": "'Nunito', sans-serif !important",
        "background-color": "#ff4444",
        "color": "#000000",
        "font-weight": "normal !important",
        "margin": "0",
        "padding": "10px 20px",
        "border-radius": "8px",
    },
}


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------

def create_auth_token(username, email):
    """Create a persistent authentication token"""
    timestamp = str(int(time.time()))
    token_data = f"{username}:{email}:{timestamp}"
    token_hash = hashlib.sha256(token_data.encode()).hexdigest()
    return f"{token_hash}:{timestamp}"

def validate_auth_token(token, username, email):
    """Validate authentication token (signature + TTL)."""
    try:
        token_hash, timestamp = token.split(':')
    except (ValueError, AttributeError):
        return False

    expected = hashlib.sha256(
        f"{username}:{email}:{timestamp}".encode()
    ).hexdigest()
    if token_hash != expected:
        return False
    return (int(time.time()) - int(timestamp)) < AUTH_TOKEN_TTL_SECONDS


# ---------------------------------------------------------------------------
# Auth state management
# ---------------------------------------------------------------------------

def save_auth_state(username, email, picture=""):
    """Save authentication state to session and browser sessionStorage."""
    auth_token = create_auth_token(username, email)

    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.user_email = email
    st.session_state.user_picture = picture
    st.session_state.auth_token = auth_token

    auth_data = {
        'username': username,
        'email': email,
        'picture': picture,
        'token': auth_token,
        'timestamp': int(time.time()),
    }
    st.html(f"""
    <script>
        sessionStorage.setItem('hydrate_auth', JSON.stringify({json.dumps(auth_data)}));
    </script>
    """)


def clear_auth_state():
    """Clear authentication state from session and browser storage."""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_email = ""
    st.session_state.user_picture = ""
    st.session_state.pop('auth_token', None)

    st.html("""
    <script>
        sessionStorage.removeItem('hydrate_auth');
        localStorage.removeItem('hydrate_auth');
    </script>
    """)


def init_session_state():
    """Initialize default session state keys."""
    defaults = {
        'logged_in': False,
        'username': "",
        'user_email': "",
        'user_picture': "",
        'sidebar_open': False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Browser-storage auth restoration
# ---------------------------------------------------------------------------

_AUTH_RESTORE_SCRIPT = """
<script>
    const authData = sessionStorage.getItem('hydrate_auth');
    if (authData) {
        try {
            const parsed = JSON.parse(authData);
            const now = Math.floor(Date.now() / 1000);
            if (now - parsed.timestamp < 86400) {
                const form = document.createElement('form');
                form.method = 'POST';
                form.style.display = 'none';
                const fields = {
                    restore_username: parsed.username,
                    restore_email: parsed.email,
                    restore_picture: parsed.picture,
                    restore_token: parsed.token,
                };
                for (const [name, value] of Object.entries(fields)) {
                    const input = document.createElement('input');
                    input.name = name;
                    input.value = value;
                    form.appendChild(input);
                }
                document.body.appendChild(form);
                window.dispatchEvent(new CustomEvent('streamlit:setComponentValue', {
                    detail: { value: 'restore_auth' }
                }));
            }
        } catch (e) {
            console.error('Error restoring auth:', e);
            sessionStorage.removeItem('hydrate_auth');
        }
    }
</script>
"""


def inject_auth_restore_script():
    """Inject JS that attempts to restore auth from sessionStorage."""
    st.html(_AUTH_RESTORE_SCRIPT)


def handle_auth_restore_from_query():
    """Restore auth from query params produced by the JS restore form."""
    if 'restore_username' not in st.query_params:
        return

    username = st.query_params.get('restore_username')
    email = st.query_params.get('restore_email')
    picture = st.query_params.get('restore_picture', '')
    token = st.query_params.get('restore_token')

    if not (username and email and token):
        return

    if validate_auth_token(token, username, email):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.user_email = email
        st.session_state.user_picture = picture
        st.session_state.auth_token = token
        st.query_params.clear()
        st.rerun()
    else:
        clear_auth_state()


# ---------------------------------------------------------------------------
# Google OAuth callback
# ---------------------------------------------------------------------------

def handle_google_oauth_callback():
    """Process the OAuth redirect callback from Google."""
    if not GOOGLE_AUTH_AVAILABLE:
        return

    query_params = st.query_params

    if 'error' in query_params:
        error_description = query_params.get(
            'error_description', ['Unknown error']
        )[0]
        st.error(f"OAuth Error: {error_description}")
        st.query_params.clear()
        st.rerun()
        return

    if 'code' not in query_params or 'oauth_processed' in st.session_state:
        return

    st.session_state.oauth_processed = True
    user_info = GoogleAuth().authenticate_user(query_params['code'])

    if user_info:
        save_auth_state(
            user_info['name'],
            user_info['email'],
            user_info.get('picture', ''),
        )
        st.success(f"Welcome, {user_info['name']}!")
        st.query_params.clear()
        st.session_state.pop('oauth_processed', None)
        st.rerun()
    else:
        st.session_state.pop('oauth_processed', None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

_GOOGLE_LOGIN_BUTTON_TEMPLATE = '''
<div class="google-login-container">
    <a href="{auth_url}" class="google-login-btn" target="_self">
        <svg class="google-logo" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
        </svg>
        Continue with Google
    </a>
</div>
'''


def render_sidebar():
    with st.sidebar:
        st.header("Login")
        if st.session_state.logged_in:
            _render_logged_in_sidebar()
        else:
            _render_login_form()


def _render_login_form():
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input(
        "Password", type="password", placeholder="Enter your password"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            _handle_local_login(username, password)
    with col2:
        if st.button("Register", use_container_width=True):
            st.info("Registration feature coming soon!")

    _render_google_login_button()

    st.markdown("---")
    st.markdown("**Demo Credentials:**")
    st.markdown(f"Username: `{DEMO_USERNAME}`")
    st.markdown(f"Password: `{DEMO_PASSWORD}`")


def _handle_local_login(username, password):
    if not (username and password):
        st.error("Please enter both username and password")
        return
    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        save_auth_state(username, f"{username}@local.dev")
        st.rerun()
    else:
        st.error("Invalid username or password")


def _render_google_login_button():
    if not GOOGLE_AUTH_AVAILABLE:
        return

    if not is_google_auth_configured():
        st.markdown("---")
        st.warning(
            "Google authentication not configured. Please set up Google "
            "OAuth credentials."
        )
        return

    auth_url = GoogleAuth().get_authorization_url()
    google_button_style()
    st.markdown(
        _GOOGLE_LOGIN_BUTTON_TEMPLATE.format(auth_url=auth_url),
        unsafe_allow_html=True,
    )


def _render_logged_in_sidebar():
    st.success(f"Welcome, {st.session_state.username}!")
    st.markdown("---")
    st.subheader("User Profile")
    if st.session_state.user_picture:
        st.image(st.session_state.user_picture, width=60)
    st.write(f"**Name:** {st.session_state.username}")
    if st.session_state.user_email:
        st.write(f"**Email:** {st.session_state.user_email}")
    st.write("**Status:** Online")
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        clear_auth_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

PAGE_RENDERERS = {
    "Home": home.home_page,
    "Upload Data": data_upload.upload_data,
    "Data Analysis": data_analysis.data_analysis,
    "Help": help.help_page,
}


def render_main_content():
    selected = option_menu(
        menu_title=None,
        options=NAV_OPTIONS,
        icons=NAV_ICONS,
        orientation="horizontal",
        styles=NAV_STYLES,
    )
    renderer = PAGE_RENDERERS.get(selected)
    if renderer:
        renderer()


def load_global_styles():
    st.markdown(
        '<link href="https://fonts.googleapis.com/css2?family=Nunito:'
        'wght@300;400;500;600;700&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(initial_sidebar_state="collapsed")
    init_session_state()

    if not st.session_state.logged_in:
        inject_auth_restore_script()

    handle_auth_restore_from_query()
    handle_google_oauth_callback()
    load_global_styles()
    render_sidebar()

    if st.session_state.logged_in:
        render_main_content()
    else:
        landing_page.landing_page()


main()
