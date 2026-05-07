import requests
import streamlit as st
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import google.auth.transport.requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOOGLE_CLIENT_ID = st.secrets.get(
    "GOOGLE_CLIENT_ID", "your-client-id.apps.googleusercontent.com"
)
GOOGLE_CLIENT_SECRET = st.secrets.get(
    "GOOGLE_CLIENT_SECRET", "your-client-secret"
)
REDIRECT_URI = "http://localhost:8501"

OAUTH_SCOPES = [
    'openid',
    'https://www.googleapis.com/auth/userinfo.profile',
    'https://www.googleapis.com/auth/userinfo.email',
]

USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo"


# ---------------------------------------------------------------------------
# Google OAuth client
# ---------------------------------------------------------------------------

class GoogleAuth:
    def __init__(self):
        self.client_config = {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
                "issuer": "https://accounts.google.com",
                "auth_provider_x509_cert_url":
                    "https://www.googleapis.com/oauth2/v1/certs",
            }
        }

    def _build_flow(self):
        return Flow.from_client_config(
            self.client_config,
            scopes=OAUTH_SCOPES,
            redirect_uri=REDIRECT_URI,
        )

    def get_authorization_url(self):
        """Generate Google OAuth authorization URL."""
        flow = self._build_flow()
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
        )
        return auth_url

    def verify_token(self, token):
        """Verify a Google OAuth ID token and return user info."""
        try:
            idinfo = id_token.verify_oauth2_token(
                token,
                google.auth.transport.requests.Request(),
                GOOGLE_CLIENT_ID,
            )
        except ValueError as e:
            st.error(f"Token verification failed: {e}")
            return None

        return {
            'email': idinfo.get('email'),
            'name': idinfo.get('name'),
            'picture': idinfo.get('picture'),
            'verified_email': idinfo.get('email_verified', False),
        }

    def authenticate_user(self, authorization_code):
        """Exchange an authorization code for user information."""
        try:
            flow = self._build_flow()
            flow.fetch_token(code=authorization_code)
            return self._fetch_userinfo(flow.credentials.token)
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            st.query_params.clear()
            return None

    def _fetch_userinfo(self, access_token):
        """Fetch user info from Google's userinfo endpoint."""
        response = requests.get(
            USERINFO_ENDPOINT,
            headers={'Authorization': f'Bearer {access_token}'},
        )
        if response.status_code != 200:
            st.error(f"Failed to get user info: {response.status_code}")
            return None

        user_data = response.json()
        return {
            'email': user_data.get('email'),
            'name': user_data.get('name'),
            'picture': user_data.get('picture'),
            'verified_email': user_data.get('verified_email', False),
        }


def is_google_auth_configured():
    """Check if Google OAuth is properly configured."""
    return (
        GOOGLE_CLIENT_ID
        and GOOGLE_CLIENT_SECRET
        and GOOGLE_CLIENT_ID != "your-client-id.apps.googleusercontent.com"
        and GOOGLE_CLIENT_SECRET != "your-client-secret"
    )
