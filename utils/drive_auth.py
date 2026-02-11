"""Google Drive OAuth authentication helper.

SETUP INSTRUCTIONS
==================
1. Go to the Google Cloud Console: https://console.cloud.google.com/
2. Create a project (or select an existing one).
3. Enable the **Google Drive API** for that project.
4. Go to Credentials → Create Credentials → OAuth client ID.
   - Application type: **Desktop app**
   - Download the JSON file.
5. Rename the downloaded file to **credentials.json** and place it at:

       OboretaOracle/config/credentials.json

6. The first time you connect, a browser window will open asking you to
   sign in and grant Drive read access.  After that, a **token.json** is
   saved at config/token.json so you won't need to log in again until the
   token expires.

Both files are gitignored by default — never commit them.
"""

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Read-only scope — we only need to list and download files.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Paths relative to the project root.
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
CREDENTIALS_PATH = _CONFIG_DIR / "credentials.json"
TOKEN_PATH = _CONFIG_DIR / "token.json"


def authenticate() -> Credentials:
    """Return valid Google OAuth credentials, launching a browser flow if needed.

    Resolution order:
      1. Load an existing token.json and refresh it if expired.
      2. If no token exists (or it can't be refreshed), run the
         InstalledAppFlow so the user can authorize via their browser.
      3. Persist the resulting token to config/token.json for next time.

    Returns:
        A google.oauth2.credentials.Credentials object ready for use
        with GoogleDriveLoader or the Drive API directly.

    Raises:
        FileNotFoundError: If config/credentials.json is missing.
    """
    creds = None

    # -- 1. Try loading a cached token ----------------------------------------
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    # -- 2. Refresh or run the full OAuth flow --------------------------------
    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        print("Refreshing expired Google Drive token…")
        creds.refresh(Request())
    else:
        # No usable token — need the user to log in.
        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"Google OAuth credentials not found at {CREDENTIALS_PATH}.\n"
                "Download your Desktop-app OAuth JSON from the Google Cloud "
                "Console and save it as config/credentials.json.\n"
                "See utils/drive_auth.py for full setup instructions."
            )

        print("Launching Google OAuth flow — a browser window should open…")
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CREDENTIALS_PATH), SCOPES
        )
        creds = flow.run_local_server(port=0)

    # -- 3. Persist the token for future sessions -----------------------------
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    print(f"Token saved to {TOKEN_PATH}")

    return creds
