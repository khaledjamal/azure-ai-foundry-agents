"""Where the lab scripts' settings come from: one `.env` file at the repo root, next to
`.env.example`.

Every script calls `load_env()` instead of building the path itself, so moving a script
doesn't break its settings. Values already set in the shell win over the file
(python-dotenv's default).
"""

from pathlib import Path

from dotenv import load_dotenv

# src/workshop_utils/settings.py -> repo root. The package is installed editable (`-e
# .`), so this is the checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def load_env(required=False):
    """Load ENV_FILE into os.environ and return its path; with required=True a missing
    file is an error."""
    if required and not ENV_FILE.exists():
        raise FileNotFoundError(f".env file not found: {ENV_FILE} (copy .env.example)")
    load_dotenv(ENV_FILE)
    return ENV_FILE
