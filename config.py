import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# --- GitHub (Read-Only) ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_ORG = os.getenv("GITHUB_ORG", "")

# --- Linear ---
LINEAR_API_KEY = os.getenv("LINEAR_API_KEY", "")
LINEAR_TEAM_ID = os.getenv("LINEAR_TEAM_ID", "")
LINEAR_API_URL = "https://api.linear.app/graphql"

# --- Gemini (unlimited keys for rotation) ---
# Collects all GEMINI_API_KEY_* env vars (GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...)
GEMINI_KEYS = sorted(
    [v for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY_") and v],
    key=lambda _: 0,  # preserve discovery order
)
# Fallback: also check plain GEMINI_API_KEY if no numbered keys found
if not GEMINI_KEYS:
    _single = os.getenv("GEMINI_API_KEY")
    if _single:
        GEMINI_KEYS = [_single]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Models to rotate through (only models with non-zero RPD on free tier)
# N keys × 4 models = N×4 slots before repeating
GEMINI_MODELS = [
    "gemini-3-flash-preview",     # Gemini 3 Flash, 5 RPM, 20 RPD
    "gemini-3.1-flash-lite-preview",  # 15 RPM, 500 RPD — best for volume
    "gemini-2.5-flash",           # 5 RPM, 20 RPD
    "gemini-2.5-flash-lite",      # 10 RPM, 20 RPD
]

# --- Tuning ---
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "120"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
BATCH_TIMEOUT_SECONDS = int(os.getenv("BATCH_TIMEOUT_SECONDS", "3600"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
WORKSPACE_REFRESH_SECONDS = int(os.getenv("WORKSPACE_REFRESH_SECONDS", "1800"))  # 30 min
AUDIT_INTERVAL_SECONDS = int(os.getenv("AUDIT_INTERVAL_SECONDS", "600"))  # 10 min — missed commit audit
MODEL_REFRESH_SECONDS = int(os.getenv("MODEL_REFRESH_SECONDS", "3600"))  # 1 hour — Gemini model discovery
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VERBOSE_REPO_SCAN = os.getenv("VERBOSE_REPO_SCAN", "false").lower() == "true"  # Log each repo's SHA check

# --- Per-user exclusion ---
# Comma-separated GitHub usernames whose commits should NOT create Linear tasks.
# Their commits are still tracked (SHA advances) but skipped during batch processing.
# Example: EXCLUDED_GITHUB_USERS=bot-user,intern-account
EXCLUDED_GITHUB_USERS = {
    u.strip().lower()
    for u in os.getenv("EXCLUDED_GITHUB_USERS", "").split(",")
    if u.strip()
}

# --- Paths ---
STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


def validate():
    """Fail fast if required config is missing."""
    missing = []
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not GITHUB_ORG:
        missing.append("GITHUB_ORG")
    if not LINEAR_API_KEY:
        missing.append("LINEAR_API_KEY")
    if not LINEAR_TEAM_ID:
        missing.append("LINEAR_TEAM_ID")
    if not GEMINI_KEYS:
        missing.append("GEMINI_API_KEY_1 (at least one Gemini key)")

    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def setup_logging():
    """Configure structured logging to console and file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "sync.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
