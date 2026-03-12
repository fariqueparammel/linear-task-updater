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

# --- Gemini (6 keys for rotation) ---
GEMINI_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY_1"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
        os.getenv("GEMINI_API_KEY_4"),
        os.getenv("GEMINI_API_KEY_5"),
        os.getenv("GEMINI_API_KEY_6"),
    ] if k
]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Models to rotate through (only models with non-zero RPD on free tier)
# 6 keys × 4 models = 24 slots before repeating
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
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

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
