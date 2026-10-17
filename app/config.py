"""
Application configuration: paths, limits, supported formats.
"""

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent          # flute-studio-app/
WORKSPACE_DIR = ROOT_DIR.parent    # contains flute_processor.py

DATA_DIR = Path(os.environ.get("FLUTE_STUDIO_DATA", ROOT_DIR / "data"))

UPLOADS_DIR = DATA_DIR / "uploads"
STEMS_DIR = DATA_DIR / "stems"
MIXES_DIR = DATA_DIR / "mixes"
MASTERS_DIR = DATA_DIR / "masters"

ALL_DATA_DIRS = (UPLOADS_DIR, STEMS_DIR, MIXES_DIR, MASTERS_DIR)

HOST = os.environ.get("FLUTE_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLUTE_STUDIO_PORT", "8000"))

# Upload guardrails
MAX_UPLOAD_MB = int(os.environ.get("FLUTE_STUDIO_MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus"}

# Rendered outputs are always 24-bit WAV
OUTPUT_BITS = 24

# Job log cap (lines kept in memory per job)
JOB_LOG_LIMIT = 500


def ensure_dirs() -> None:
    for path in ALL_DATA_DIRS:
        path.mkdir(parents=True, exist_ok=True)
