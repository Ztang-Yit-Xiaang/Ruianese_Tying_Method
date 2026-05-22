from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
RAW_AUDIO_DIR = STORAGE_DIR / "raw"
WAV_AUDIO_DIR = STORAGE_DIR / "wav"
DATABASE_PATH = DATA_DIR / "collector_app.sqlite3"
CONSENT_VERSION = "research-consent-v1"


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    WAV_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
