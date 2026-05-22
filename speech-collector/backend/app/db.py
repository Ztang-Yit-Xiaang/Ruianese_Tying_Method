import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DATABASE_PATH, ensure_directories


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS invitations (
    code TEXT PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    dialect_hint TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS speakers (
    id TEXT PRIMARY KEY,
    invite_code TEXT NOT NULL,
    region TEXT NOT NULL,
    age_group TEXT NOT NULL,
    gender TEXT NOT NULL DEFAULT '',
    dialect_point TEXT NOT NULL,
    consent_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(invite_code) REFERENCES invitations(code)
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    dialect TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('word', 'sentence')),
    text TEXT NOT NULL,
    romanization TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_filters ON tasks(dialect, type, status, priority);

CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY,
    invite_code TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    dialect TEXT NOT NULL,
    raw_audio_path TEXT NOT NULL,
    wav_audio_path TEXT NOT NULL DEFAULT '',
    duration_seconds REAL NOT NULL DEFAULT 0,
    browser_info TEXT NOT NULL DEFAULT '{}',
    consent_version TEXT NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewer_notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(invite_code) REFERENCES invitations(code),
    FOREIGN KEY(speaker_id) REFERENCES speakers(id),
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE INDEX IF NOT EXISTS idx_submissions_review ON submissions(review_status, dialect, task_id);
"""


def init_db(path: Path = DATABASE_PATH) -> None:
    ensure_directories()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT OR IGNORE INTO invitations(code, label, dialect_hint)
            VALUES ('DEMO-RUIAN', 'Demo contributor', 'ruian'),
                   ('DEMO-WENZHOU', 'Demo contributor', 'wenzhou')
            """
        )
        conn.commit()


@contextmanager
def get_conn(path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None
