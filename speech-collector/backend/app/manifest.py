import csv
import io
import json
import sqlite3
from typing import Iterable


MANIFEST_FIELDS = [
    "audio_filepath",
    "text",
    "dialect",
    "speaker_id",
    "duration",
    "split",
    "source_task_id",
]


def manifest_rows(conn: sqlite3.Connection, include_review: bool = False) -> list[dict]:
    statuses = ("approved", "needs_review") if include_review else ("approved",)
    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"""
        SELECT s.wav_audio_path, s.raw_audio_path, s.duration_seconds, s.speaker_id,
               s.task_id, s.dialect, t.text
        FROM submissions s
        JOIN tasks t ON t.id = s.task_id
        WHERE s.review_status IN ({placeholders})
        ORDER BY s.created_at ASC
        """,
        statuses,
    ).fetchall()

    return [
        {
            "audio_filepath": row["wav_audio_path"] or row["raw_audio_path"],
            "text": row["text"],
            "dialect": row["dialect"],
            "speaker_id": row["speaker_id"],
            "duration": row["duration_seconds"],
            "split": "train",
            "source_task_id": row["task_id"],
        }
        for row in rows
    ]


def render_jsonl(rows: Iterable[dict]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


def render_csv(rows: Iterable[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
