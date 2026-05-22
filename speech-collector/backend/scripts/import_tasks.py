import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.db import get_conn, init_db
from app.task_importer import tasks_from_rime, tasks_from_tsv


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def upsert_tasks(rows: list[dict]) -> int:
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO tasks(id, dialect, type, text, romanization, source, priority, status)
            VALUES (:id, :dialect, :type, :text, :romanization, :source, :priority, :status)
            """,
            rows,
        )
    return len(rows)


def build_default_tasks(limit_per_source: int | None = 250) -> list[dict]:
    source_specs = [
        (
            first_existing(
                ROOT / "ruianese.jie_yong_ki.dict.yaml",
                ROOT / "Ruianese_upload" / "Ruianese_Tying_Method" / "ruianese.jie_yong_ki.dict.yaml",
            ),
            "ruian",
            "word",
            "ruianese.jie_yong_ki.dict.yaml",
        ),
        (
            first_existing(
                ROOT / "rime-wenzhounese" / "wenzhounese.character_04.dict.yaml",
                ROOT.parent.parent / "rime-wenzhounese" / "wenzhounese.character_04.dict.yaml",
            ),
            "wenzhou",
            "word",
            "wenzhounese.character_04.dict.yaml",
        ),
        (
            first_existing(
                ROOT / "rime-wenzhounese" / "wenzhounese.phrases.dict.yaml",
                ROOT.parent.parent / "rime-wenzhounese" / "wenzhounese.phrases.dict.yaml",
            ),
            "wenzhou",
            "sentence",
            "wenzhounese.phrases.dict.yaml",
        ),
    ]
    rows: list[dict] = []
    for path, dialect, task_type, source in source_specs:
        if path:
            rows.extend(tasks_from_rime(path, dialect, task_type, source, limit_per_source))

    sample_sentences = BACKEND / "data" / "sample_sentences.tsv"
    if sample_sentences.exists():
        rows.extend(tasks_from_tsv(sample_sentences, "ruian", "sentence"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Wenzhounese/Ruianese collection tasks.")
    parser.add_argument("--limit-per-source", type=int, default=250)
    parser.add_argument("--sentences-tsv", type=Path, default=None)
    args = parser.parse_args()

    init_db()
    rows = build_default_tasks(args.limit_per_source)
    if args.sentences_tsv:
        rows.extend(tasks_from_tsv(args.sentences_tsv, "ruian", "sentence"))
    count = upsert_tasks(rows)
    print(f"Imported {count} tasks")


if __name__ == "__main__":
    main()
