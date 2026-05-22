import csv
import hashlib
from pathlib import Path
from typing import Iterable


TaskRow = dict[str, str | int]


def stable_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def iter_rime_entries(path: Path) -> Iterable[tuple[str, str]]:
    in_body = False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "...":
            in_body = True
            continue
        if not in_body or line == "---" or ":" in line:
            continue
        parts = raw_line.split("\t")
        text = parts[0].strip()
        romanization = parts[1].strip() if len(parts) > 1 else ""
        if text:
            yield text, romanization


def tasks_from_rime(path: Path, dialect: str, task_type: str, source: str, limit: int | None = None) -> list[TaskRow]:
    rows: list[TaskRow] = []
    seen: set[tuple[str, str]] = set()
    for index, (text, romanization) in enumerate(iter_rime_entries(path), start=1):
        key = (text, romanization)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": stable_id(dialect, task_type, source, text, romanization),
                "dialect": dialect,
                "type": task_type,
                "text": text,
                "romanization": romanization,
                "source": source,
                "priority": 100 + index,
                "status": "ready",
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def tasks_from_tsv(path: Path, dialect: str, default_type: str = "sentence") -> list[TaskRow]:
    rows: list[TaskRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader, start=1):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            task_type = (row.get("type") or default_type).strip()
            romanization = (row.get("romanization") or "").strip()
            source = (row.get("source") or path.name).strip()
            rows.append(
                {
                    "id": stable_id(dialect, task_type, source, text, romanization),
                    "dialect": dialect,
                    "type": task_type,
                    "text": text,
                    "romanization": romanization,
                    "source": source,
                    "priority": int(row.get("priority") or 200 + index),
                    "status": row.get("status") or "ready",
                }
            )
    return rows
