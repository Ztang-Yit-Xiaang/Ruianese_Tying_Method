from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def imwrite(path: str | Path, image: np.ndarray) -> None:
    """Write an image to a path that may contain non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image for: {path}")
    data.tofile(str(path))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def page_number_from_path(path: str | Path) -> int | None:
    name = Path(path).stem
    patterns = [
        r"(?:page|p|页面|頁面)[_\-\s]*(\d+)",
        r"_(\d{3,4})$",
        r"(\d{3,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def stable_cell_id(page_no: int | None, row_index: int) -> str:
    page = f"p{page_no:04d}" if page_no is not None else "punknown"
    return f"{page}_r{row_index:04d}"


def to_repo_path(path: str | Path, root: str | Path | None = None) -> str:
    path = Path(path)
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return str(path)
