from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .visual_labels import IPA_FINAL_TO_ROMAN, IPA_INITIAL_TO_ROMAN, IPA_RIME_MAPPING_VERSION


def sha256_file(path: str | Path | None) -> str | None:
    """Return a stable SHA-256 digest, or None when no path was supplied."""

    if path is None:
        return None
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mapping_hash() -> str:
    return sha256_json(
        {
            "version": IPA_RIME_MAPPING_VERSION,
            "initials": IPA_INITIAL_TO_ROMAN,
            "finals": IPA_FINAL_TO_ROMAN,
        }
    )


def current_git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
