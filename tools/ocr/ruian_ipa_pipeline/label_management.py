from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .inventory import Inventory
from .io_utils import read_jsonl, write_jsonl
from .label_semantics import canonicalize_authoritative_label


def summarize_labels(labels_path: Path, inventory: Inventory) -> dict[str, Any]:
    rows = read_jsonl(labels_path)
    statuses = Counter(str(row.get("label_status", "missing")) for row in rows)
    sources = Counter(str(row.get("source", "missing")) for row in rows)
    needs_review = sum(1 for row in rows if bool(row.get("needs_review")))
    invalid = sum(1 for row in rows if not inventory.is_valid_romanization(str(row.get("romanization", ""))))
    confidence_values = [_confidence(row) for row in rows if _confidence(row) is not None]
    summary: dict[str, Any] = {
        "total": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "source_counts": dict(sorted(sources.items())),
        "needs_review": needs_review,
        "invalid_romanization": invalid,
    }
    if confidence_values:
        summary["confidence_min"] = min(confidence_values)
        summary["confidence_max"] = max(confidence_values)
        summary["confidence_avg"] = round(sum(confidence_values) / len(confidence_values), 6)
    return summary


def promote_labels(
    labels_path: Path,
    output_path: Path,
    inventory: Inventory,
    confidence_min: float = 0.95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(labels_path)
    promoted = 0
    kept_reviewed_or_gold = 0
    invalid = 0
    needs_review = 0
    low_confidence = 0
    wrong_source = 0
    unsafe_status = 0
    incomplete_ipa = 0
    out: list[dict[str, Any]] = []
    for row in rows:
        item = deepcopy(row)
        if item.get("label_status") in {"reviewed", "gold"}:
            kept_reviewed_or_gold += 1
            out.append(item)
            continue
        canonical = canonicalize_authoritative_label(
            item, inventory, default_ipa_source="api", default_tone_source="api"
        )
        valid = canonical.label is not None
        confidence = _confidence(item) or 0.0
        if item.get("source") != "openai_api":
            wrong_source += 1
            item["label_status"] = "weak"
        elif item.get("status", "labeled") != "labeled":
            unsafe_status += 1
            item["label_status"] = "weak"
        elif not valid:
            invalid += 1
            if canonical.reject_reason and canonical.reject_reason.startswith("missing_"):
                incomplete_ipa += 1
            item["label_status"] = "weak"
        elif bool(item.get("needs_review")):
            needs_review += 1
            item["label_status"] = "weak"
        elif confidence < confidence_min:
            low_confidence += 1
            item["label_status"] = "weak"
        else:
            promoted += 1
            item = canonical.label or item
            item["label_status"] = "reviewed"
            item["promotion_reason"] = (
                f"source=openai_api; valid_legal_romanization; "
                f"needs_review=false; confidence>={confidence_min:g}"
            )
        out.append(item)
    write_jsonl(output_path, out)
    summary = summarize_labels(output_path, inventory)
    summary.update(
        {
            "promoted": promoted,
            "kept_reviewed_or_gold": kept_reviewed_or_gold,
            "blocked_wrong_source": wrong_source,
            "blocked_invalid_romanization": invalid,
            "blocked_needs_review": needs_review,
            "blocked_low_confidence": low_confidence,
            "blocked_unsafe_status": unsafe_status,
            "blocked_incomplete_ipa": incomplete_ipa,
            "confidence_min_for_promotion": confidence_min,
        }
    )
    return out, summary


def _confidence(row: dict[str, Any]) -> float | None:
    value = row.get("confidence")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
