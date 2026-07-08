from __future__ import annotations

import csv
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inventory import Inventory
from .io_utils import read_jsonl, write_jsonl
from .label_semantics import canonicalize_authoritative_label
from .visual_labels import normalize_ipa_final, normalize_ipa_initial


REVIEW_FIELDS = [
    "cluster_id",
    "current_romanization",
    "current_ipa_initial",
    "current_ipa_final",
    "current_tone",
    "confidence",
    "needs_review",
    "issue",
    "contact_sheet",
    "decision",
    "correct_romanization",
    "correct_ipa_initial",
    "correct_ipa_final",
    "correct_tone",
    "rime_initial_override",
    "rime_final_override",
    "rime_override_reason",
    "review_status",
    "review_note",
]

REQUIRED_REVIEW_FIELDS = [
    "cluster_id",
    "current_romanization",
    "confidence",
    "needs_review",
    "issue",
    "contact_sheet",
    "decision",
    "correct_romanization",
    "review_status",
    "review_note",
]

ALLOWED_DECISIONS = {"", "accept", "incorrect", "correct", "reject", "mixed", "gold"}
DECISION_ALIASES = {"correct": "incorrect"}


def build_review_queue(
    labels_path: Path,
    cluster_manifest_path: Path,
    out_dir: Path,
    inventory: Inventory,
    confidence_min: float = 0.95,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    labels = read_jsonl(labels_path)
    clusters = {row["cluster_id"]: row for row in read_jsonl(cluster_manifest_path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = out_dir / "review_contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "review_queue.tsv"
    existing_rows = _read_existing_review_rows(queue_path)

    rows: list[dict[str, str]] = []
    for label in labels:
        if label.get("label_status") != "weak":
            continue
        cluster_id = str(label.get("cluster_id", ""))
        sheet_rel = _copy_contact_sheet(label, clusters.get(cluster_id, {}), cluster_manifest_path, sheet_dir)
        row = {
            "cluster_id": cluster_id,
            "current_romanization": str(label.get("romanization", "")),
            "current_ipa_initial": str(label.get("ipa_initial", "")),
            "current_ipa_final": str(label.get("ipa_final", "")),
            "current_tone": str(label.get("tone", "")),
            "confidence": str(label.get("confidence", "")),
            "needs_review": str(bool(label.get("needs_review"))).lower(),
            "issue": _issue(label, inventory, confidence_min),
            "contact_sheet": sheet_rel,
            "decision": "",
            "correct_romanization": "",
            "correct_ipa_initial": "",
            "correct_ipa_final": "",
            "correct_tone": "",
            "rime_initial_override": "",
            "rime_final_override": "",
            "rime_override_reason": "",
            "review_status": "",
            "review_note": "",
        }
        previous = existing_rows.get(cluster_id)
        if previous:
            for field in (
                "decision",
                "correct_romanization",
                "correct_ipa_initial",
                "correct_ipa_final",
                "correct_tone",
                "rime_initial_override",
                "rime_final_override",
                "rime_override_reason",
                "review_status",
                "review_note",
            ):
                row[field] = previous.get(field, "")
        rows.append(row)

    rows.sort(key=_review_sort_key)
    with queue_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    report_path = out_dir / "review_report.md"
    summary = _review_summary(rows)
    report_path.write_text(_render_report(rows, summary), encoding="utf-8", newline="\n")
    summary.update({"queue": str(queue_path), "report": str(report_path), "contact_sheets": str(sheet_dir)})
    return rows, summary


def _read_existing_review_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read a previous queue so regeneration preserves all human-entered fields."""

    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            cluster_id = str(row.get("cluster_id", "")).strip()
            if not cluster_id:
                continue
            rows[cluster_id] = {field: str(row.get(field, "") or "").strip() for field in REVIEW_FIELDS}
    return rows


def apply_review_decisions(
    labels_path: Path,
    review_path: Path,
    output_path: Path,
    inventory: Inventory,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = read_jsonl(labels_path)
    decisions = _read_review_decisions(review_path)
    now = datetime.now(timezone.utc).isoformat()
    changed = 0
    accepted = 0
    incorrect = 0
    rejected = 0
    mixed = 0
    gold = 0
    unchanged = 0
    out: list[dict[str, Any]] = []

    for label in labels:
        item = deepcopy(label)
        cluster_id = str(item.get("cluster_id", ""))
        review = decisions.get(cluster_id)
        if review is None or not review.get("decision"):
            unchanged += 1
            out.append(item)
            continue

        decision = review["decision"]
        note = review.get("review_note", "")
        item["review_decision"] = decision
        item["review_note"] = note
        item["reviewed_at"] = now

        if decision == "accept":
            item = _canonical_review_item(item, {}, inventory, cluster_id)
            item["label_status"] = "reviewed"
            item["needs_review"] = False
            accepted += 1
        elif decision == "incorrect":
            _apply_correction(item, review, inventory, cluster_id)
            item["label_status"] = "reviewed"
            item["needs_review"] = False
            incorrect += 1
        elif decision == "gold":
            has_visual_correction = bool(
                review.get("correct_ipa_initial", "").strip()
                or review.get("correct_ipa_final", "").strip()
                or review.get("correct_tone", "").strip()
            )
            if review.get("correct_romanization") or has_visual_correction or review.get("rime_override_reason"):
                _apply_correction(item, review, inventory, cluster_id)
            else:
                item = _canonical_review_item(item, {}, inventory, cluster_id)
            item["label_status"] = "gold"
            item["needs_review"] = False
            gold += 1
        elif decision == "reject":
            item["label_status"] = "weak"
            item["needs_review"] = True
            rejected += 1
        elif decision == "mixed":
            item["label_status"] = "mixed"
            item["status"] = "mixed_cluster"
            item["needs_review"] = True
            mixed += 1
        else:
            raise ValueError(f"Unsupported decision for {cluster_id}: {decision}")
        changed += 1
        out.append(item)

    write_jsonl(output_path, out)
    summary = {
        "total": len(out),
        "changed": changed,
        "unchanged": unchanged,
        "accepted": accepted,
        "incorrect": incorrect,
        "corrected": incorrect,
        "gold": gold,
        "rejected": rejected,
        "mixed": mixed,
        "output": str(output_path),
    }
    return out, summary


def _read_review_decisions(path: Path) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = [field for field in REQUIRED_REVIEW_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Review TSV is missing required columns: {', '.join(missing)}")
        for row in reader:
            cluster_id = row.get("cluster_id", "").strip()
            if not cluster_id:
                continue
            if cluster_id in decisions:
                raise ValueError(f"Duplicate review row for cluster_id: {cluster_id}")
            decision = row.get("decision", "").strip().lower()
            if decision not in ALLOWED_DECISIONS:
                raise ValueError(f"Invalid decision for {cluster_id}: {decision}")
            decision = DECISION_ALIASES.get(decision, decision)
            clean = {key: (row.get(key, "") or "").strip() for key in REVIEW_FIELDS}
            clean["decision"] = decision
            decisions[cluster_id] = clean
    return decisions


def _copy_contact_sheet(label: dict[str, Any], cluster: dict[str, Any], cluster_manifest_path: Path, sheet_dir: Path) -> str:
    cluster_id = str(label.get("cluster_id") or cluster.get("cluster_id") or "unknown")
    rel = str(label.get("contact_sheet") or cluster.get("contact_sheet") or "")
    if not rel:
        return ""
    src = _resolve_contact_sheet(cluster_manifest_path, rel)
    dst = sheet_dir / f"{cluster_id}.png"
    if src.exists():
        shutil.copyfile(src, dst)
    return f"review_contact_sheets/{dst.name}"


def _resolve_contact_sheet(cluster_manifest_path: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    for candidate in (cluster_manifest_path.parent / path, cluster_manifest_path.parent.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return cluster_manifest_path.parent / path


def _issue(label: dict[str, Any], inventory: Inventory, confidence_min: float) -> str:
    issues: list[str] = []
    romanization = str(label.get("romanization", ""))
    confidence = _confidence(label)
    if bool(label.get("needs_review")):
        issues.append("needs_review")
    if not inventory.is_valid_romanization(romanization):
        issues.append("invalid_romanization")
    if confidence is None or confidence < confidence_min:
        issues.append(f"confidence<{confidence_min:g}")
    notes = str(label.get("notes", "")).lower()
    if any(token in notes for token in ["mixed", "unclear", "although", "first", "second", "variation", "mostly", "does not fit"]):
        issues.append("note_suggests_manual_check")
    return ";".join(issues) if issues else "weak_unpromoted"


def _review_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row["issue"].split(";"):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return {"total_review_rows": len(rows), "issue_counts": issue_counts}


def _render_report(rows: list[dict[str, str]], summary: dict[str, Any]) -> str:
    lines = [
        "# IPA Label Review Report",
        "",
        f"Review rows: {summary['total_review_rows']}",
        "",
        "## Issue Counts",
        "",
    ]
    for issue, count in sorted(summary["issue_counts"].items()):
        lines.append(f"- {issue}: {count}")
    lines.extend(
        [
            "",
            "## How To Fill review_queue.tsv",
            "",
            "- accept: current romanization is correct; promote to reviewed.",
            "- incorrect: correct the authoritative IPA initial/final/tone; Rime is derived automatically.",
            "- gold: use only when the IPA visual label and tone are especially certain.",
            "- rime override: fill override fields and a reason only for a genuine exceptional spelling.",
            "- reject: keep weak; do not train.",
            "- mixed: cluster contains multiple syllables; do not propagate as a cluster.",
            "",
            "## Queue",
            "",
            "| cluster_id | current | confidence | issue | contact_sheet |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['cluster_id']} | {row['current_romanization']} | {row['confidence']} | "
            f"{row['issue']} | {row['contact_sheet']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _review_sort_key(row: dict[str, str]) -> tuple[int, float, str]:
    issue = row["issue"]
    priority = 0
    if "invalid_romanization" in issue or "needs_review" in issue:
        priority = 0
    elif "note_suggests_manual_check" in issue:
        priority = 1
    elif "confidence<" in issue:
        priority = 2
    else:
        priority = 3
    try:
        confidence = float(row["confidence"])
    except ValueError:
        confidence = 0.0
    return (priority, confidence, row["cluster_id"])


def _apply_correction(item: dict[str, Any], review: dict[str, str], inventory: Inventory, cluster_id: str) -> None:
    corrected = _canonical_review_item(item, review, inventory, cluster_id)
    item.clear()
    item.update(corrected)


def _require_valid(item: dict[str, Any], inventory: Inventory, cluster_id: str) -> None:
    _canonical_review_item(item, {}, inventory, cluster_id)


def _canonical_review_item(
    item: dict[str, Any],
    review: dict[str, str],
    inventory: Inventory,
    cluster_id: str,
) -> dict[str, Any]:
    candidate = dict(item)
    if review:
        raw_initial = review.get("correct_ipa_initial", "").strip()
        raw_final = review.get("correct_ipa_final", "").strip()
        raw_tone = review.get("correct_tone", "").strip()
        has_authoritative_correction = bool(raw_initial or raw_final or raw_tone)
        if raw_initial or raw_final:
            if raw_initial:
                candidate["ipa_initial"] = normalize_ipa_initial(raw_initial)
            candidate["ipa_final"] = normalize_ipa_final(raw_final or candidate.get("ipa_final", ""))
        if raw_tone:
            candidate["tone"] = raw_tone
        for field in ("rime_initial_override", "rime_final_override", "rime_override_reason"):
            if review.get(field, "").strip():
                candidate[field] = review[field].strip()
        provided = review.get("correct_romanization", "").strip()
        if provided:
            candidate["romanization"] = provided
            candidate.pop("rime_syllable", None)
        elif has_authoritative_correction:
            for field in ("rime_initial", "rime_final", "rime_syllable", "initial", "final", "romanization"):
                candidate.pop(field, None)
    candidate["status"] = "labeled"
    candidate["ipa_label_source"] = "human"
    candidate["tone_label_source"] = "human"
    result = canonicalize_authoritative_label(
        candidate,
        inventory,
        default_ipa_source="human",
        default_tone_source="human",
    )
    if result.label is None:
        reason = result.reject_reason or "invalid authoritative label"
        if result.conflict:
            reason += f" ({result.conflict['provided_rime']} vs {result.conflict['derived_rime']})"
        raise ValueError(f"{cluster_id}: {reason}")
    return result.label


def _confidence(row: dict[str, Any]) -> float | None:
    try:
        return float(row.get("confidence"))
    except (TypeError, ValueError):
        return None
