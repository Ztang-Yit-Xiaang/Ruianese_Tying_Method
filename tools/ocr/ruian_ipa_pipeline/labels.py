from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .inventory import Inventory
from .io_utils import read_jsonl, to_repo_path, write_jsonl
from .label_semantics import canonicalize_authoritative_label


TRAINABLE_STATUSES = {"reviewed", "gold"}


def build_cell_labels(
    manifest_path: Path,
    cell_clusters_path: Path,
    cluster_labels_path: Path,
    output_path: Path,
    inventory: Inventory,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    statuses = statuses or TRAINABLE_STATUSES
    manifest = {row["id"]: row for row in read_jsonl(manifest_path)}
    cluster_to_cells: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(cell_clusters_path):
        cluster_to_cells[row["cluster_id"]].append(row["cell_id"])

    labels: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for label in read_jsonl(cluster_labels_path):
        if label.get("label_status") not in statuses:
            continue
        default_source = "api" if label.get("source") == "openai_api" else "human"
        result = canonicalize_authoritative_label(
            label,
            inventory,
            default_ipa_source=default_source,
            default_tone_source=default_source,
        )
        if result.label is None:
            if result.conflict:
                conflicts.append(result.conflict)
            rejected.append(
                {
                    "cluster_id": label.get("cluster_id", ""),
                    "cell_id": label.get("cell_id", ""),
                    "reject_reason": result.reject_reason,
                    "label_status": label.get("label_status", ""),
                }
            )
            continue
        canonical = result.label
        source_cluster = label.get("cluster_id")
        cell_ids = [label["cell_id"]] if label.get("cell_id") else cluster_to_cells.get(source_cluster, [])
        for cell_id in cell_ids:
            cell = manifest.get(cell_id)
            if not cell:
                continue
            image_path = _resolve_path(manifest_path, cell.get("paths", {}).get("ipa_clean", ""))
            tone_image_path = _resolve_path(manifest_path, cell.get("paths", {}).get("tone_clean", ""))
            labels.append(
                {
                    "cell_id": cell_id,
                    "cluster_id": source_cluster,
                    "page_no": cell.get("page_no"),
                    "row_index": cell.get("row_index"),
                    "image": to_repo_path(image_path, output_path.parent),
                    "tone_image": to_repo_path(tone_image_path, output_path.parent),
                    "ipa_initial": canonical["ipa_initial"],
                    "ipa_final": canonical["ipa_final"],
                    "tone": canonical["tone"],
                    "rime_initial": canonical["rime_initial"],
                    "rime_final": canonical["rime_final"],
                    "rime_syllable": canonical["rime_syllable"],
                    "initial": canonical["rime_initial"],
                    "final": canonical["rime_final"],
                    "romanization": canonical["rime_syllable"],
                    "label_status": label.get("label_status"),
                    "label_source": label.get("source", "manual"),
                    "ipa_label_source": canonical["ipa_label_source"],
                    "rime_label_source": canonical["rime_label_source"],
                    "tone_label_source": canonical["tone_label_source"],
                    "visual_label_complete": True,
                    "propagated_from_cluster": bool(source_cluster and not label.get("cell_id")),
                }
            )
    write_jsonl(output_path, labels)
    _write_label_reports(output_path.parent, conflicts, rejected)
    return labels


def _write_label_reports(
    output_dir: Path,
    conflicts: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    write_jsonl(output_dir / "label_conflicts.jsonl", conflicts)
    write_jsonl(output_dir / "label_rejected.jsonl", rejected)
    fields = [
        "cluster_id",
        "cell_id",
        "ipa_initial",
        "ipa_final",
        "tone",
        "derived_rime",
        "provided_rime",
        "source",
        "conflict_reason",
    ]
    with (output_dir / "label_conflicts.tsv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(conflicts)


def _resolve_path(anchor: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (anchor.parent / path, anchor.parent.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return anchor.parent / path
