from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .visual_labels import roman_from_ipa


TARGET_CLUSTERS = {
    "cluster_0073",
    "cluster_0085",
    "cluster_0101",
    "cluster_0107",
    "cluster_0119",
    "cluster_0129",
    "cluster_0140",
    "cluster_0153",
}

REVIEW_COLUMNS = [
    "cluster_id",
    "current_rime",
    "current_ipa_initial",
    "current_ipa_final",
    "current_tone",
    "suggested_ipa_initial",
    "suggested_ipa_final",
    "suggested_tone",
    "derived_rime",
    "suggestion_confidence",
    "requires_visual_review",
    "reason",
    "contact_sheet",
]

_CANDIDATES: dict[str, tuple[str, str, str, float, str]] = {
    "kung1": ("k", "ong", "1", 0.65, "Contact sheet reads k+uŋ; current mapping gives gong1. Confirm the tone mark and absence of aspiration."),
    "nyi7": ("ȵ", "i", "7", 0.55, "Rime ny is obsolete here; candidate assumes the visible initial is IPA ȵ."),
    "nyie8": ("ȵ", "ie", "8", 0.55, "Candidate follows IPA ȵ -> Rime nj, pending image confirmation."),
    "nyiou8": ("ȵ", "iəʉ", "8", 0.5, "Candidate follows IPA ȵ and iəʉ -> njiou, pending image confirmation."),
    "fung3": ("f", "ong", "3", 0.45, "Candidate assumes the printed final is ong; uŋ/oŋ-like forms need visual confirmation."),
    "sye4": ("s", "yue", "4", 0.45, "Image reads s+yə; candidate visual class is yue, but syue4 is not yet in the legal-pair evidence table."),
    "dung6": ("d", "ong", "6", 0.65, "Contact sheet reads d+uŋ; current mapping gives ddong6. Confirm the corner tone mark."),
}


def generate_review_fix_suggestions(
    review_tsv: Path,
    output_tsv: Path,
    *,
    cluster_manifest_path: Path | None = None,
    cluster_labels_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Create non-applying suggestions for the eight known ambiguous review rows."""

    with review_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        review_rows = list(csv.DictReader(handle, delimiter="\t"))
    manifests = _index_jsonl(cluster_manifest_path)
    labels = _index_jsonl(cluster_labels_path)
    suggestions: list[dict[str, Any]] = []
    for review in review_rows:
        cluster_id = str(review.get("cluster_id", ""))
        if cluster_id not in TARGET_CLUSTERS:
            continue
        label = labels.get(cluster_id, {})
        manifest = manifests.get(cluster_id, {})
        current_rime = str(
            review.get("correct_romanization")
            or review.get("current_romanization")
            or label.get("rime_syllable")
            or label.get("romanization")
            or ""
        ).strip()
        candidate = _CANDIDATES.get(current_rime.lower())
        if candidate is None:
            suggested_initial = suggested_final = suggested_tone = derived_rime = ""
            confidence = 0.0
            reason = _mandatory_reason(current_rime)
        else:
            suggested_initial, suggested_final, suggested_tone, confidence, reason = candidate
            mapped = roman_from_ipa(suggested_initial, suggested_final)
            derived_rime = f"{mapped[0]}{mapped[1]}{suggested_tone}" if mapped else ""
        suggestions.append(
            {
                "cluster_id": cluster_id,
                "current_rime": current_rime,
                "current_ipa_initial": review.get("current_ipa_initial") or label.get("ipa_initial", ""),
                "current_ipa_final": review.get("current_ipa_final") or label.get("ipa_final", ""),
                "current_tone": _current_tone(review, label, current_rime),
                "suggested_ipa_initial": suggested_initial,
                "suggested_ipa_final": suggested_final,
                "suggested_tone": suggested_tone,
                "derived_rime": derived_rime,
                "suggestion_confidence": confidence,
                "requires_visual_review": True,
                "reason": reason,
                "contact_sheet": review.get("contact_sheet") or manifest.get("contact_sheet", ""),
            }
        )
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(suggestions)
    return suggestions


def _index_jsonl(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {str(row.get("cluster_id")): row for row in read_jsonl(path) if row.get("cluster_id")}


def _mandatory_reason(current_rime: str) -> str:
    if current_rime.lower() == "kung1":
        return "Initial voicing/aspiration and the final must be read from the contact sheet; no string correction is safe."
    if current_rime.lower() == "dung6":
        return "The image must distinguish IPA d from other stops before choosing Rime d or dd."
    if current_rime.lower() == "zioe":
        return "Tone is missing and ioe is not established; inspect both the IPA body and corner tone mark."
    return "No safe IPA-layer candidate can be derived from this Rime string."


def _current_tone(review: dict[str, Any], label: dict[str, Any], current_rime: str) -> str:
    explicit = str(review.get("current_tone") or "").strip()
    if explicit in set("12345678"):
        return explicit
    label_tone = str(label.get("tone") or "").strip()
    if label_tone in set("12345678"):
        return label_tone
    match = re.search(r"([1-8])$", current_rime)
    return match.group(1) if match else ""
