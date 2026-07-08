from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .inventory import Inventory
from .visual_labels import normalize_ipa_final, normalize_ipa_initial, roman_from_ipa


LABEL_SOURCES = frozenset({"human", "api", "derived", "legacy", "override", "model", "unknown"})
NON_PROPAGATING_STATUSES = frozenset(
    {"uncertain", "mixed_cluster", "needs_split", "unreadable", "insufficient_evidence"}
)


@dataclass(frozen=True)
class CanonicalLabelResult:
    label: dict[str, Any] | None
    conflict: dict[str, Any] | None = None
    reject_reason: str | None = None


def canonicalize_authoritative_label(
    row: dict[str, Any],
    inventory: Inventory,
    *,
    default_ipa_source: str = "unknown",
    default_tone_source: str = "unknown",
) -> CanonicalLabelResult:
    """Validate authoritative IPA/tone fields and derive the Rime layer."""

    status = str(row.get("status", "labeled") or "labeled").strip()
    if status in NON_PROPAGATING_STATUSES:
        return CanonicalLabelResult(None, reject_reason=f"api_status:{status}")
    if "ipa_initial" not in row:
        return CanonicalLabelResult(None, reject_reason="missing_ipa_initial")
    if "ipa_final" not in row:
        return CanonicalLabelResult(None, reject_reason="missing_ipa_final")

    ipa_initial = normalize_ipa_initial(row.get("ipa_initial"))
    ipa_final = normalize_ipa_final(row.get("ipa_final"))
    if not ipa_final or ipa_final.lower() in {"none", "null", "unknown"}:
        return CanonicalLabelResult(None, reject_reason="missing_ipa_final")
    if ipa_initial.lower() in {"none", "null", "unknown"}:
        return CanonicalLabelResult(None, reject_reason="invalid_ipa_initial")

    mapped = roman_from_ipa(ipa_initial, ipa_final)
    if mapped is None:
        return CanonicalLabelResult(None, reject_reason="unmapped_ipa_pair")

    tone = str(row.get("tone", "")).strip()
    if tone not in inventory.tones:
        return CanonicalLabelResult(None, reject_reason="missing_or_invalid_tone")

    derived_initial, derived_final = mapped
    derived_syllable = inventory.compose(derived_initial, derived_final, tone)
    if not inventory.is_valid_romanization(derived_syllable):
        return CanonicalLabelResult(None, reject_reason="invalid_derived_rime_pair")

    override_initial = str(row.get("rime_initial_override", "")).strip().lower()
    override_final = str(row.get("rime_final_override", "")).strip().lower()
    override_reason = str(row.get("rime_override_reason", "")).strip()
    has_override = bool(override_initial or override_final)
    if has_override and not override_reason:
        return CanonicalLabelResult(None, reject_reason="rime_override_missing_reason")
    if has_override:
        rime_initial = override_initial if override_initial or derived_initial == "" else derived_initial
        rime_final = override_final or derived_final
        rime_syllable = inventory.compose(rime_initial, rime_final, tone)
        if not inventory.is_valid_romanization(rime_syllable):
            return CanonicalLabelResult(None, reject_reason="invalid_rime_override")
        rime_source = "override"
    else:
        rime_initial, rime_final, rime_syllable = derived_initial, derived_final, derived_syllable
        rime_source = "derived"

    provided = _provided_rime(row, inventory)
    if provided is not None and provided != (derived_initial, derived_final, tone) and not has_override:
        conflict = {
            "cluster_id": row.get("cluster_id", ""),
            "cell_id": row.get("cell_id", row.get("id", "")),
            "ipa_initial": ipa_initial,
            "ipa_final": ipa_final,
            "tone": tone,
            "derived_rime": derived_syllable,
            "provided_rime": inventory.compose(*provided),
            "source": row.get("source", "unknown"),
            "conflict_reason": "provided_rime_differs_from_ipa_mapping",
        }
        return CanonicalLabelResult(None, conflict=conflict, reject_reason="ipa_rime_conflict")

    label = dict(row)
    label.update(
        {
            "status": status,
            "ipa_initial": ipa_initial,
            "ipa_final": ipa_final,
            "tone": tone,
            "rime_initial": rime_initial,
            "rime_final": rime_final,
            "rime_syllable": rime_syllable,
            "initial": rime_initial,
            "final": rime_final,
            "romanization": rime_syllable,
            "ipa_label_source": _source(row.get("ipa_label_source"), default_ipa_source),
            "rime_label_source": rime_source,
            "tone_label_source": _source(row.get("tone_label_source"), default_tone_source),
        }
    )
    if has_override:
        label["rime_initial_override"] = override_initial
        label["rime_final_override"] = override_final
        label["rime_override_reason"] = override_reason
    return CanonicalLabelResult(label)


def _source(value: Any, default: str) -> str:
    text = str(value or default).strip().lower()
    return text if text in LABEL_SOURCES else "unknown"


def _provided_rime(row: dict[str, Any], inventory: Inventory) -> tuple[str, str, str] | None:
    code = str(row.get("rime_syllable") or row.get("romanization") or "").strip()
    if code:
        return inventory.parse_romanization(code)
    has_parts = any(key in row for key in ("rime_initial", "rime_final", "initial", "final"))
    if not has_parts:
        return None
    initial = str(row.get("rime_initial", row.get("initial", ""))).strip().lower()
    final = str(row.get("rime_final", row.get("final", ""))).strip().lower()
    tone = str(row.get("tone", "")).strip()
    return initial, final, tone
