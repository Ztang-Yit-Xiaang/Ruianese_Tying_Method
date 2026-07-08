from __future__ import annotations

from collections import defaultdict
from typing import Any


IPA_RIME_MAPPING_VERSION = "2026-06-19.1"

IPA_INITIAL_TO_ROMAN: dict[str, str] = {
    "": "",
    "p": "b",
    "pʰ": "p",
    "b": "bb",
    "m": "m",
    "f": "f",
    "v": "v",
    "t": "d",
    "tʰ": "t",
    "d": "dd",
    "n": "n",
    "l": "l",
    "k": "g",
    "kʰ": "k",
    "g": "gg",
    "ŋ": "ng",
    "h": "h",
    "ɦ": "hh",
    "tɕ": "j",
    "tɕʰ": "q",
    "dʑ": "jj",
    "ȵ": "nj",
    "ɕ": "x",
    "ts": "z",
    "tsʰ": "c",
    "dz": "zz",
    "s": "s",
    "z": "ss",
    "z̠": "zs",
    "ʑ": "zs",
}

IPA_FINAL_TO_ROMAN: dict[str, str] = {
    "a": "a",
    "o": "o",
    "ɔ": "oe",
    "æ": "ae",
    "e": "e",
    "ɛ": "eh",
    "i": "i",
    "u": "u",
    "ʉ": "yu",
    "y̟u": "yu",
    "ao": "ao",
    "ai": "ai",
    "əʉ": "ou",
    "ei": "ei",
    "ia": "ia",
    "iao": "iao",
    "iəʉ": "iou",
    "ie": "ie",
    "iæ": "iae",
    "io": "io",
    "uai": "uai",
    "uo": "uo",
    "uɔ": "uoe",
    "yo": "yo",
    "yue": "yue",
    "yɔ": "yoe",
    "ang": "ang",
    "eng": "eng",
    "ong": "ong",
    "iang": "iang",
    "iong": "iong",
    "ŋ": "ng",
}

INITIAL_ALIASES: dict[str, str] = {
    "zero": "",
    "zero initial": "",
    "none": "",
    "null": "",
    "-": "",
    "∅": "",
    "Ø": "",
    "ø": "",
    "ng": "ŋ",
    "ny": "ȵ",
    "nj": "ȵ",
    "tsh": "tsʰ",
    "ts h": "tsʰ",
    "ts'": "tsʰ",
    "tsʼ": "tsʰ",
    "ch": "tsʰ",
    "tɕh": "tɕʰ",
    "tɕ h": "tɕʰ",
    "tɕ'": "tɕʰ",
    "tɕʼ": "tɕʰ",
    "kh": "kʰ",
    "ph": "pʰ",
    "th": "tʰ",
    "zh": "ʑ",
}

FINAL_ALIASES: dict[str, str] = {
    "oe": "ɔ",
    "ae": "æ",
    "eh": "ɛ",
    "ou": "əʉ",
    "iou": "iəʉ",
    "iae": "iæ",
    "uoe": "uɔ",
    "yoe": "yɔ",
    "ng": "ŋ",
}


def normalize_ipa_initial(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    compact = " ".join(text.split())
    key = compact.lower()
    return INITIAL_ALIASES.get(key, compact)


def normalize_ipa_final(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    compact = "".join(text.split())
    key = compact.lower()
    return FINAL_ALIASES.get(key, compact)


def roman_from_ipa(ipa_initial: Any, ipa_final: Any) -> tuple[str, str] | None:
    initial = normalize_ipa_initial(ipa_initial)
    final = normalize_ipa_final(ipa_final)
    if initial not in IPA_INITIAL_TO_ROMAN or final not in IPA_FINAL_TO_ROMAN:
        return None
    return IPA_INITIAL_TO_ROMAN[initial], IPA_FINAL_TO_ROMAN[final]


def infer_ipa_initial(roman_initial: Any) -> str | None:
    return _unique_reverse(IPA_INITIAL_TO_ROMAN).get(str(roman_initial or "").strip())


def infer_ipa_final(roman_final: Any) -> str | None:
    return _unique_reverse(IPA_FINAL_TO_ROMAN).get(str(roman_final or "").strip())


def legacy_fill_visual_fields_from_roman(row: dict[str, Any], roman_initial: str, roman_final: str) -> bool:
    """Best-effort legacy migration only; never use its output as training truth."""
    ipa_initial = normalize_ipa_initial(row.get("ipa_initial", ""))
    ipa_final = normalize_ipa_final(row.get("ipa_final", ""))

    if ipa_initial or ipa_final:
        row["ipa_initial"] = ipa_initial
        row["ipa_final"] = ipa_final
        if has_complete_visual_fields(row, roman_initial, roman_final):
            return True
        row["ipa_initial"] = ""
        row["ipa_final"] = ""

    inferred_initial = infer_ipa_initial(roman_initial)
    inferred_final = infer_ipa_final(roman_final)
    if inferred_initial is None or inferred_final is None:
        row["ipa_initial"] = ""
        row["ipa_final"] = ""
        return False
    row["ipa_initial"] = inferred_initial
    row["ipa_final"] = inferred_final
    return True


def fill_visual_fields_from_roman(row: dict[str, Any], roman_initial: str, roman_final: str) -> bool:
    """Deprecated compatibility alias for explicit legacy migration tools only."""

    return legacy_fill_visual_fields_from_roman(row, roman_initial, roman_final)


def has_complete_visual_fields(
    row: dict[str, Any],
    roman_initial: str | None = None,
    roman_final: str | None = None,
) -> bool:
    ipa_initial = normalize_ipa_initial(row.get("ipa_initial", ""))
    ipa_final = normalize_ipa_final(row.get("ipa_final", ""))
    if ipa_initial == "" and str(row.get("ipa_initial", "")) not in {"", "0", "None", "none", "null"}:
        # Empty string is a valid zero initial; the final still decides completeness.
        pass
    if not ipa_final:
        return False
    mapped = roman_from_ipa(ipa_initial, ipa_final)
    if mapped is None:
        return False
    if roman_initial is not None and roman_final is not None and mapped != (roman_initial, roman_final):
        return False
    return True


def apply_review_visual_fields(
    row: dict[str, Any],
    review: dict[str, str],
    roman_initial: str,
    roman_final: str,
    cluster_id: str,
) -> bool:
    raw_initial = review.get("correct_ipa_initial", "")
    raw_final = review.get("correct_ipa_final", "")
    has_review_visual = bool(raw_initial.strip() or raw_final.strip())
    if has_review_visual:
        if not raw_final.strip():
            raise ValueError(f"{cluster_id}: correct_ipa_final is required when correcting IPA visual labels")
        ipa_initial = normalize_ipa_initial(raw_initial)
        ipa_final = normalize_ipa_final(raw_final)
        mapped = roman_from_ipa(ipa_initial, ipa_final)
        if mapped is None:
            raise ValueError(f"{cluster_id}: invalid IPA visual label: {ipa_initial}+{ipa_final}")
        if mapped != (roman_initial, roman_final):
            raise ValueError(
                f"{cluster_id}: IPA visual label {ipa_initial}+{ipa_final} maps to "
                f"{mapped[0]}+{mapped[1]}, not {roman_initial}+{roman_final}"
            )
        row["ipa_initial"] = ipa_initial
        row["ipa_final"] = ipa_final
        return True
    if not has_complete_visual_fields(row, roman_initial, roman_final):
        raise ValueError(
            f"{cluster_id}: authoritative IPA labels are missing; "
            "Rime romanization cannot be used to infer CNN training truth"
        )
    row["ipa_initial"] = normalize_ipa_initial(row.get("ipa_initial", ""))
    row["ipa_final"] = normalize_ipa_final(row.get("ipa_final", ""))
    return True


def ipa_initial_inventory() -> list[str]:
    values = [value for value in IPA_INITIAL_TO_ROMAN if value]
    return sorted(values)


def ipa_final_inventory() -> list[str]:
    return sorted(IPA_FINAL_TO_ROMAN)


def _unique_reverse(mapping: dict[str, str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, value in mapping.items():
        grouped[value].append(key)
    return {value: keys[0] for value, keys in grouped.items() if len(keys) == 1}
