from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TONES = [str(i) for i in range(1, 9)]


LEGAL_PAIR_STATUSES = (
    "invalid",
    "allowed_unattested",
    "attested_unreviewed",
    "reviewed",
    "observed_training",
)


@dataclass(frozen=True)
class LegalPairRecord:
    initial: str
    final: str
    tone: str | None = None
    phonologically_allowed: bool = True
    attested_in_dictionary: bool = False
    human_reviewed: bool = False
    observed_in_training: bool = False
    source: str = ""
    notes: str = ""

    @property
    def status(self) -> str:
        if self.observed_in_training:
            return "observed_training"
        if self.human_reviewed:
            return "reviewed"
        if self.attested_in_dictionary:
            return "attested_unreviewed"
        if self.phonologically_allowed:
            return "allowed_unattested"
        return "invalid"


@dataclass(frozen=True)
class Inventory:
    initials: tuple[str, ...]
    finals: tuple[str, ...]
    tones: tuple[str, ...] = tuple(DEFAULT_TONES)
    legal_pairs: frozenset[tuple[str, str]] | None = None
    legal_pair_records: tuple[LegalPairRecord, ...] = ()

    def parse_romanization(self, code: str) -> tuple[str, str, str] | None:
        code = normalize_code(code)
        match = re.search(r"([1-8])$", code)
        if not match:
            return None
        tone = match.group(1)
        body = code[: -len(tone)]
        candidates: list[tuple[int, int, str, str, str]] = []
        for ini in ("", *self.initials):
            if not body.startswith(ini):
                continue
            fin = body[len(ini) :]
            if fin in self.finals:
                candidates.append((len(ini), len(fin), ini, fin, tone))
        if not candidates:
            return None
        # Prefer explicit finals such as ng over empty/ambiguous splits.
        _, _, ini, fin, tone = sorted(candidates, reverse=True)[0]
        return ini, fin, tone

    def is_valid_romanization(self, code: str) -> bool:
        return self.validate_romanization(code)["status"] != "invalid"

    def validate_romanization(self, code: str) -> dict[str, str | bool | None]:
        parsed = self.parse_romanization(code)
        if parsed is None:
            return {"syllable": normalize_code(code), "status": "invalid", "source": None, "notes": "unparseable"}
        initial, final, tone = parsed
        if self.legal_pairs is None:
            return {"syllable": self.compose(initial, final, tone), "status": "allowed_unattested", "source": None, "notes": "no legal-pair file"}
        matching = [
            record
            for record in self.legal_pair_records
            if record.initial == initial and record.final == final and (record.tone is None or record.tone == tone)
        ]
        if not matching:
            return {"syllable": self.compose(initial, final, tone), "status": "invalid", "source": None, "notes": "pair not listed"}
        rank = {status: index for index, status in enumerate(LEGAL_PAIR_STATUSES)}
        record = max(matching, key=lambda item: rank[item.status])
        return {
            "syllable": self.compose(initial, final, tone),
            "status": record.status,
            "source": record.source or None,
            "notes": record.notes,
        }

    def compose(self, initial: str, final: str, tone: str | int) -> str:
        return f"{initial}{final}{tone}"


def normalize_code(code: str) -> str:
    return code.strip().lower().replace(" ", "").replace("'", "")


def load_inventory(schema_path: str | Path | None, legal_pairs_path: str | Path | None = None) -> Inventory:
    records = load_legal_pair_records(legal_pairs_path) if legal_pairs_path else ()
    legal_pairs = frozenset((record.initial, record.final) for record in records if record.status != "invalid") if legal_pairs_path else None
    if schema_path is None:
        return Inventory(
            initials=(), finals=(), tones=tuple(DEFAULT_TONES), legal_pairs=legal_pairs, legal_pair_records=records
        )
    path = Path(schema_path)
    text = path.read_text(encoding="utf-8")
    initials = tuple(_read_yaml_list(text, "initials"))
    finals = tuple(_read_yaml_list(text, "finals"))
    tones = tuple(t for t in _read_yaml_list(text, "tone_marks") if t) or tuple(DEFAULT_TONES)
    if not tones or not set(tones).intersection(DEFAULT_TONES):
        tones = tuple(DEFAULT_TONES)
    return Inventory(
        initials=initials,
        finals=finals,
        tones=tones,
        legal_pairs=legal_pairs,
        legal_pair_records=records,
    )


def load_legal_pairs(path: str | Path | None) -> frozenset[tuple[str, str]] | None:
    if path is None:
        return None
    records = load_legal_pair_records(path)
    return frozenset((record.initial, record.final) for record in records if record.status != "invalid")


def load_legal_pair_records(path: str | Path | None) -> tuple[LegalPairRecord, ...]:
    if path is None:
        return ()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Legal-pairs file not found: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return ()
    first = lines[0].split("\t")
    has_header = "initial" in first and "final" in first
    records: list[LegalPairRecord] = []
    if has_header:
        for row in csv.DictReader(lines, delimiter="\t"):
            records.append(_record_from_mapping(row))
    else:
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            initial = _normalize_initial(parts[0])
            final = parts[1].strip()
            source = parts[2].strip() if len(parts) > 2 else ""
            source_lower = source.lower()
            records.append(
                LegalPairRecord(
                    initial=initial,
                    final=final,
                    phonologically_allowed=True,
                    attested_in_dictionary=bool(source),
                    human_reviewed="human" in source_lower or "review" in source_lower,
                    observed_in_training="training" in source_lower,
                    source=source,
                )
            )
    return tuple(records)


def _record_from_mapping(row: dict[str, str]) -> LegalPairRecord:
    source = str(row.get("source", "")).strip()
    return LegalPairRecord(
        initial=_normalize_initial(str(row.get("initial", ""))),
        final=str(row.get("final", "")).strip(),
        tone=str(row.get("tone", "")).strip() or None,
        phonologically_allowed=_as_bool(row.get("phonologically_allowed"), default=True),
        attested_in_dictionary=_as_bool(row.get("attested_in_dictionary")),
        human_reviewed=_as_bool(row.get("human_reviewed")),
        observed_in_training=_as_bool(row.get("observed_in_training")),
        source=source,
        notes=str(row.get("notes", "")).strip(),
    )


def _normalize_initial(value: str) -> str:
    text = value.strip()
    return "" if text in {"Ø", "∅", "-", ""} else text


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_yaml_list(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    active = False
    base_indent = 0
    for line in lines:
        stripped = line.strip()
        if not active:
            if stripped == f"{key}:":
                active = True
                base_indent = len(line) - len(line.lstrip())
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("- "):
            value = stripped[2:].strip()
            value = value.split("#", 1)[0].strip().strip('"').strip("'")
            out.append(value)
            continue
        if stripped and indent <= base_indent:
            break
    return out
