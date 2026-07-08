from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from .inventory import load_inventory


@dataclass(frozen=True)
class SyllableInventory:
    initials: set[str]
    finals: set[str]
    syllables: set[str]
    tones: set[str]

    @classmethod
    def load_from_yaml(
        cls,
        schema_path: str | Path,
        legal_pairs_path: str | Path | None = None,
    ) -> "SyllableInventory":
        inventory = load_inventory(schema_path, legal_pairs_path)
        syllables: set[str] = set()
        pairs = inventory.legal_pairs or frozenset((initial, final) for initial in inventory.initials for final in inventory.finals)
        for initial, final in pairs:
            for tone in inventory.tones:
                syllables.add(f"{initial}{final}{tone}")
        return cls(
            initials=set(inventory.initials),
            finals=set(inventory.finals),
            syllables=syllables,
            tones=set(inventory.tones),
        )

    def is_valid_syllable(self, value: str) -> bool:
        return value.strip().lower().replace(" ", "").replace("'", "") in self.syllables

    def nearest_valid_candidates(self, value: str, limit: int = 5) -> list[str]:
        query = value.strip().lower().replace(" ", "").replace("'", "")
        return get_close_matches(query, sorted(self.syllables), n=limit, cutoff=0.55)
