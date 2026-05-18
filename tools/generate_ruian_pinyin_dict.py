#!/usr/bin/env python3
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
PAIR_PATH = ROOT / "ruian_legal_pairs.tsv"
DICT_PATH = ROOT / "ruian_pinyin.dict.yaml"
ZERO_INITIAL = "Ø"
TONES = tuple(str(i) for i in range(1, 9))


def load_pairs(path):
    pairs = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no}: expected initial<TAB>final")
            initial = "" if parts[0].strip() == ZERO_INITIAL else parts[0].strip()
            final = parts[1].strip()
            key = (initial, final)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def syllables_from_pairs(pairs):
    for initial, final in pairs:
        base = f"{initial}{final}"
        yield base
        for tone in TONES:
            yield f"{base}{tone}"


def write_dict(syllables, path):
    header = f"""# Rime dictionary
# Rui'an dialect romanization syllables
# Generated from ruian_legal_pairs.tsv.
# Official vowel policy: IPA o -> o, IPA ɔ -> oe, IPA uɔ -> uoe, IPA yɔ -> yoe.
# Tones: 1-8 suffixes; checked-tone codas -p/-t/-k are not required in ordinary input.
# Generated: {date.today().isoformat()}
---
name: ruian_pinyin
version: "{date.today().isoformat()}"
sort: by_weight
use_preset_vocabulary: false
columns:
  - text
  - weight
...
"""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for syllable in syllables:
            f.write(f"{syllable}\t1\n")


def main():
    pairs = load_pairs(PAIR_PATH)
    syllables = list(syllables_from_pairs(pairs))
    write_dict(syllables, DICT_PATH)
    print(f"Wrote {len(syllables)} syllables from {len(pairs)} legal pairs to {DICT_PATH}")


if __name__ == "__main__":
    main()
