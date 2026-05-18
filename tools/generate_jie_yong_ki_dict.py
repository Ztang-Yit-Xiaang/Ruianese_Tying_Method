#!/usr/bin/env python3
"""Generate the formal Jie Yong Ki / 張永愷 Rime character dictionary."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "output" / "jie_yong_ki" / "ruianese_characters_converted.tsv"
DEFAULT_SYLLABLES = ROOT / "ruian_pinyin.dict.yaml"
DEFAULT_OUTPUT = ROOT / "ruianese.jie_yong_ki.dict.yaml"
VALID_STATUSES = {"ok", "converted"}


def load_valid_syllables(path):
    syllables = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                syllables.add(line.split("\t", 1)[0].strip())
    return syllables


def load_filtered_entries(path, valid_syllables):
    entries = []
    rejected = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for line_no, row in enumerate(reader, 2):
            status = (row.get("status") or "").strip()
            char = (row.get("char") or "").strip()
            code = (row.get("code") or "").strip()
            weight = (row.get("weight") or "1").strip() or "1"
            if status not in VALID_STATUSES:
                rejected.append((line_no, char, code, status, "status"))
                continue
            if not re.fullmatch(r"[\u3400-\u9fff]", char):
                rejected.append((line_no, char, code, status, "character"))
                continue
            if code not in valid_syllables:
                rejected.append((line_no, char, code, status, "syllable"))
                continue
            key = (char, code, weight)
            if key in seen:
                continue
            seen.add(key)
            entries.append(key)
    return sorted(entries, key=lambda item: (item[0], item[1], item[2])), rejected


def write_rime_dict(entries, out_path):
    header = f"""# Rime dictionary
# Project: 瑞安話輸入法製作
# Source: Jie Yong Ki / 張永愷《瑞安方言讀音字典》converted review data
# Encoding: UTF-8
# Generated: {date.today().isoformat()}
---
name: ruianese_jie_yong_ki
version: "{date.today().strftime('%Y.%m.%d')}"
sort: by_weight
use_preset_vocabulary: false
columns:
  - text
  - code
  - weight
...
"""
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for char, code, weight in entries:
            f.write(f"{char}\t{code}\t{weight}\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--syllables", type=Path, default=DEFAULT_SYLLABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    valid_syllables = load_valid_syllables(args.syllables)
    entries, rejected = load_filtered_entries(args.source, valid_syllables)
    write_rime_dict(entries, args.output)
    unresolved = sum(1 for item in rejected if item[4] == "status")
    invalid = [item for item in rejected if item[4] != "status"]
    if invalid:
        details = "; ".join(f"line {line}: {char} {code} ({reason})" for line, char, code, _, reason in invalid[:10])
        raise SystemExit(f"Generated {args.output}, but found invalid validated rows: {details}")
    print(f"Wrote {len(entries)} validated entries to {args.output}")
    print(f"Excluded {unresolved} unresolved rows from {args.source}")


if __name__ == "__main__":
    main()
