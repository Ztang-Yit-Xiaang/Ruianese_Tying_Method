#!/usr/bin/env python3
"""Convert the existing Jie Yong Ki / 張永愷 readings to the new pinyin policy."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.lib.jie_yong_ki_transcription import convert_code
DEFAULT_SOURCE = ROOT / "ruianese.character_01.dict.yaml"
DEFAULT_SYLLABLES = ROOT / "ruian_pinyin.dict.yaml"
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki"


def load_valid_syllables(path):
    syllables = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            syllables.add(line.split("\t", 1)[0].strip())
    return syllables


def parse_entries(path):
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            text = raw.rstrip("\n")
            content, _, comment = text.partition("#")
            parts = content.split()
            if len(parts) < 2 or not re.fullmatch(r"[\u3400-\u9fff]", parts[0]):
                continue
            entries.append(
                {
                    "line": line_no,
                    "char": parts[0],
                    "old_code": parts[1],
                    "comment": comment.strip(),
                }
            )
    return entries


def write_tsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, entries, converted_rows, unresolved_rows):
    char_counts = Counter(entry["char"] for entry in entries)
    polyphones = {char: count for char, count in char_counts.items() if count > 1}
    status_counts = Counter(row["status"] for row in converted_rows)
    code_counts = defaultdict(set)
    for row in converted_rows:
        code_counts[row["char"]].add(row["code"])
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Jie Yong Ki / 張永愷 Character Conversion Report\n\n")
        f.write(f"- Source entries: {len(entries)}\n")
        f.write(f"- Converted/review rows: {len(converted_rows)}\n")
        f.write(f"- OK unchanged: {status_counts.get('ok', 0)}\n")
        f.write(f"- Auto-converted: {status_counts.get('converted', 0)}\n")
        f.write(f"- Unresolved: {len(unresolved_rows)}\n")
        f.write(f"- Unique characters: {len(char_counts)}\n")
        f.write(f"- Characters with multiple readings: {len(polyphones)}\n\n")
        f.write("## Unresolved Samples\n\n")
        f.write("| Line | Character | Old code | Note |\n")
        f.write("|---:|---|---|---|\n")
        for row in unresolved_rows[:50]:
            f.write(f"| {row['line']} | {row['char']} | {row['old_code']} | {row['note']} |\n")
        f.write("\n## Multiple Reading Samples\n\n")
        f.write("| Character | Readings |\n")
        f.write("|---|---|\n")
        for char in sorted(polyphones)[:50]:
            readings = ", ".join(sorted(code_counts[char]))
            f.write(f"| {char} | {readings} |\n")


def convert_entries(entries, valid_syllables):
    converted_rows = []
    unresolved_rows = []
    for entry in entries:
        result = convert_code(entry["old_code"], valid_syllables)
        row = {
            "char": entry["char"],
            "code": result.code,
            "weight": "1",
            "source": f"ruianese.character_01.dict.yaml:{entry['line']}",
            "note": result.note if not entry["comment"] else f"{result.note}; {entry['comment']}",
            "old_code": entry["old_code"],
            "status": result.status,
            "line": str(entry["line"]),
        }
        converted_rows.append(row)
        if result.status == "unresolved":
            unresolved_rows.append(row)
    return converted_rows, unresolved_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--syllables", type=Path, default=DEFAULT_SYLLABLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    valid_syllables = load_valid_syllables(args.syllables)
    entries = parse_entries(args.source)
    converted_rows, unresolved_rows = convert_entries(entries, valid_syllables)

    converted_path = args.out_dir / "ruianese_characters_converted.tsv"
    unresolved_path = args.out_dir / "unresolved_codes.tsv"
    report_path = args.out_dir / "character_conversion_report.md"
    review_fields = ["char", "code", "weight", "source", "note", "old_code", "status"]
    unresolved_fields = ["line", "char", "old_code", "code", "source", "note", "status"]
    write_tsv(converted_path, converted_rows, review_fields)
    write_tsv(unresolved_path, unresolved_rows, unresolved_fields)
    write_report(report_path, entries, converted_rows, unresolved_rows)
    print(f"Wrote {len(converted_rows)} rows to {converted_path}")
    print(f"Wrote {len(unresolved_rows)} unresolved rows to {unresolved_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
