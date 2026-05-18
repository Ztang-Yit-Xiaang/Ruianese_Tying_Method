#!/usr/bin/env python3
import sys, csv, io, re
from pathlib import Path
from datetime import datetime

def load_tsv(tsv_path):
    items = []
    seen = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln or ln.strip().startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            han = parts[0].strip()
            code = parts[1].strip()
            weight = parts[2].strip() if len(parts) >=3 and parts[2].strip() else "1"
            # 基本清洗：只接受單個 CJK 字
            if not re.match(r'^[\u3400-\u9FFF]$', han):
                continue
            key = (han, code, weight)
            if key in seen:
                continue
            seen.add(key)
            items.append((han, code, weight))
    return sorted(items, key=lambda item: (item[0], item[1], item[2]))

def write_dict_yaml(items, out_path):
    header = f"""# Rime dictionary
# Project: 瑞安話輸入法製作
# Source: 瑞安方言讀音字典（請補充具體書目）
# Encoding: UTF-8
# Generated: {datetime.now().strftime('%Y-%m-%d')}
---
name: ruianese_characters
version: \"{datetime.now().strftime('%Y.%m.%d')}\"
sort: by_weight
use_preset_vocabulary: false
columns:
  - text
  - code
  - weight
...
"""
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        for han, code, weight in items:
            f.write(f"{han}\t{code}\t{weight}\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/build_ruianese_dict_from_tsv.py ruianese_characters_template.tsv ruianese.char.dict.yaml")
        sys.exit(1)
    tsv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    items = load_tsv(tsv_path)
    write_dict_yaml(items, out_path)
    print(f"Wrote {len(items)} entries to {out_path}")

if __name__ == "__main__":
    main()
