#!/usr/bin/env python3
"""Build clean Jie Yong Ki rows from VLM table output plus IPA glyph crops.

The Mandarin pinyin column is kept only as dictionary lookup metadata. Rui'an
codes and tone numbers are derived from the Rui'an IPA cell, with the printed
half-circle tone glyph handled separately from the IPA body.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

import compare_vlm_backends
import ocr_jie_yong_ki_book_table as table_ocr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "vlm_backend_trials"
TONE_CATEGORY_BY_POSITION = {
    "left_lower": "平",
    "left_upper": "上",
    "right_upper": "去",
    "right_lower": "入",
}
TONE_NUMBER_BY_GLYPH = {
    ("left_lower", "yin"): "1",
    ("left_upper", "yin"): "2",
    ("right_upper", "yin"): "3",
    ("right_lower", "yin"): "4",
    ("left_lower", "yang"): "5",
    ("left_upper", "yang"): "6",
    ("right_upper", "yang"): "7",
    ("right_lower", "yang"): "8",
}
CLEAN_COLUMNS = [
    "page",
    "row",
    "hanzi_raw",
    "hanzi",
    "hanzi_source",
    "mandarin_pinyin",
    "fanqie",
    "category_tone",
    "category_rhyme",
    "category_initial",
    "ruian_ipa_raw_vlm",
    "ruian_ipa_corrected",
    "ipa_correction_source",
    "ruian_ipa_body",
    "ruian_initial",
    "ruian_final",
    "tone_glyph_position",
    "tone_glyph_register",
    "tone_glyph_source",
    "tone_glyph_confidence",
    "tone_number",
    "ruian_code",
    "homophone",
    "note",
    "status",
    "review_note",
]
LATEX_SYMBOLS = {
    "alpha": "a",
    "epsilon": "eh",
    "varepsilon": "eh",
    "eta": "ng",
}
IPA_TO_CODE_REPLACEMENTS = [
    ("ɕ", "x"),
    ("ȵ", "nj"),
    ("ŋ", "ng"),
    ("η", "ng"),
    ("ɛ", "eh"),
    ("æ", "ae"),
    ("ɔ", "oe"),
    ("ɿ", "i"),
    ("ʉ", "yu"),
    ("y̟u", "yu"),
    ("ʰ", "h"),
]
PROTECTED_INITIAL_REPLACEMENTS = [
    ("tɕʰ", "@Q@"),
    ("dʑ", "@JJ@"),
    ("tɕ", "@J@"),
    ("tsh", "@C@"),
    ("tsʰ", "@C@"),
    ("dz", "@ZZ@"),
    ("ts", "@Z@"),
    ("z̠", "@ZS@"),
    ("ʑ", "@ZS@"),
]
PROTECTED_INITIAL_RESTORE = {
    "@Q@": "q",
    "@JJ@": "jj",
    "@J@": "j",
    "@C@": "c",
    "@ZZ@": "zz",
    "@Z@": "z",
    "@ZS@": "zs",
}


@dataclass(frozen=True)
class Component:
    area: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def is_horizontal(self) -> bool:
        return self.width >= 7 and self.height <= 7 and self.width >= self.height * 2

    def as_box(self) -> str:
        return f"{self.left},{self.top},{self.width},{self.height}"


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_text(value: Any) -> str:
    return compare_vlm_backends.flatten_text(value)


def load_vlm_text(page: int, out_dir: Path) -> str:
    raw_json = out_dir / f"page_{page:03d}_paddleocr_vl_raw.json"
    raw_md = out_dir / f"page_{page:03d}_paddleocr_vl_raw.md"
    if raw_json.exists():
        return flatten_text(json.loads(raw_json.read_text(encoding="utf-8")))
    if raw_md.exists():
        return raw_md.read_text(encoding="utf-8")
    raise SystemExit(f"Missing PaddleOCR-VL raw output for page {page:03d}: {raw_json}")


def strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return normalize_cell(html.unescape(value))


def extract_html_table_rows(page: int, text: str) -> list[dict[str, str]]:
    match = re.search(r"<table.*?</table>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise SystemExit("No HTML table found in PaddleOCR-VL raw output.")
    rows: list[dict[str, str]] = []
    for tr in re.findall(r"<tr.*?</tr>", match.group(0), flags=re.IGNORECASE | re.DOTALL):
        cells = [strip_tags(cell) for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.IGNORECASE | re.DOTALL)]
        if len(cells) < 10:
            continue
        if {"汉字", "拼音", "反切"} & set(cells):
            continue
        rows.append(
            {
                "page": f"{page:03d}",
                "row": str(len(rows) + 1),
                "hanzi_raw": cells[1],
                "hanzi": cells[1],
                "hanzi_source": "explicit" if table_ocr.CJK_RE.search(cells[1]) else "",
                "mandarin_pinyin": cells[2],
                "fanqie": cells[3],
                "category_tone": cells[4],
                "category_rhyme": cells[5],
                "category_initial": cells[6],
                "ruian_ipa_raw_vlm": cells[7],
                "ruian_ipa_corrected": "",
                "ipa_correction_source": "vlm_exact",
                "homophone": cells[8],
                "note": cells[9],
            }
        )
    return rows


def load_overrides(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            (row.get("page", ""), row.get("row", "")): row
            for row in csv.DictReader(f, delimiter="\t")
            if row.get("page") and row.get("row")
        }


def find_image(page: int, image_dir: Path, start_after: int) -> Path:
    for number, path in table_ocr.collect_images(image_dir, start_after):
        if number == page:
            return path
    raise SystemExit(f"Page {page:03d} was not found in {image_dir}")


def page_cell_boxes(image_path: Path, args: argparse.Namespace) -> list[list[table_ocr.Box]]:
    image = Image.open(image_path).convert("RGB")
    boxes, _ = table_ocr.detect_cell_boxes(image, args.min_row_fraction, args.min_col_fraction)
    if len(boxes) < 2:
        raise SystemExit(f"Could not detect table cells in {image_path}")
    return boxes


def connected_components(crop: Image.Image) -> list[Component]:
    gray = ImageOps.grayscale(crop)
    arr = np.asarray(gray)
    dark = arr < 150
    height, width = dark.shape
    seen = np.zeros(dark.shape, dtype=bool)
    components: list[Component] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not dark[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                px, py = stack.pop()
                xs.append(px)
                ys.append(py)
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if 0 <= nx < width and 0 <= ny < height and dark[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(xs) >= 4:
                components.append(Component(len(xs), min(xs), min(ys), max(xs) + 1, max(ys) + 1))
    return [
        comp
        for comp in components
        if comp.width >= 2
        and comp.height >= 2
        and not (comp.height > crop.height * 0.8 and comp.width <= 3)
        and not (comp.width > crop.width * 0.8 and comp.height <= 3)
    ]


def detect_tone_glyph(crop: Image.Image) -> dict[str, str]:
    comps = connected_components(crop)
    if not comps:
        return {
            "tone_glyph_position": "unknown",
            "tone_glyph_register": "unknown",
            "tone_glyph_source": "unknown",
            "tone_glyph_confidence": "0.00",
            "tone_glyph_note": "tone_glyph_unclear:no_components",
        }

    body_candidates = [c for c in comps if c.area >= 250 or (c.width >= 20 and c.height >= 20)]
    if not body_candidates:
        body_candidates = sorted(comps, key=lambda c: c.area, reverse=True)[:2]
    body_left = min(c.left for c in body_candidates)
    body_right = max(c.right for c in body_candidates)
    body_top = min(c.top for c in body_candidates)
    body_bottom = max(c.bottom for c in body_candidates)
    body_mid_y = (body_top + body_bottom) / 2

    glyph_candidates = [
        c
        for c in comps
        if 15 <= c.area <= 260
        and 4 <= c.width <= 28
        and 4 <= c.height <= 38
        and (c.right <= body_left + 4 or c.left >= body_right - 4)
    ]
    nonbar_candidates = [c for c in glyph_candidates if not c.is_horizontal]
    if not nonbar_candidates:
        return {
            "tone_glyph_position": "unknown",
            "tone_glyph_register": "unknown",
            "tone_glyph_source": "unknown",
            "tone_glyph_confidence": "0.00",
            "tone_glyph_note": "tone_glyph_unclear:no_side_glyph",
        }

    glyph = max(nonbar_candidates, key=lambda c: (min(abs(c.center_x - body_left), abs(c.center_x - body_right)), c.area))
    side = "left" if glyph.center_x < (body_left + body_right) / 2 else "right"
    vertical = "upper" if glyph.center_y < body_mid_y else "lower"
    position = f"{side}_{vertical}"

    bars = [
        c
        for c in glyph_candidates
        if c.is_horizontal
        and c.center_y > glyph.center_y + 4
        and not (c.right < glyph.left - 8 or c.left > glyph.right + 8)
    ]
    register = "yang" if bars else "yin"
    side_gap = min(abs(glyph.center_x - body_left), abs(glyph.center_x - body_right))
    confidence = 0.85 if side_gap <= 30 else 0.65
    if bars:
        confidence = min(0.95, confidence + 0.05)
    return {
        "tone_glyph_position": position if position in TONE_CATEGORY_BY_POSITION else "unknown",
        "tone_glyph_register": register,
        "tone_glyph_source": "image_glyph",
        "tone_glyph_confidence": f"{confidence:.2f}",
        "tone_glyph_note": f"glyph={glyph.as_box()};bar={';'.join(bar.as_box() for bar in bars)}",
    }


def raw_marker_fallback(raw: str) -> dict[str, str] | None:
    compact = normalize_cell(raw)
    match = re.search(r"(?:_|\^)\s*([123])\b|([₁₂₃¹²³])", compact)
    if not match:
        return None
    marker = match.group(1) or match.group(2)
    normalized = {"₁": "1", "¹": "1", "₂": "2", "²": "2", "₃": "3", "³": "3"}.get(marker, marker)
    mapping = {
        "1": ("left_lower", "yin", "1"),
        "2": ("right_lower", "yang", "8"),
        "3": ("right_lower", "yin", "4"),
    }
    if normalized not in mapping:
        return None
    position, register, tone_number = mapping[normalized]
    return {
        "tone_glyph_position": position,
        "tone_glyph_register": register,
        "tone_glyph_source": "raw_marker_fallback",
        "tone_glyph_confidence": "0.40",
        "tone_glyph_note": f"tone_glyph_from_raw_marker:{marker}",
        "tone_number": tone_number,
    }


def clean_ipa_body(raw: str) -> str:
    value = normalize_cell(raw)
    for name, replacement in LATEX_SYMBOLS.items():
        value = re.sub(rf"\\{name}\b", replacement, value)
    value = re.sub(r"[_^]\s*[0-9０-９]+", "", value)
    value = value.translate(str.maketrans({"₂": "", "²": "", "₁": "", "¹": "", "₃": "", "³": ""}))
    value = re.sub(r"(^|\s)[cC]\s+", " ", value)
    value = re.sub(r"[()）]", "", value)
    value = re.sub(r"\s+", "", value)
    return value.strip()


def split_initial_final(code_body: str, valid_syllables: set[str]) -> tuple[str, str]:
    # Keep this explicit; the formal legal-pair table uses these initial spellings.
    for initial in ["jj", "nj", "ng", "bb", "dd", "gg", "zs", "zz", "ss", "j", "q", "x", "b", "p", "m", "f", "v", "d", "t", "n", "l", "g", "k", "h", "z", "c", "s"]:
        if code_body.startswith(initial):
            return initial, code_body[len(initial):]
    return "", code_body


def ipa_body_to_code_body(body: str) -> tuple[str, str]:
    if r"\theta" in body or "θ" in body:
        return "", "ipa_artifact_theta"
    code = body.lower()
    notes: list[str] = []
    for old, protected in PROTECTED_INITIAL_REPLACEMENTS:
        if old in code:
            code = code.replace(old, protected)
    if "z" in code:
        code = code.replace("z", "ss")
        notes.append("z->ss")
    for protected, restored in PROTECTED_INITIAL_RESTORE.items():
        if protected in code:
            code = code.replace(protected, restored)
    for old, new in IPA_TO_CODE_REPLACEMENTS:
        if old in code:
            code = code.replace(old, new)
    code = code.replace("ɑ", "a").replace("ɒ", "o")
    code = code.replace("ə", "e")
    code = code.replace("aa", "a")
    code = re.sub(r"[^a-z]", "", code)
    if code.endswith("w"):
        code = code[:-1] + "u"
        notes.append("w->u")
    if code.endswith("ung"):
        code = code[:-3] + "ong"
        notes.append("ung->ong")
    if code.endswith("uang"):
        code = code[:-4] + "ong"
        notes.append("uang->ong")
    if code.endswith("an") and not code.endswith(("ian", "uan")):
        code = code[:-2] + "ang"
        notes.append("an->ang")
    if code.startswith("kh"):
        code = "k" + code[2:]
        notes.append("kh->k")
    if code.startswith("th"):
        code = "t" + code[2:]
        notes.append("th->t")
    if code.startswith("ph"):
        code = "p" + code[2:]
        notes.append("ph->p")
    return code, ",".join(notes)


def load_valid_syllables(path: Path) -> set[str]:
    valid = set()
    in_body = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "...":
                in_body = True
                continue
            if not in_body or not line or line.startswith("#"):
                continue
            valid.add(line.split("\t", 1)[0])
    return valid


def build_clean_rows(page: int, image_path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    image = Image.open(image_path).convert("RGB")
    boxes_by_row = page_cell_boxes(image_path, args)
    valid_syllables = load_valid_syllables(args.ruian_pinyin_dict)
    overrides = load_overrides(args.ipa_overrides)
    clean_rows: list[dict[str, str]] = []
    glyph_rows: list[dict[str, str]] = []
    inherited_hanzi = ""

    for index, source in enumerate(rows, 1):
        notes: list[str] = []
        override = overrides.get((f"{page:03d}", str(index)), {})
        hanzi_raw = source.get("hanzi_raw", source.get("hanzi", ""))
        if table_ocr.CJK_RE.search(hanzi_raw):
            hanzi = hanzi_raw
            hanzi_source = "explicit"
            inherited_hanzi = hanzi
        elif inherited_hanzi:
            hanzi = inherited_hanzi
            hanzi_source = "inherited"
        else:
            hanzi = hanzi_raw
            hanzi_source = "missing"
            notes.append("missing_hanzi")

        grid_row_index = index
        glyph_info = {
            "tone_glyph_position": "unknown",
            "tone_glyph_register": "unknown",
            "tone_glyph_source": "unknown",
            "tone_glyph_confidence": "0.00",
            "tone_glyph_note": "tone_glyph_unclear:no_crop",
        }
        if grid_row_index < len(boxes_by_row) and len(boxes_by_row[grid_row_index]) >= 8:
            box = boxes_by_row[grid_row_index][7].padded(image.width, image.height, args.crop_pad)
            crop = image.crop((box.left, box.top, box.right, box.bottom))
            glyph_info = detect_tone_glyph(crop)
        else:
            notes.append("missing_ipa_cell_crop")

        raw_vlm = source.get("ruian_ipa_raw_vlm", source.get("ruian_ipa_raw", ""))
        if glyph_info["tone_glyph_position"] == "unknown":
            fallback = raw_marker_fallback(raw_vlm)
            if fallback:
                glyph_info.update({key: value for key, value in fallback.items() if key != "tone_number"})
                notes.append("tone_glyph_from_raw_marker")

        ipa_corrected = normalize_cell(override.get("ruian_ipa_corrected")) or raw_vlm
        correction_source = "manual_override" if override else ("review" if (r"\theta" in raw_vlm or "θ" in raw_vlm) else "vlm_exact")
        ipa_body = clean_ipa_body(ipa_corrected)
        code_body, code_note = ipa_body_to_code_body(ipa_body)
        if code_note == "ipa_artifact_theta":
            notes.append(code_note)
        tone_number = normalize_cell(override.get("tone_number")) or TONE_NUMBER_BY_GLYPH.get((glyph_info["tone_glyph_position"], glyph_info["tone_glyph_register"]), "")
        if not tone_number:
            fallback = raw_marker_fallback(raw_vlm)
            tone_number = fallback.get("tone_number", "") if fallback else ""
        ruian_code = f"{code_body}{tone_number}" if code_body and tone_number else ""
        if override.get("ruian_code"):
            ruian_code = normalize_cell(override.get("ruian_code"))
            code_body = re.sub(r"[1-8]$", "", ruian_code)
            tone_number = normalize_cell(override.get("tone_number")) or (ruian_code[-1] if ruian_code[-1:].isdigit() else tone_number)
        if ruian_code and ruian_code not in valid_syllables:
            notes.append("code_not_in_scheme")
        if not ruian_code:
            notes.append("ruian_code_unresolved")
        expected_category = TONE_CATEGORY_BY_POSITION.get(glyph_info["tone_glyph_position"], "")
        if expected_category and source.get("category_tone") and expected_category != source.get("category_tone"):
            notes.append("tone_category_conflict")
        if glyph_info["tone_glyph_position"] == "unknown":
            notes.append("tone_glyph_unclear")
        ruian_initial = normalize_cell(override.get("ruian_initial"))
        ruian_final = normalize_cell(override.get("ruian_final"))
        if not (ruian_initial or ruian_final) and code_body:
            ruian_initial, ruian_final = split_initial_final(code_body, valid_syllables)

        row = {
            **source,
            "hanzi_raw": hanzi_raw,
            "hanzi": hanzi,
            "hanzi_source": hanzi_source,
            "ruian_ipa_raw_vlm": raw_vlm,
            "ruian_ipa_corrected": ipa_corrected,
            "ipa_correction_source": correction_source,
            "ruian_ipa_body": ipa_body,
            "ruian_initial": ruian_initial,
            "ruian_final": ruian_final,
            "tone_glyph_position": glyph_info["tone_glyph_position"],
            "tone_glyph_register": glyph_info["tone_glyph_register"],
            "tone_glyph_source": glyph_info["tone_glyph_source"],
            "tone_glyph_confidence": glyph_info["tone_glyph_confidence"],
            "tone_number": tone_number,
            "ruian_code": ruian_code if ruian_code in valid_syllables else "",
            "status": "review" if notes else "ok",
            "review_note": ",".join(dict.fromkeys(note for note in notes if note)),
        }
        clean_rows.append(row)
        glyph_rows.append(
            {
                "page": f"{page:03d}",
                "row": str(index),
                "hanzi": hanzi,
                "hanzi_source": hanzi_source,
                "ruian_ipa_raw_vlm": raw_vlm,
                "ruian_ipa_corrected": ipa_corrected,
                "ruian_ipa_body": ipa_body,
                **glyph_info,
                "category_tone": source.get("category_tone", ""),
                "tone_number": tone_number,
                "review_note": row["review_note"],
            }
        )
    return clean_rows, glyph_rows


def write_report(path: Path, clean_rows: list[dict[str, str]], glyph_rows: list[dict[str, str]], image_path: Path) -> None:
    resolved = [row for row in clean_rows if row["tone_number"]]
    conflicts = [row for row in clean_rows if "tone_category_conflict" in row["review_note"]]
    lines = [
        "# Zhang Yongkai Tone Glyph Validation",
        "",
        f"- Image: `{image_path}`",
        f"- Rows: {len(clean_rows)}",
        f"- Tone glyph resolved: {len(resolved)}",
        f"- Tone/category conflicts: {len(conflicts)}",
        f"- Review rows: {sum(1 for row in clean_rows if row['status'] != 'ok')}",
        "",
        "Tone numbers are derived from the printed half-circle glyph, not from Mandarin pinyin or OCR digits.",
        "",
        "## Row Summary",
        "",
        "| Row | Hanzi | Source | IPA raw | IPA corrected | IPA body | Glyph source | Glyph | Register | Tone | Code | Status | Notes |",
        "|---:|---|---|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for row in clean_rows:
        lines.append(
            f"| {row['row']} | {row['hanzi']} | {row['hanzi_source']} | {row['ruian_ipa_raw_vlm']} | "
            f"{row['ruian_ipa_corrected']} | {row['ruian_ipa_body']} | {row['tone_glyph_source']} | "
            f"{row['tone_glyph_position']} | {row['tone_glyph_register']} | {row['tone_number']} | "
            f"{row['ruian_code']} | {row['status']} | {row['review_note']} |"
        )
    write_text(path, "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="059")
    parser.add_argument("--image-dir", type=Path, default=table_ocr.DEFAULT_IMAGE_DIR)
    parser.add_argument("--start-after", type=int, default=58)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ruian-pinyin-dict", type=Path, default=ROOT / "ruian_pinyin.dict.yaml")
    parser.add_argument("--ipa-overrides", type=Path, default=DEFAULT_OUT_DIR / "ipa_overrides.tsv")
    parser.add_argument("--min-row-fraction", type=float, default=0.18)
    parser.add_argument("--min-col-fraction", type=float, default=0.24)
    parser.add_argument("--crop-pad", type=int, default=5)
    args = parser.parse_args(argv)

    page = int(args.page)
    image_path = find_image(page, args.image_dir, args.start_after)
    raw_text = load_vlm_text(page, args.out_dir)
    source_rows = extract_html_table_rows(page, raw_text)
    clean_rows, glyph_rows = build_clean_rows(page, image_path, source_rows, args)

    clean_path = args.out_dir / f"page_{page:03d}_clean_rows.tsv"
    glyph_path = args.out_dir / f"page_{page:03d}_tone_glyphs.tsv"
    report_path = args.out_dir / "tone_glyph_validation_report.md"
    write_tsv(clean_path, clean_rows, CLEAN_COLUMNS)
    write_tsv(
        glyph_path,
        glyph_rows,
        [
            "page",
            "row",
            "hanzi",
            "hanzi_source",
            "ruian_ipa_raw_vlm",
            "ruian_ipa_corrected",
            "ruian_ipa_body",
            "tone_glyph_position",
            "tone_glyph_register",
            "tone_glyph_source",
            "tone_glyph_confidence",
            "tone_glyph_note",
            "category_tone",
            "tone_number",
            "review_note",
        ],
    )
    write_report(report_path, clean_rows, glyph_rows, image_path)
    print(f"Wrote clean rows: {clean_path}")
    print(f"Wrote tone glyph rows: {glyph_path}")
    print(f"Wrote tone glyph report: {report_path}")


if __name__ == "__main__":
    main()
