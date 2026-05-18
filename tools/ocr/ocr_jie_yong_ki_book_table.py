#!/usr/bin/env python3
"""Table-aware OCR for Jie Yong Ki / 張永愷 dictionary pages.

This keeps the raw whole-page OCR as a separate artifact and builds a more
reviewable cell-level extraction by detecting the printed table grid first.
It uses PIL/numpy for grid detection and the Tesseract CLI for OCR, so it does
not require cv2 or pytesseract.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT.parents[1] / "張永愷書之圖"
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "book_ocr_structured"
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
PAGE_RE = re.compile(r"页面_(\d+)\.png$")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
PINYIN_RE = re.compile(r"^[A-Za-z0-9āáǎàēéěèīíǐìōóǒòūúǔùüǖǘǚǜɿŋʔʰ'`´ˉˊˇˋ\- .,;/()]+$")
GRID_NOISE_RE = re.compile(r"^[|_\-—=+:.·,;。．、\s]+$")

COLUMN_NAMES = [
    "radical_strokes",
    "hanzi",
    "pinyin",
    "fanqie",
    "tone",
    "rhyme",
    "initial",
    "ipa",
    "homophone",
    "note",
]

CELL_FIELDS = ["page", "row", "column", "column_name", "text", "confidence", "bbox", "status", "note"]
ROW_FIELDS = [
    "page",
    "row",
    *COLUMN_NAMES,
    "avg_confidence",
    "status",
    "issues",
]


@dataclass(frozen=True)
class Box:
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

    def padded(self, width: int, height: int, pad: int) -> "Box":
        return Box(
            max(0, self.left + pad),
            max(0, self.top + pad),
            min(width, self.right - pad),
            min(height, self.bottom - pad),
        )

    def as_string(self) -> str:
        return f"{self.left},{self.top},{self.width},{self.height}"


def page_number(path: Path) -> int | None:
    match = PAGE_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_images(image_dir: Path, start_after: int) -> list[tuple[int, Path]]:
    images = []
    for path in image_dir.glob("*.png"):
        number = page_number(path)
        if number is not None and number > start_after:
            images.append((number, path))
    return sorted(images, key=lambda item: item[0])


def tesseract_command() -> str:
    if TESSERACT_EXE.exists():
        return str(TESSERACT_EXE)
    found = shutil.which("tesseract")
    if found:
        return found
    raise RuntimeError(
        "Tesseract executable was not found. Install Tesseract or update "
        f"TESSERACT_EXE in {Path(__file__).name}."
    )


def normalize_text(value: str) -> str:
    value = value.replace("\ufeff", "")
    value = value.replace("|", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def threshold_image(image: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(image)
    arr = np.asarray(gray)
    # Printed grid/table text is dark on a light page. The threshold is kept
    # conservative so mild scan shadows do not become fake lines.
    return arr < 175


def group_runs(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value - previous <= max_gap + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return runs


def run_centers(runs: list[tuple[int, int]]) -> list[int]:
    return [(start + end) // 2 for start, end in runs]


def detect_grid_lines(dark: np.ndarray, min_row_fraction: float, min_col_fraction: float) -> tuple[list[int], list[int]]:
    horizontal_density = dark.mean(axis=1)
    vertical_density = dark.mean(axis=0)
    h_indices = np.where(horizontal_density >= min_row_fraction)[0]
    v_indices = np.where(vertical_density >= min_col_fraction)[0]
    h_lines = run_centers(group_runs(h_indices, max_gap=2))
    v_lines = run_centers(group_runs(v_indices, max_gap=2))
    return h_lines, v_lines


def line_coverage(dark: np.ndarray, h_lines: list[int], v_lines: list[int]) -> tuple[list[int], list[int]]:
    height, width = dark.shape
    filtered_h = [line for line in h_lines if 0 <= line < height and dark[line, :].mean() >= 0.12]
    filtered_v = [line for line in v_lines if 0 <= line < width and dark[:, line].mean() >= 0.12]
    return filtered_h, filtered_v


def detect_cell_boxes(image: Image.Image, min_row_fraction: float, min_col_fraction: float) -> tuple[list[list[Box]], dict[str, int]]:
    dark = threshold_image(image)
    width, height = image.size
    h_lines, v_lines = detect_grid_lines(dark, min_row_fraction, min_col_fraction)
    h_lines, v_lines = line_coverage(dark, h_lines, v_lines)
    h_lines = sorted(set(line for line in h_lines if 10 < line < height - 10))
    v_lines = sorted(set(line for line in v_lines if 10 < line < width - 10))

    rows: list[list[Box]] = []
    if len(h_lines) < 3 or len(v_lines) < 3:
        return rows, {"horizontal_lines": len(h_lines), "vertical_lines": len(v_lines)}

    for row_index in range(len(h_lines) - 1):
        top = h_lines[row_index]
        bottom = h_lines[row_index + 1]
        if bottom - top < 10:
            continue
        row: list[Box] = []
        for col_index in range(len(v_lines) - 1):
            left = v_lines[col_index]
            right = v_lines[col_index + 1]
            if right - left < 8:
                continue
            row.append(Box(left, top, right, bottom))
        if row:
            rows.append(row)
    return rows, {"horizontal_lines": len(h_lines), "vertical_lines": len(v_lines)}


def parse_tsv_confidence(path: Path) -> float:
    values = []
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = float(row.get("conf", "-1"))
            except ValueError:
                continue
            if conf >= 0:
                values.append(conf)
    return round(sum(values) / len(values), 2) if values else 0.0


def is_blank_crop(image: Image.Image, max_dark_fraction: float) -> bool:
    return bool(threshold_image(image).mean() <= max_dark_fraction)


def run_tesseract_cell(image: Image.Image, lang: str, psm: int, tmp_dir: Path, stem: str, keep_cell_images: bool) -> tuple[str, float]:
    image_path = tmp_dir / f"{stem}.png"
    output_base = tmp_dir / stem
    image.save(image_path)
    command = [tesseract_command(), str(image_path), str(output_base), "-l", lang, "--psm", str(psm), "txt", "tsv"]
    subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    txt_path = output_base.with_suffix(".txt")
    text = normalize_text(txt_path.read_text(encoding="utf-8", errors="ignore")) if txt_path.exists() else ""
    confidence = parse_tsv_confidence(output_base.with_suffix(".tsv"))
    if not keep_cell_images:
        for path in [image_path, output_base.with_suffix(".txt"), output_base.with_suffix(".tsv")]:
            try:
                path.unlink()
            except (FileNotFoundError, PermissionError):
                pass
    return text, confidence


def psm_for_column(column_name: str, default_psm: int) -> int:
    if default_psm:
        return default_psm
    if column_name in {"hanzi", "tone", "rhyme", "initial"}:
        return 8
    return 7


def classify_cell(column_name: str, text: str, confidence: float, min_conf: float) -> tuple[str, str]:
    notes = []
    status = "ok"
    compact = text.strip()
    if not compact:
        return "review", "blank"
    if confidence and confidence < min_conf:
        notes.append("low_confidence")
        status = "review"
    if GRID_NOISE_RE.match(compact):
        notes.append("grid_noise")
        status = "suspicious"
    if column_name == "hanzi" and not CJK_RE.search(compact):
        notes.append("missing_hanzi")
        status = "review"
    if column_name == "pinyin" and CJK_RE.search(compact):
        notes.append("pinyin_contains_hanzi")
        status = "suspicious"
    elif column_name == "pinyin" and compact and not PINYIN_RE.match(compact):
        notes.append("pinyin_pattern")
        status = "review"
    if column_name in {"tone", "rhyme", "initial"} and len(compact) > 12:
        notes.append("possible_cross_column")
        status = "suspicious"
    return status, ",".join(notes)


def ocr_page(number: int, image_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    image = Image.open(image_path).convert("RGB")
    boxes_by_row, grid_stats = detect_cell_boxes(image, args.min_row_fraction, args.min_col_fraction)
    width, height = image.size
    cells: list[dict[str, str]] = []
    logical_rows: list[dict[str, str]] = []

    tmp_dir = args.out_dir / "_tmp_cells" / f"run_{number:03d}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for row_index, boxes in enumerate(boxes_by_row, 1):
        row_cells = []
        for col_index, box in enumerate(boxes, 1):
            column_name = COLUMN_NAMES[col_index - 1] if col_index <= len(COLUMN_NAMES) else f"extra_{col_index}"
            crop_box = box.padded(width, height, args.crop_pad)
            crop = image.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))
            if is_blank_crop(crop, args.blank_dark_fraction):
                text, confidence = "", 0.0
            else:
                text, confidence = run_tesseract_cell(
                    crop,
                    args.lang,
                    psm_for_column(column_name, args.psm),
                    tmp_dir,
                    f"page_{number:03d}_r{row_index:03d}_c{col_index:02d}",
                    args.keep_cell_images,
                )
            status, note = classify_cell(column_name, text, confidence, args.min_conf)
            cell = {
                "page": f"{number:03d}",
                "row": str(row_index),
                "column": str(col_index),
                "column_name": column_name,
                "text": text,
                "confidence": f"{confidence:.2f}",
                "bbox": box.as_string(),
                "status": status,
                "note": note,
            }
            cells.append(cell)
            row_cells.append(cell)
        logical = build_dictionary_row(number, row_index, row_cells, args.min_conf)
        if logical:
            logical_rows.append(logical)

    stat = {
        "page": f"{number:03d}",
        "image": str(image_path),
        "rows_detected": str(len(boxes_by_row)),
        "cells_detected": str(len(cells)),
        "dictionary_rows": str(len(logical_rows)),
        "horizontal_lines": str(grid_stats.get("horizontal_lines", 0)),
        "vertical_lines": str(grid_stats.get("vertical_lines", 0)),
        "review_cells": str(sum(1 for cell in cells if cell["status"] != "ok")),
        "review_rows": str(sum(1 for row in logical_rows if row["status"] != "ok")),
        "status": "review"
        if any(cell["status"] != "ok" for cell in cells) or any(row["status"] != "ok" for row in logical_rows)
        else "ok",
    }
    if not cells:
        stat["status"] = "failed"
    return cells, logical_rows, stat


def build_dictionary_row(number: int, row_index: int, row_cells: list[dict[str, str]], min_conf: float) -> dict[str, str] | None:
    if row_index == 1:
        return None
    values = {cell["column_name"]: cell["text"] for cell in row_cells}
    if not any(values.get(field, "").strip() for field in COLUMN_NAMES):
        return None
    hanzi = values.get("hanzi", "")

    issues = []
    if not CJK_RE.search(hanzi):
        issues.append("missing_hanzi")
    confidences = []
    for cell in row_cells:
        try:
            conf = float(cell["confidence"])
        except ValueError:
            conf = 0.0
        if conf:
            confidences.append(conf)
        if cell["status"] != "ok":
            issues.append(f"{cell['column_name']}:{cell['note'] or cell['status']}")

    pinyin = values.get("pinyin", "")
    if pinyin and CJK_RE.search(pinyin):
        issues.append("pinyin_contains_hanzi")
    elif pinyin and not PINYIN_RE.match(pinyin):
        issues.append("pinyin_pattern")

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    if avg_conf and avg_conf < min_conf:
        issues.append("low_row_confidence")

    row = {field: values.get(field, "") for field in COLUMN_NAMES}
    row.update(
        {
            "page": f"{number:03d}",
            "row": str(row_index),
            "avg_confidence": f"{avg_conf:.2f}",
            "status": "review" if issues else "ok",
            "issues": ",".join(dict.fromkeys(issue for issue in issues if issue)),
        }
    )
    return row


def write_cells(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CELL_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_dictionary_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, stats: list[dict[str, str]], cells: list[dict[str, str]], rows: list[dict[str, str]], image_dir: Path, start_after: int) -> None:
    review_pages = [stat for stat in stats if stat["status"] != "ok"]
    review_cells = [cell for cell in cells if cell["status"] != "ok"]
    review_rows = [row for row in rows if row["status"] != "ok"]
    suspicious_cells = [cell for cell in cells if cell["status"] == "suspicious"]
    low_conf_cells = [cell for cell in cells if "low_confidence" in cell["note"]]
    missing_hanzi_rows = [row for row in rows if "missing_hanzi" in row["issues"]]

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Jie Yong Ki / 張永愷 Structured Table OCR Validation\n\n")
        f.write(f"- Image directory: `{image_dir}`\n")
        f.write(f"- Start after page: {start_after:03d}\n")
        f.write(f"- Pages processed: {len(stats)}\n")
        f.write(f"- Pages needing review: {len(review_pages)}\n")
        f.write(f"- Cells extracted: {len(cells)}\n")
        f.write(f"- Dictionary rows extracted: {len(rows)}\n")
        f.write(f"- Review cells: {len(review_cells)}\n")
        f.write(f"- Suspicious cells: {len(suspicious_cells)}\n")
        f.write(f"- Low-confidence cells: {len(low_conf_cells)}\n")
        f.write(f"- Missing-hanzi rows: {len(missing_hanzi_rows)}\n\n")
        f.write("## Page Summary\n\n")
        f.write("| Page | Rows | Cells | Dict rows | Grid lines | Review cells | Status |\n")
        f.write("|---|---:|---:|---:|---|---:|---|\n")
        for stat in stats[:300]:
            grid = f"{stat['horizontal_lines']}h/{stat['vertical_lines']}v"
            f.write(
                f"| {stat['page']} | {stat['rows_detected']} | {stat['cells_detected']} | "
                f"{stat['dictionary_rows']} | {grid} | {stat['review_cells']} | {stat['status']} |\n"
            )
        if review_rows:
            f.write("\n## Review Row Samples\n\n")
            f.write("| Page | Row | Hanzi | Pinyin | Avg conf | Issues |\n")
            f.write("|---|---:|---|---|---:|---|\n")
            for row in review_rows[:80]:
                f.write(
                    f"| {row['page']} | {row['row']} | {row['hanzi']} | {row['pinyin']} | "
                    f"{row['avg_confidence']} | {row['issues']} |\n"
                )


def run(args: argparse.Namespace) -> None:
    images = collect_images(args.image_dir, args.start_after)
    if args.page:
        wanted = {int(page) for page in args.page}
        images = [(number, path) for number, path in images if number in wanted]
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No page images found after {args.start_after:03d} in {args.image_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_cells: list[dict[str, str]] = []
    all_rows: list[dict[str, str]] = []
    stats: list[dict[str, str]] = []

    print(f"Structured OCR pages: {len(images)} from {args.image_dir}")
    for index, (number, image_path) in enumerate(images, 1):
        per_page_path = args.out_dir / f"page_{number:03d}_cells.tsv"
        if args.skip_existing and per_page_path.exists():
            print(f"[{index}/{len(images)}] Skip existing structured page {number:03d}")
            with per_page_path.open("r", encoding="utf-8", newline="") as f:
                page_cells = list(csv.DictReader(f, delimiter="\t"))
            page_rows = []
            for row_id in sorted({cell["row"] for cell in page_cells}, key=lambda value: int(value)):
                row_cells = [cell for cell in page_cells if cell["row"] == row_id]
                logical = build_dictionary_row(number, int(row_id), row_cells, args.min_conf)
                if logical:
                    page_rows.append(logical)
            stat = {
                "page": f"{number:03d}",
                "image": str(image_path),
                "rows_detected": str(len({cell["row"] for cell in page_cells})),
                "cells_detected": str(len(page_cells)),
                "dictionary_rows": str(len(page_rows)),
                "horizontal_lines": "0",
                "vertical_lines": "0",
                "review_cells": str(sum(1 for cell in page_cells if cell["status"] != "ok")),
                "review_rows": str(sum(1 for row in page_rows if row["status"] != "ok")),
                "status": "review"
                if any(cell["status"] != "ok" for cell in page_cells) or any(row["status"] != "ok" for row in page_rows)
                else "ok",
            }
        else:
            print(f"[{index}/{len(images)}] Structured OCR page {number:03d}")
            page_cells, page_rows, stat = ocr_page(number, image_path, args)
            write_cells(per_page_path, page_cells)
        all_cells.extend(page_cells)
        all_rows.extend(page_rows)
        stats.append(stat)

    write_cells(args.out_dir / "all_cells.tsv", all_cells)
    write_dictionary_rows(args.out_dir / "dictionary_rows.tsv", all_rows)
    write_report(args.out_dir / "table_validation_report.md", stats, all_cells, all_rows, args.image_dir, args.start_after)
    print(f"Wrote structured OCR cells: {args.out_dir / 'all_cells.tsv'}")
    print(f"Wrote structured dictionary rows: {args.out_dir / 'dictionary_rows.tsv'}")
    print(f"Wrote structured validation: {args.out_dir / 'table_validation_report.md'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-after", type=int, default=58)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N selected pages; 0 means all.")
    parser.add_argument("--page", action="append", help="Process a specific page number, e.g. --page 059.")
    parser.add_argument("--lang", default="chi_sim+eng")
    parser.add_argument("--psm", type=int, default=0, help="Force one Tesseract PSM for every cell; 0 chooses by column.")
    parser.add_argument("--min-conf", type=float, default=45.0)
    parser.add_argument("--min-row-fraction", type=float, default=0.18)
    parser.add_argument("--min-col-fraction", type=float, default=0.24)
    parser.add_argument("--crop-pad", type=int, default=3)
    parser.add_argument("--blank-dark-fraction", type=float, default=0.012)
    parser.add_argument("--keep-cell-images", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args(argv)

    try:
        run(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
