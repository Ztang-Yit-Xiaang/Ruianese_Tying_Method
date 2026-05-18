#!/usr/bin/env python3
"""Extract Jie Yong Ki / 張永愷 initial-final table images into review TSV files.

The OCR path needs OpenCV and pytesseract. When those optional dependencies are
missing, the script fails with installation guidance instead of producing noisy
CSV. Use --from-official to seed the review files from the curated
ruian_legal_pairs.tsv while OCR dependencies are being set up.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = [Path("p437.png"), Path("p438.png"), Path("p439.png")]
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki"
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
ZERO_INITIAL = "Ø"


INITIAL_MAP = {
    "": "",
    ZERO_INITIAL: "",
    "p": "b",
    "ph": "p",
    "b": "bb",
    "m": "m",
    "f": "f",
    "v": "v",
    "t": "d",
    "th": "t",
    "d": "dd",
    "n": "n",
    "l": "l",
    "k": "g",
    "kh": "k",
    "g": "gg",
    "ng": "ng",
    "h": "h",
    "hh": "hh",
    "j": "j",
    "q": "q",
    "jj": "jj",
    "nj": "nj",
    "x": "x",
    "z": "z",
    "c": "c",
    "zz": "zz",
    "s": "s",
    "zs": "zs",
}

FINAL_MAP = {
    "a": "a",
    "o": "o",
    "oe": "oe",
    "ae": "ae",
    "e": "e",
    "i": "i",
    "u": "u",
    "yu": "yu",
    "ao": "ao",
    "ai": "ai",
    "ou": "ou",
    "ei": "ei",
    "ia": "ia",
    "iao": "iao",
    "iou": "iou",
    "ie": "ie",
    "iae": "iae",
    "io": "io",
    "uai": "uai",
    "uo": "uo",
    "uoe": "uoe",
    "yo": "yo",
    "yue": "yue",
    "yoe": "yoe",
    "ang": "ang",
    "eng": "eng",
    "ong": "ong",
    "iang": "iang",
    "iong": "iong",
    "ng": "ng",
}


def require_ocr_dependencies():
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        missing.append("pytesseract")
    if missing:
        names = ", ".join(missing)
        install = "python -m pip install " + " ".join(missing)
        raise RuntimeError(
            f"Missing OCR Python dependencies: {names}\n"
            f"Install them with: {install}\n"
            "Tesseract itself is also required; on this machine the expected path is "
            f"{TESSERACT_EXE}"
        )
    if not TESSERACT_EXE.exists() and shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract executable was not found. Install Tesseract or update "
            f"TESSERACT_EXE in {Path(__file__).name}."
        )


def normalize_token(value):
    return " ".join((value or "").strip().split())


def map_initial(raw):
    key = normalize_token(raw).replace("ʰ", "h")
    return INITIAL_MAP.get(key)


def map_final(raw):
    key = normalize_token(raw)
    return FINAL_MAP.get(key)


def read_official_pairs(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            initial, final, *rest = line.split("\t")
            source = rest[0] if rest else "ruian_legal_pairs.tsv"
            rows.append(
                {
                    "initial_raw": initial,
                    "final_raw": final,
                    "initial_code": "" if initial == ZERO_INITIAL else initial,
                    "final_code": final,
                    "source_page": source,
                    "status": "official_seed",
                }
            )
    return rows


def load_and_upright(path, cv2, pytesseract):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    try:
        osd = pytesseract.image_to_osd(img)
        rot = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                rot = int(line.split(":")[1].strip())
        if rot:
            rotates = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            img = cv2.rotate(img, rotates[(360 - rot) % 360])
    except Exception:
        pass
    return img


def binarize(img, cv2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    return 255 - bw


def detect_cells(bw, cv2):
    horiz = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    )
    vert = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    )
    grid = cv2.add(horiz, vert)
    contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= 3000 and w >= 15 and h >= 15:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: (box[1], box[0]))
    rows = []
    current = []
    for box in boxes:
        if not current or abs(box[1] - current[-1][1]) < 20:
            current.append(box)
        else:
            rows.append(sorted(current, key=lambda item: item[0]))
            current = [box]
    if current:
        rows.append(sorted(current, key=lambda item: item[0]))
    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    return [row for row in rows if len(row) >= max_cols * 0.7]


def ocr_cell(img, pytesseract):
    text = pytesseract.image_to_string(img, config="--psm 7 -l chi_sim+eng")
    return normalize_token(text)


def extract_table(path):
    require_ocr_dependencies()
    import cv2
    import pytesseract

    if TESSERACT_EXE.exists():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)

    img = load_and_upright(path, cv2, pytesseract)
    bw = binarize(img, cv2)
    rows = detect_cells(bw, cv2)
    height, width = bw.shape[:2]
    table = []
    for row in rows:
        row_text = []
        for x, y, w, h in row:
            pad = 3
            crop = bw[max(0, y + pad) : min(height, y + h - pad), max(0, x + pad) : min(width, x + w - pad)]
            row_text.append(ocr_cell(crop, pytesseract))
        table.append(row_text)
    return table


def pairs_from_table(table, page):
    if not table or len(table) < 2:
        return [], {"page": page, "rows": len(table), "cols": 0, "pairs": 0, "unmapped": 0}
    col_heads = table[0][1:]
    pairs = []
    unmapped = 0
    for row in table[1:]:
        if not row:
            continue
        final_raw = row[0]
        for idx, cell in enumerate(row[1:]):
            if idx >= len(col_heads) or not cell.replace(" ", ""):
                continue
            initial_raw = col_heads[idx]
            initial_code = map_initial(initial_raw)
            final_code = map_final(final_raw)
            if initial_code is None or final_code is None:
                unmapped += 1
            pairs.append(
                {
                    "initial_raw": initial_raw,
                    "final_raw": final_raw,
                    "initial_code": initial_code or "",
                    "final_code": final_code or "",
                    "source_page": page,
                    "status": "mapped" if initial_code is not None and final_code is not None else "needs_review",
                }
            )
    return pairs, {
        "page": page,
        "rows": len(table),
        "cols": max(len(row) for row in table),
        "pairs": len(pairs),
        "unmapped": unmapped,
    }


def write_outputs(raw_rows, review_rows, stats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "legal_pairs_raw.tsv"
    review_path = out_dir / "legal_pairs_review.tsv"
    report_path = out_dir / "ocr_report.md"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["initial_raw", "final_raw", "source_page"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "initial_raw",
                "final_raw",
                "initial_code",
                "final_code",
                "source_page",
                "status",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(review_rows)
    needs_review = sum(1 for row in review_rows if row["status"] == "needs_review")
    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Jie Yong Ki / 張永愷 OCR Report\n\n")
        f.write(f"- Raw pairs: {len(raw_rows)}\n")
        f.write(f"- Review rows: {len(review_rows)}\n")
        f.write(f"- Rows needing review: {needs_review}\n\n")
        f.write("| Page | Rows | Columns | Pairs | Unmapped |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for item in stats:
            f.write(
                f"| {item['page']} | {item['rows']} | {item['cols']} | "
                f"{item['pairs']} | {item['unmapped']} |\n"
            )
    return raw_path, review_path, report_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--from-official",
        action="store_true",
        help="Seed review outputs from ruian_legal_pairs.tsv instead of running OCR.",
    )
    parser.add_argument("images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)

    if args.from_official:
        review_rows = read_official_pairs(ROOT / "ruian_legal_pairs.tsv")
        raw_rows = [
            {
                "initial_raw": row["initial_raw"],
                "final_raw": row["final_raw"],
                "source_page": row["source_page"],
            }
            for row in review_rows
        ]
        stats = [
            {
                "page": "ruian_legal_pairs.tsv",
                "rows": 0,
                "cols": 0,
                "pairs": len(review_rows),
                "unmapped": 0,
            }
        ]
    else:
        raw_rows = []
        review_rows = []
        stats = []
        for image in args.images:
            path = image if image.is_absolute() else Path(__file__).resolve().parent / image
            table = extract_table(path)
            pairs, stat = pairs_from_table(table, path.stem)
            review_rows.extend(pairs)
            raw_rows.extend(
                {
                    "initial_raw": row["initial_raw"],
                    "final_raw": row["final_raw"],
                    "source_page": row["source_page"],
                }
                for row in pairs
            )
            stats.append(stat)

    outputs = write_outputs(raw_rows, review_rows, stats, args.out_dir)
    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
