#!/usr/bin/env python3
"""Compare Tesseract and PaddleOCR on Jie Yong Ki dictionary table cells."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

from PIL import Image

import ocr_jie_yong_ki_book_table as table_ocr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "ocr_backend_trials"
STRUCTURED_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "book_ocr_structured"
BACKEND_FIELDS = [*table_ocr.CELL_FIELDS, "backend"]


def write_backend_cells(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BACKEND_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_tesseract_page(page: int) -> list[dict[str, str]]:
    path = STRUCTURED_OUT_DIR / f"page_{page:03d}_cells.tsv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row, backend="tesseract") for row in csv.DictReader(f, delimiter="\t")]


def read_existing_backend_page(path: Path, backend: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row, backend=backend) for row in csv.DictReader(f, delimiter="\t")]


def make_tesseract_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=STRUCTURED_OUT_DIR,
        lang=args.tesseract_lang,
        psm=args.tesseract_psm,
        min_conf=args.min_conf,
        min_row_fraction=args.min_row_fraction,
        min_col_fraction=args.min_col_fraction,
        crop_pad=args.crop_pad,
        blank_dark_fraction=args.blank_dark_fraction,
        keep_cell_images=False,
    )


def find_image(args: argparse.Namespace) -> tuple[int, Path]:
    images = table_ocr.collect_images(args.image_dir, args.start_after)
    wanted = int(args.page)
    for number, path in images:
        if number == wanted:
            return number, path
    raise SystemExit(f"Page {wanted:03d} was not found in {args.image_dir}")


def tesseract_cells(page: int, image_path: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.rerun_tesseract:
        existing = read_existing_tesseract_page(page)
        if existing:
            return existing
    cells, _, _ = table_ocr.ocr_page(page, image_path, make_tesseract_args(args))
    return [dict(cell, backend="tesseract") for cell in cells]


def require_paddleocr() -> tuple[object | None, str | None]:
    if importlib.util.find_spec("paddleocr") is None:
        return None, (
            "PaddleOCR is not installed. Install it with:\n\n"
            "    python -m pip install paddleocr paddlepaddle\n\n"
            "Then rerun:\n\n"
            "    python tools/ocr/compare_ocr_backends.py --page 059"
        )
    paddle_home = DEFAULT_OUT_DIR / "_paddle_home"
    paddle_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(paddle_home)
    os.environ["USERPROFILE"] = str(paddle_home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_home / "paddlex")
    os.environ["PADDLE_HOME"] = str(paddle_home / "paddle")
    os.environ["XDG_CACHE_HOME"] = str(paddle_home / "cache")
    os.environ["HF_HOME"] = str(paddle_home / "huggingface")
    os.environ["MODELSCOPE_CACHE"] = str(paddle_home / "modelscope")
    os.environ.setdefault("PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG", "1")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("PADDLE_PDX_USE_PIR_TRT", "False")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover - depends on optional package.
        return None, f"PaddleOCR import failed: {exc}"

    init_attempts = [
        {"use_textline_orientation": False, "lang": "ch"},
        {"use_angle_cls": False, "lang": "ch"},
        {"lang": "ch"},
    ]
    errors = []
    for kwargs in init_attempts:
        try:
            return PaddleOCR(**kwargs), None
        except Exception as exc:  # pragma: no cover - depends on optional package.
            errors.append(f"{kwargs}: {exc}")
    return None, "PaddleOCR initialization failed:\n" + "\n".join(errors)


def parse_paddle_result(result: object) -> tuple[str, float]:
    texts: list[str] = []
    confidences: list[float] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key in ("rec_texts", "texts"):
                items = value.get(key)
                if isinstance(items, list):
                    texts.extend(str(item) for item in items if str(item).strip())
            for key in ("rec_scores", "scores"):
                items = value.get(key)
                if isinstance(items, list):
                    for item in items:
                        try:
                            confidences.append(float(item) * 100)
                        except (TypeError, ValueError):
                            pass
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            if len(value) >= 2 and isinstance(value[1], (list, tuple)) and len(value[1]) >= 2:
                maybe_text, maybe_conf = value[1][0], value[1][1]
                if isinstance(maybe_text, str):
                    texts.append(maybe_text)
                    try:
                        confidences.append(float(maybe_conf) * 100)
                    except (TypeError, ValueError):
                        pass
                    return
            for item in value:
                walk(item)

    walk(result)
    text = table_ocr.normalize_text(" ".join(texts))
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return text, confidence


def paddle_ocr_cell(ocr: object, image: Image.Image, image_path: Path) -> tuple[str, float]:
    image.save(image_path)
    if hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(str(image_path))
        except TypeError:
            result = ocr.ocr(str(image_path), cls=False)
    elif hasattr(ocr, "predict"):
        result = ocr.predict(str(image_path))
    else:
        raise RuntimeError("Unsupported PaddleOCR object: missing ocr()/predict().")
    return parse_paddle_result(result)


def paddle_cells(page: int, image_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, str]], str | None]:
    paddle_path = args.out_dir / f"page_{page:03d}_paddle_cells.tsv"
    if not args.rerun_paddle:
        existing = read_existing_backend_page(paddle_path, "paddleocr")
        if existing:
            return existing, None

    ocr, error = require_paddleocr()
    if error:
        return [], error

    image = Image.open(image_path).convert("RGB")
    boxes_by_row, _ = table_ocr.detect_cell_boxes(image, args.min_row_fraction, args.min_col_fraction)
    width, height = image.size
    tmp_dir = args.out_dir / "_tmp_cells" / f"paddle_{page:03d}_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, str]] = []

    for row_index, boxes in enumerate(boxes_by_row, 1):
        for col_index, box in enumerate(boxes, 1):
            column_name = table_ocr.COLUMN_NAMES[col_index - 1] if col_index <= len(table_ocr.COLUMN_NAMES) else f"extra_{col_index}"
            crop_box = box.padded(width, height, args.crop_pad)
            crop = image.crop((crop_box.left, crop_box.top, crop_box.right, crop_box.bottom))
            if table_ocr.is_blank_crop(crop, args.blank_dark_fraction):
                text, confidence = "", 0.0
            else:
                text, confidence = paddle_ocr_cell(
                    ocr,
                    crop,
                    tmp_dir / f"page_{page:03d}_r{row_index:03d}_c{col_index:02d}.png",
                )
            status, note = table_ocr.classify_cell(column_name, text, confidence, args.min_conf)
            cells.append(
                {
                    "page": f"{page:03d}",
                    "row": str(row_index),
                    "column": str(col_index),
                    "column_name": column_name,
                    "text": text,
                    "confidence": f"{confidence:.2f}",
                    "bbox": box.as_string(),
                    "status": status,
                    "note": note,
                    "backend": "paddleocr",
                }
            )
    return cells, None


def build_rows(page: int, cells: list[dict[str, str]], min_conf: float) -> list[dict[str, str]]:
    rows = []
    row_ids = sorted({cell["row"] for cell in cells}, key=lambda value: int(value))
    for row_id in row_ids:
        logical = table_ocr.build_dictionary_row(page, int(row_id), [cell for cell in cells if cell["row"] == row_id], min_conf)
        if logical:
            rows.append(logical)
    return rows


def metrics(cells: list[dict[str, str]], rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "cells": len(cells),
        "rows": len(rows),
        "review_cells": sum(1 for cell in cells if cell["status"] != "ok"),
        "suspicious_cells": sum(1 for cell in cells if cell["status"] == "suspicious"),
        "low_confidence_cells": sum(1 for cell in cells if "low_confidence" in cell["note"]),
        "hanzi_cells": sum(1 for cell in cells if cell["column_name"] == "hanzi" and table_ocr.CJK_RE.search(cell["text"])),
        "missing_hanzi_rows": sum(1 for row in rows if "missing_hanzi" in row["issues"]),
        "ok_rows": sum(1 for row in rows if row["status"] == "ok"),
    }


def write_report(path: Path, page: int, tesseract: list[dict[str, str]], paddle: list[dict[str, str]], paddle_error: str | None, min_conf: float) -> None:
    t_rows = build_rows(page, tesseract, min_conf)
    p_rows = build_rows(page, paddle, min_conf)
    table = [
        ("tesseract", metrics(tesseract, t_rows)),
        ("paddleocr", metrics(paddle, p_rows)),
    ]
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# OCR Backend Comparison\n\n")
        f.write(f"- Page: {page:03d}\n")
        f.write("- Grid/cell segmentation: shared PIL/numpy detector\n")
        if paddle_error:
            f.write(f"- PaddleOCR status: unavailable\n\n```text\n{paddle_error}\n```\n\n")
        else:
            f.write("- PaddleOCR status: available\n\n")
        f.write("| Backend | Cells | Rows | Hanzi cells | Missing-hanzi rows | Review cells | Suspicious cells | Low-conf cells | OK rows |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for name, item in table:
            f.write(
                f"| {name} | {item['cells']} | {item['rows']} | {item['hanzi_cells']} | "
                f"{item['missing_hanzi_rows']} | {item['review_cells']} | {item['suspicious_cells']} | "
                f"{item['low_confidence_cells']} | {item['ok_rows']} |\n"
            )

        f.write("\n## Hanzi Column Samples\n\n")
        f.write("| Row | Tesseract | PaddleOCR |\n")
        f.write("|---:|---|---|\n")
        t_hanzi = {cell["row"]: cell["text"] for cell in tesseract if cell["column_name"] == "hanzi"}
        p_hanzi = {cell["row"]: cell["text"] for cell in paddle if cell["column_name"] == "hanzi"}
        for row in sorted(set(t_hanzi) | set(p_hanzi), key=lambda value: int(value))[:40]:
            f.write(f"| {row} | {t_hanzi.get(row, '')} | {p_hanzi.get(row, '')} |\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="059")
    parser.add_argument("--image-dir", type=Path, default=table_ocr.DEFAULT_IMAGE_DIR)
    parser.add_argument("--start-after", type=int, default=58)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rerun-tesseract", action="store_true")
    parser.add_argument("--rerun-paddle", action="store_true")
    parser.add_argument("--tesseract-lang", default="chi_sim+eng")
    parser.add_argument("--tesseract-psm", type=int, default=0)
    parser.add_argument("--min-conf", type=float, default=45.0)
    parser.add_argument("--min-row-fraction", type=float, default=0.18)
    parser.add_argument("--min-col-fraction", type=float, default=0.24)
    parser.add_argument("--crop-pad", type=int, default=3)
    parser.add_argument("--blank-dark-fraction", type=float, default=0.012)
    args = parser.parse_args(argv)

    page, image_path = find_image(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing OCR backends on page {page:03d}")
    t_cells = tesseract_cells(page, image_path, args)
    write_backend_cells(args.out_dir / f"page_{page:03d}_tesseract_cells.tsv", t_cells)

    p_cells, paddle_error = paddle_cells(page, image_path, args)
    write_backend_cells(args.out_dir / f"page_{page:03d}_paddle_cells.tsv", p_cells)

    report = args.out_dir / "backend_comparison_report.md"
    write_report(report, page, t_cells, p_cells, paddle_error, args.min_conf)
    print(f"Wrote Tesseract trial cells: {args.out_dir / f'page_{page:03d}_tesseract_cells.tsv'}")
    print(f"Wrote PaddleOCR trial cells: {args.out_dir / f'page_{page:03d}_paddle_cells.tsv'}")
    print(f"Wrote backend comparison report: {report}")
    if paddle_error:
        print("\nPaddleOCR is unavailable:")
        print(paddle_error)


if __name__ == "__main__":
    main()
