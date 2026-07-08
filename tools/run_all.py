#!/usr/bin/env python3
"""One-command pipeline for Ruianese/Jie Yong Ki dictionary generation and OCR."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT.parents[1] / "張永愷書之圖"
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki"
BOOK_OCR_DIR = DEFAULT_OUT_DIR / "book_ocr"
TESSERACT_EXE = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
PAGE_RE = re.compile(r"页面_(\d+)\.png$")


def run_python(script, *args):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, str(script), *map(str, args)]
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def page_number(path):
    match = PAGE_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def collect_images(image_dir, start_after):
    images = []
    for path in image_dir.glob("*.png"):
        number = page_number(path)
        if number is not None and number > start_after:
            images.append((number, path))
    return sorted(images, key=lambda item: item[0])


def tesseract_command():
    if TESSERACT_EXE.exists():
        return str(TESSERACT_EXE)
    return "tesseract"


def run_tesseract(image, output_base, lang, psm):
    command = [
        tesseract_command(),
        str(image),
        str(output_base),
        "-l",
        lang,
        "--psm",
        str(psm),
        "txt",
        "tsv",
    ]
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_tsv_confidence(path):
    confidences = []
    if not path.exists():
        return confidences
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
                confidences.append(conf)
    return confidences


def validate_page(number, image, out_base, min_chars, min_conf):
    txt_path = out_base.with_suffix(".txt")
    tsv_path = out_base.with_suffix(".tsv")
    text = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
    compact = re.sub(r"\s+", "", text)
    han_count = len(re.findall(r"[\u3400-\u9fff]", text))
    confidences = parse_tsv_confidence(tsv_path)
    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    low_conf_words = sum(1 for conf in confidences if conf < min_conf)
    issues = []
    if not txt_path.exists():
        issues.append("missing_txt")
    if not tsv_path.exists():
        issues.append("missing_tsv")
    if len(compact) < min_chars:
        issues.append("short_text")
    if confidences and avg_conf < min_conf:
        issues.append("low_confidence")
    if not confidences:
        issues.append("no_confidence")
    return {
        "page": f"{number:03d}",
        "image": str(image),
        "txt": str(txt_path),
        "tsv": str(tsv_path),
        "chars": str(len(compact)),
        "han_chars": str(han_count),
        "word_count": str(len(confidences)),
        "avg_conf": f"{avg_conf:.2f}",
        "low_conf_words": str(low_conf_words),
        "status": "ok" if not issues else "review",
        "issues": ",".join(issues),
    }


def write_validation(rows, images, image_dir, start_after, out_dir):
    index_path = out_dir / "pages_index.tsv"
    report_path = out_dir / "ocr_validation_report.md"
    combined_path = out_dir / "all_pages.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "page",
        "image",
        "txt",
        "tsv",
        "chars",
        "han_chars",
        "word_count",
        "avg_conf",
        "low_conf_words",
        "status",
        "issues",
    ]
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with combined_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            txt_path = Path(row["txt"])
            f.write(f"\n\n===== Page {row['page']} =====\n\n")
            if txt_path.exists():
                f.write(txt_path.read_text(encoding="utf-8", errors="ignore").strip())
                f.write("\n")

    page_numbers = [number for number, _ in images]
    expected = set(range(min(page_numbers), max(page_numbers) + 1)) if page_numbers else set()
    missing = sorted(expected - set(page_numbers))
    review_rows = [row for row in rows if row["status"] != "ok"]
    avg_conf_values = [float(row["avg_conf"]) for row in rows if row["avg_conf"]]
    overall_conf = round(sum(avg_conf_values) / len(avg_conf_values), 2) if avg_conf_values else 0.0

    with report_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Jie Yong Ki / 張永愷 Book OCR Validation\n\n")
        f.write(f"- Image directory: `{image_dir}`\n")
        f.write(f"- Start after page: {start_after:03d}\n")
        f.write(f"- Pages selected: {len(images)}\n")
        if page_numbers:
            f.write(f"- Page range: {min(page_numbers):03d}-{max(page_numbers):03d}\n")
        f.write(f"- Missing page numbers in selected range: {len(missing)}\n")
        f.write(f"- Pages needing review: {len(review_rows)}\n")
        f.write(f"- Mean page confidence: {overall_conf:.2f}\n\n")
        f.write(f"- Combined OCR text: `{combined_path}`\n\n")
        if missing:
            f.write("## Missing Pages\n\n")
            f.write(", ".join(f"{number:03d}" for number in missing[:200]))
            f.write("\n\n")
        f.write("## Review Pages\n\n")
        f.write("| Page | Chars | Han chars | Avg conf | Issues |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for row in review_rows[:200]:
            f.write(
                f"| {row['page']} | {row['chars']} | {row['han_chars']} | "
                f"{row['avg_conf']} | {row['issues']} |\n"
            )
    return index_path, report_path, combined_path


def run_book_ocr(args):
    image_dir = args.image_dir
    images = collect_images(image_dir, args.start_after)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No page images found after {args.start_after:03d} in {image_dir}")
    out_dir = args.out_dir / "book_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"OCR pages: {len(images)} from {image_dir}")
    rows = []
    for index, (number, image) in enumerate(images, 1):
        out_base = out_dir / f"page_{number:03d}"
        txt_path = out_base.with_suffix(".txt")
        tsv_path = out_base.with_suffix(".tsv")
        if not args.skip_existing or not (txt_path.exists() and tsv_path.exists()):
            print(f"[{index}/{len(images)}] OCR page {number:03d}")
            run_tesseract(image, out_base, args.lang, args.psm)
        rows.append(validate_page(number, image, out_base, args.min_chars, args.min_conf))
    index_path, report_path, combined_path = write_validation(rows, images, image_dir, args.start_after, out_dir)
    print(f"Wrote OCR index: {index_path}")
    print(f"Wrote OCR validation: {report_path}")
    print(f"Wrote combined OCR text: {combined_path}")


def validate_dicts():
    valid = {
        line.split("\t", 1)[0]
        for line in (ROOT / "ruian_pinyin.dict.yaml").read_text(encoding="utf-8").splitlines()
        if "\t" in line
    }
    jie_entries = [
        tuple(line.split("\t")[:3])
        for line in (ROOT / "ruianese.jie_yong_ki.dict.yaml").read_text(encoding="utf-8").splitlines()
        if "\t" in line
    ]
    converted_rows = [
        line
        for line in (DEFAULT_OUT_DIR / "ruianese_characters_converted.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    invalid = [(char, code) for char, code, _ in jie_entries if code not in valid]
    if invalid:
        raise SystemExit(f"Invalid Jie Yong Ki codes: {invalid[:10]}")
    print(f"Validated: {len(valid)} pinyin syllables, {len(converted_rows)} converted rows, {len(jie_entries)} formal entries")


def run_vlm_clean_rows(args):
    vlm_dir = args.out_dir / "vlm_backend_trials"
    raw = vlm_dir / f"page_{args.vlm_clean_page:03d}_paddleocr_vl_raw.json"
    if not raw.exists():
        print(f"Skip VLM clean rows: missing {raw}")
        return
    run_python(
        ROOT / "tools" / "ocr" / "build_clean_vlm_rows.py",
        "--page",
        f"{args.vlm_clean_page:03d}",
        "--image-dir",
        args.image_dir,
        "--out-dir",
        vlm_dir,
        "--start-after",
        args.start_after,
    )


def run_ipa_review(args):
    vlm_dir = args.out_dir / "vlm_backend_trials"
    clean_rows = vlm_dir / f"page_{args.vlm_clean_page:03d}_clean_rows.tsv"
    if not clean_rows.exists():
        print(f"Skip IPA review: missing {clean_rows}")
        return
    run_python(
        ROOT / "tools" / "ocr" / "review_ipa_with_llm.py",
        "--clean-rows",
        clean_rows,
        "--out-dir",
        vlm_dir,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-after", type=int, default=58)
    parser.add_argument("--limit", type=int, default=0, help="OCR only the first N selected pages; 0 means all.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--skip-book-ocr", action="store_true")
    parser.add_argument("--skip-structured-book-ocr", action="store_true")
    parser.add_argument("--skip-vlm-clean-rows", action="store_true")
    parser.add_argument("--skip-ipa-review", action="store_true")
    parser.add_argument("--vlm-clean-page", type=int, default=59)
    parser.add_argument(
        "--structured-limit",
        type=int,
        default=0,
        help="Structured table OCR only the first N selected pages; 0 means all.",
    )
    parser.add_argument("--lang", default="chi_sim+eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--min-conf", type=float, default=45.0)
    args = parser.parse_args(argv)

    run_python(ROOT / "tools" / "generate_ruian_pinyin_dict.py")
    run_python(ROOT / "tools" / "ocr" / "ocr_ruian_table.py", "--from-official")
    run_python(ROOT / "tools" / "convert_jie_yong_ki_character_dict.py")
    run_python(ROOT / "tools" / "generate_jie_yong_ki_dict.py")
    validate_dicts()
    if not args.skip_book_ocr:
        run_book_ocr(args)
    if not args.skip_structured_book_ocr:
        structured_args = [
            "--image-dir",
            args.image_dir,
            "--out-dir",
            args.out_dir / "book_ocr_structured",
            "--start-after",
            args.start_after,
            "--lang",
            args.lang,
            "--min-conf",
            args.min_conf,
        ]
        if args.structured_limit:
            structured_args.extend(["--limit", args.structured_limit])
        if not args.skip_existing:
            structured_args.append("--rerun-existing")
        run_python(ROOT / "tools" / "ocr" / "ocr_jie_yong_ki_book_table.py", *structured_args)
    if not args.skip_vlm_clean_rows:
        run_vlm_clean_rows(args)
    if not args.skip_ipa_review:
        run_ipa_review(args)


if __name__ == "__main__":
    main()
