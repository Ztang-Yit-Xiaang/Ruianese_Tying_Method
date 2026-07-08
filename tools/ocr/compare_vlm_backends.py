#!/usr/bin/env python3
"""Validity-first VLM OCR trials for Jie Yong Ki / 張永愷 dictionary pages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

import ocr_jie_yong_ki_book_table as table_ocr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "vlm_backend_trials"
COLUMNS = [
    "page",
    "row",
    "hanzi",
    "pinyin",
    "fanqie",
    "tone_base",
    "tone_register",
    "tone_number",
    "rhyme",
    "initial",
    "ipa",
    "homophone",
    "note",
    "status",
    "review_note",
]
TONE_BASES = {"平", "上", "去", "入", ""}
TONE_REGISTERS = {"yin", "yang", "unknown", ""}
TONE_NUMBERS = {str(index) for index in range(1, 9)} | {""}
TONE_NUMBER_MAP = {
    ("平", "yin"): "1",
    ("上", "yin"): "2",
    ("去", "yin"): "3",
    ("入", "yin"): "4",
    ("平", "yang"): "5",
    ("上", "yang"): "6",
    ("去", "yang"): "7",
    ("入", "yang"): "8",
}
QWEN_PROMPT = """Read this scanned Rui'an dialect dictionary table page.

Return ONLY valid JSON in this shape:
{
  "rows": [
    {
      "row": 1,
      "hanzi": "",
      "pinyin": "",
      "fanqie": "",
      "tone_base": "平|上|去|入|null",
      "tone_register": "yin|yang|unknown",
      "tone_number": "1|2|3|4|5|6|7|8|null",
      "rhyme": "",
      "initial": "",
      "ipa": "",
      "homophone": "",
      "note": "",
      "review_note": ""
    }
  ]
}

Rules:
- Do not guess. Use null and explain in review_note when uncertain.
- Tone number mapping: yin 平上去入 = 1,2,3,4; yang 平上去入 = 5,6,7,8.
- Underline/dot tone marks distinguish yang from yin.
- Keep IPA as visually seen; do not normalize it into pinyin.
- Preserve traditional/simplified variants in hanzi when visible.
"""


def configure_model_cache(out_dir: Path) -> None:
    home = out_dir / "_model_home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(home / "paddlex")
    os.environ["PADDLE_HOME"] = str(home / "paddle")
    os.environ["XDG_CACHE_HOME"] = str(home / "cache")
    os.environ["HF_HOME"] = str(home / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(home / "huggingface" / "transformers")
    os.environ["MODELSCOPE_CACHE"] = str(home / "modelscope")
    os.environ.setdefault("PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG", "1")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("PADDLE_PDX_USE_PIR_TRT", "False")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")


def find_image(args: argparse.Namespace) -> tuple[int, Path]:
    images = table_ocr.collect_images(args.image_dir, args.start_after)
    wanted = int(args.page)
    for number, path in images:
        if number == wanted:
            return number, path
    raise SystemExit(f"Page {wanted:03d} was not found in {args.image_dir}")


def safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if hasattr(value, "json"):
        try:
            return safe_json(value.json)
        except Exception:
            pass
    return repr(value)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(data), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def status_for_row(row: dict[str, str]) -> tuple[str, str]:
    notes = []
    if not table_ocr.CJK_RE.search(row.get("hanzi", "")):
        notes.append("missing_hanzi")
    tone_base = row.get("tone_base", "")
    tone_register = row.get("tone_register", "")
    tone_number = row.get("tone_number", "")
    if tone_base not in TONE_BASES:
        notes.append("invalid_tone_base")
    if tone_register not in TONE_REGISTERS:
        notes.append("invalid_tone_register")
    if tone_number not in TONE_NUMBERS:
        notes.append("invalid_tone_number")
    if tone_number and (not tone_base or tone_register in {"", "unknown"}):
        notes.append("tone_number_without_full_tone")
    expected = TONE_NUMBER_MAP.get((tone_base, tone_register))
    if tone_number and expected and tone_number != expected:
        notes.append(f"tone_number_mismatch_expected_{expected}")
    review_note = row.get("review_note", "")
    if review_note:
        notes.append(review_note)
    return ("review" if notes else "ok"), ",".join(dict.fromkeys(note for note in notes if note))


def normalize_row(page: int, index: int, raw: dict[str, Any]) -> dict[str, str]:
    row = {field: "" for field in COLUMNS}
    row["page"] = f"{page:03d}"
    row["row"] = normalize_cell(raw.get("row")) or str(index)
    for field in COLUMNS:
        if field in {"page", "row", "status"}:
            continue
        row[field] = normalize_cell(raw.get(field))
    if row["tone_base"].lower() in {"none", "null"}:
        row["tone_base"] = ""
    if row["tone_number"].lower() in {"none", "null"}:
        row["tone_number"] = ""
    if row["tone_register"].lower() in {"none", "null"}:
        row["tone_register"] = "unknown"
    row["status"], validation_note = status_for_row(row)
    if validation_note:
        row["review_note"] = ",".join(dict.fromkeys(filter(None, [row["review_note"], validation_note])))
    return row


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def flatten_text(value: Any) -> str:
    chunks: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict):
            for key in ("markdown", "md", "html", "text", "content"):
                if key in item and isinstance(item[key], str):
                    chunks.append(item[key])
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif hasattr(item, "json"):
            try:
                walk(item.json)
            except Exception:
                chunks.append(repr(item))
        else:
            chunks.append(repr(item))

    walk(value)
    return "\n".join(dict.fromkeys(chunk for chunk in chunks if chunk.strip()))


def markdown_rows_to_tsv_rows(page: int, markdown: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    table_lines = [line.strip() for line in markdown.splitlines() if "|" in line]
    header: list[str] | None = None
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if header is None:
            header = cells
            continue
        raw = {}
        for key, value in zip(header, cells):
            canonical = {
                "汉字": "hanzi",
                "漢字": "hanzi",
                "拼音": "pinyin",
                "反切": "fanqie",
                "调": "tone_base",
                "調": "tone_base",
                "韵": "rhyme",
                "韻": "rhyme",
                "声": "initial",
                "聲": "initial",
                "国际音标": "ipa",
                "國際音標": "ipa",
                "同音字": "homophone",
                "备注": "note",
                "備註": "note",
            }.get(key, key)
            raw[canonical] = value
        rows.append(normalize_row(page, len(rows) + 1, raw))
    return rows


def html_table_rows_to_tsv_rows(page: int, text: str) -> list[dict[str, str]]:
    if "<table" not in text.lower():
        return []
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []

    soup = BeautifulSoup(text, "html.parser")
    rows: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = [normalize_cell(cell.get_text(" ", strip=True)) for cell in tr.find_all(["td", "th"])]
            if len(cells) < 10:
                continue
            if any(label in cells for label in ["汉字", "漢字", "拼音", "反切"]):
                continue
            raw = {
                "hanzi": cells[1],
                "pinyin": cells[2],
                "fanqie": cells[3],
                "tone_base": cells[4],
                "tone_register": "unknown",
                "tone_number": "",
                "rhyme": cells[5],
                "initial": cells[6],
                "ipa": cells[7],
                "homophone": cells[8],
                "note": cells[9],
                "review_note": "tone_register_unresolved",
            }
            rows.append(normalize_row(page, len(rows) + 1, raw))
    return rows


def extract_json_rows(page: int, text: str) -> list[dict[str, str]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    raw_rows = data.get("rows", []) if isinstance(data, dict) else []
    return [normalize_row(page, index, raw) for index, raw in enumerate(raw_rows, 1) if isinstance(raw, dict)]


def load_cached_paddle_rows(page: int, args: argparse.Namespace) -> list[dict[str, str]]:
    raw_json_path = args.out_dir / f"page_{page:03d}_paddleocr_vl_raw.json"
    if args.rerun_paddle_vl or not raw_json_path.exists():
        return []
    data = json.loads(raw_json_path.read_text(encoding="utf-8"))
    raw_text = flatten_text(data)
    write_text(args.out_dir / f"page_{page:03d}_paddleocr_vl_raw.md", raw_text)
    return extract_json_rows(page, raw_text) or html_table_rows_to_tsv_rows(page, raw_text) or markdown_rows_to_tsv_rows(page, raw_text)


def load_cached_qwen_rows(page: int, args: argparse.Namespace) -> list[dict[str, str]]:
    raw_json_path = args.out_dir / f"page_{page:03d}_qwen_vl_raw.json"
    if args.rerun_qwen or not raw_json_path.exists():
        return []
    data = json.loads(raw_json_path.read_text(encoding="utf-8"))
    output = data.get("output", "") if isinstance(data, dict) else ""
    return extract_json_rows(page, output)


def run_paddleocr_vl(page: int, image_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, str]]:
    started = time.time()
    cached_rows = load_cached_paddle_rows(page, args)
    if cached_rows:
        return cached_rows, {"status": "ok", "runtime": "0.00", "note": "parsed_cached_raw_output"}
    try:
        from paddleocr import PaddleOCRVL
    except Exception as exc:
        return [], {"status": "blocked", "runtime": "0.00", "note": f"PaddleOCRVL import failed: {exc}"}

    raw_json_path = args.out_dir / f"page_{page:03d}_paddleocr_vl_raw.json"
    raw_md_path = args.out_dir / f"page_{page:03d}_paddleocr_vl_raw.md"
    try:
        pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            vl_rec_backend="native",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            format_block_content=True,
            merge_layout_blocks=True,
        )
        result = pipeline.predict(
            str(image_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            format_block_content=True,
            merge_layout_blocks=True,
            max_new_tokens=args.max_new_tokens,
        )
        write_json(raw_json_path, result)
        raw_text = flatten_text(result)
        write_text(raw_md_path, raw_text)
        rows = extract_json_rows(page, raw_text) or html_table_rows_to_tsv_rows(page, raw_text) or markdown_rows_to_tsv_rows(page, raw_text)
        status = "ok" if rows else "review"
        note = "" if rows else "no_structured_rows_extracted"
        return rows, {"status": status, "runtime": f"{time.time() - started:.2f}", "note": note}
    except Exception as exc:
        details = traceback.format_exc()
        write_text(raw_md_path, f"PaddleOCR-VL failed:\n{exc}\n\n{details}\n")
        return [], {"status": "blocked", "runtime": f"{time.time() - started:.2f}", "note": f"{exc} See raw.md for traceback."}


def qwen_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QWEN_PROMPT},
            ],
        }
    ]


def run_qwen_vl(page: int, image_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, str]]:
    started = time.time()
    raw_json_path = args.out_dir / f"page_{page:03d}_qwen_vl_raw.json"
    cached_rows = load_cached_qwen_rows(page, args)
    if cached_rows:
        return cached_rows, {"status": "ok", "runtime": "0.00", "note": "parsed_cached_raw_output"}
    try:
        import torch
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        return [], {"status": "blocked", "runtime": "0.00", "note": f"Qwen imports failed: {exc}"}

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.qwen_model,
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(args.qwen_model)
        image = Image.open(image_path).convert("RGB")
        if args.qwen_max_image_edge and max(image.size) > args.qwen_max_image_edge:
            ratio = args.qwen_max_image_edge / max(image.size)
            image = image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))))
        messages = qwen_messages()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.qwen_max_new_tokens, do_sample=False)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        write_json(raw_json_path, {"model": args.qwen_model, "output": output_text})
        rows = extract_json_rows(page, output_text)
        status = "ok" if rows else "review"
        note = "" if rows else "no_valid_json_rows_extracted"
        return rows, {"status": status, "runtime": f"{time.time() - started:.2f}", "note": note}
    except Exception as exc:
        write_json(raw_json_path, {"model": args.qwen_model, "error": str(exc), "traceback": traceback.format_exc()})
        return [], {"status": "blocked", "runtime": f"{time.time() - started:.2f}", "note": f"{exc} See raw.json for traceback."}


def row_metrics(rows: list[dict[str, str]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "review_rows": sum(1 for row in rows if row["status"] != "ok"),
        "hanzi_rows": sum(1 for row in rows if table_ocr.CJK_RE.search(row.get("hanzi", ""))),
        "tone_resolved": sum(1 for row in rows if row.get("tone_number") in {str(i) for i in range(1, 9)}),
        "tone_unresolved": sum(1 for row in rows if not row.get("tone_number")),
        "invalid_tone": sum(
            1
            for row in rows
            if row.get("tone_base") not in TONE_BASES
            or row.get("tone_register") not in TONE_REGISTERS
            or row.get("tone_number") not in TONE_NUMBERS
            or "tone_number_mismatch" in row.get("review_note", "")
        ),
    }


def write_report(path: Path, stats: dict[str, dict[str, str]], rows_by_backend: dict[str, list[dict[str, str]]]) -> None:
    lines = ["# VLM Backend Comparison", ""]
    lines.append("| Backend | Status | Runtime sec | Rows | Hanzi rows | Review rows | Tone resolved | Tone unresolved | Invalid tone | Note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for backend in ["paddleocr_vl", "qwen_vl"]:
        metric = row_metrics(rows_by_backend.get(backend, []))
        stat = stats.get(backend, {})
        lines.append(
            f"| {backend} | {stat.get('status', 'not_run')} | {stat.get('runtime', '')} | "
            f"{metric['rows']} | {metric['hanzi_rows']} | {metric['review_rows']} | "
            f"{metric['tone_resolved']} | {metric['tone_unresolved']} | {metric['invalid_tone']} | "
            f"{stat.get('note', '')} |"
        )
    lines.append("")
    lines.append("## Validity Policy")
    lines.append("")
    lines.append("- Tone number must be `1-8` or blank.")
    lines.append("- Tone base must be `平`, `上`, `去`, `入`, or blank.")
    lines.append("- Tone register must be `yin`, `yang`, `unknown`, or blank.")
    lines.append("- Review rows are acceptable; confident invalid tone values are not.")
    write_text(path, "\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="059")
    parser.add_argument("--image-dir", type=Path, default=table_ocr.DEFAULT_IMAGE_DIR)
    parser.add_argument("--start-after", type=int, default=58)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", choices=["all", "paddle", "qwen"], default="all")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=4096)
    parser.add_argument("--qwen-max-image-edge", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--rerun-paddle-vl", action="store_true")
    parser.add_argument("--rerun-qwen", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_model_cache(args.out_dir)
    page, image_path = find_image(args)

    stats: dict[str, dict[str, str]] = {}
    rows_by_backend: dict[str, list[dict[str, str]]] = {}
    if args.backend in {"all", "paddle"}:
        print(f"Running PaddleOCR-VL on page {page:03d}")
        rows, stat = run_paddleocr_vl(page, image_path, args)
        rows_by_backend["paddleocr_vl"] = rows
        stats["paddleocr_vl"] = stat
        write_rows(args.out_dir / f"page_{page:03d}_paddleocr_vl_rows.tsv", rows)
    if args.backend in {"all", "qwen"}:
        print(f"Running Qwen2.5-VL on page {page:03d}: {args.qwen_model}")
        rows, stat = run_qwen_vl(page, image_path, args)
        rows_by_backend["qwen_vl"] = rows
        stats["qwen_vl"] = stat
        write_rows(args.out_dir / f"page_{page:03d}_qwen_vl_rows.tsv", rows)

    if "paddleocr_vl" not in rows_by_backend:
        cached = read_rows(args.out_dir / f"page_{page:03d}_paddleocr_vl_rows.tsv")
        if cached:
            rows_by_backend["paddleocr_vl"] = cached
            stats["paddleocr_vl"] = {"status": "cached_existing", "runtime": "", "note": "loaded_existing_rows_tsv"}
    if "qwen_vl" not in rows_by_backend:
        cached = read_rows(args.out_dir / f"page_{page:03d}_qwen_vl_rows.tsv")
        if cached:
            rows_by_backend["qwen_vl"] = cached
            stats["qwen_vl"] = {"status": "cached_existing", "runtime": "", "note": "loaded_existing_rows_tsv"}

    report = args.out_dir / "vlm_comparison_report.md"
    write_report(report, stats, rows_by_backend)
    print(f"Wrote VLM comparison report: {report}")


if __name__ == "__main__":
    main()
