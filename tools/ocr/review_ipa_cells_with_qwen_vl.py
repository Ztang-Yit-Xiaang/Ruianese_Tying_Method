#!/usr/bin/env python3
"""Review Rui'an IPA table cells with a Qwen-VL compatible API.

This is an advisory image reviewer. It crops the printed IPA cell, sends only
that crop to Qwen-VL, and writes candidate readings for human confirmation.
It never updates ipa_overrides.tsv or any formal Rime dictionary.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

import build_clean_vlm_rows
import ocr_jie_yong_ki_book_table as table_ocr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "vlm_backend_trials"
OUT_FIELDS = [
    "page",
    "row",
    "hanzi",
    "cell_image",
    "raw_ipa_vlm",
    "tone_from_glyph",
    "qwen_ipa_body",
    "qwen_tone_glyph",
    "qwen_tone_register",
    "qwen_confidence",
    "status",
    "note",
    "suggested_initial",
    "suggested_final",
    "suggested_code",
    "override_candidate",
]


PROMPT = """You are reading one cropped cell from the Rui'an dialect dictionary.

The image contains only the "Rui'an phonetic / IPA" cell, not Mandarin pinyin.

Task:
1. Read the printed Rui'an IPA body exactly as a phonetic string.
2. Identify the printed tone glyph only if it is visible in this crop.
3. Do not use Mandarin pronunciation, Mandarin pinyin, character knowledge, or outside lexical guessing.
4. If the IPA body is unclear, return status "needs_human_check".
5. If you see a theta-like OCR artifact in prior text, ignore that prior OCR and read the image directly.
6. Preserve epsilon / ɛ-like vowel quality as "eh" in ipa_body. Do not simplify it to plain "e" in ipa_body.
7. Output JSON only.

Tone glyph labels:
- left_lower, left_upper, right_upper, right_lower, unknown
- register: yin, yang, unknown

Return this JSON shape:
{
  "status": "ok | suspicious | needs_human_check",
  "ipa_body": "...",
  "tone_glyph_position": "left_lower | left_upper | right_upper | right_lower | unknown",
  "tone_register": "yin | yang | unknown",
  "confidence": 0.0,
  "note": "..."
}
"""


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_rows(value: str) -> set[str]:
    rows = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        rows.add(str(int(part)))
    return rows


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_json_response(value: str) -> dict[str, Any]:
    value = value.strip()
    if not value:
        raise ValueError("empty model response")
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model response must be a JSON object")
    return data


def openai_client(api_key_env: str, base_url: str | None):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is not set. In PowerShell, run: "
            f'$env:{api_key_env}="..."'
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI Python SDK is not installed. Run: python -m pip install openai") from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_qwen_vl(client: Any, model: str, image_path: Path, timeout: float) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url(image_path)}},
                ],
            }
        ],
        temperature=0,
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_response(content)
    parsed["_raw_content"] = content
    return parsed


def find_image(page: int, image_dir: Path, start_after: int) -> Path:
    for number, path in table_ocr.collect_images(image_dir, start_after):
        if number == page:
            return path
    raise SystemExit(f"Page {page:03d} was not found in {image_dir}")


def crop_ipa_cells(
    page: int,
    image_path: Path,
    target_rows: set[str],
    out_dir: Path,
    min_row_fraction: float,
    min_col_fraction: float,
    crop_pad: int,
) -> dict[str, Path]:
    image = Image.open(image_path).convert("RGB")
    boxes_by_row, _ = table_ocr.detect_cell_boxes(image, min_row_fraction, min_col_fraction)
    crop_dir = out_dir / "qwen_vl_cell_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops: dict[str, Path] = {}
    for row in sorted(target_rows, key=int):
        grid_row_index = int(row)
        if grid_row_index >= len(boxes_by_row) or len(boxes_by_row[grid_row_index]) < 8:
            continue
        box = boxes_by_row[grid_row_index][7].padded(image.width, image.height, crop_pad)
        crop = image.crop((box.left, box.top, box.right, box.bottom))
        crop_path = crop_dir / f"page_{page:03d}_r{int(row):03d}_ipa.png"
        crop.save(crop_path)
        crops[row] = crop_path
    return crops


def validate_review(data: dict[str, Any]) -> tuple[str, str]:
    status = normalize_cell(data.get("status"))
    if status not in {"ok", "suspicious", "needs_human_check"}:
        return "needs_human_check", "invalid_status_from_model"
    confidence = data.get("confidence", 0.0)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return "needs_human_check", "invalid_confidence_from_model"
    if confidence_value < 0.70 and status == "ok":
        return "suspicious", "low_confidence"
    return status, ""


def suggest_code(ipa_body: str, tone: str, valid_syllables: set[str]) -> tuple[str, str, str]:
    cleaned = build_clean_vlm_rows.clean_ipa_body(ipa_body)
    code_body, note = build_clean_vlm_rows.ipa_body_to_code_body(cleaned)
    if not code_body or not tone:
        return "", "", note
    code = f"{code_body}{tone}"
    if code not in valid_syllables:
        return "", "", f"{note},code_not_in_scheme".strip(",")
    initial, final = build_clean_vlm_rows.split_initial_final(code_body, valid_syllables)
    return code, f"{initial}\t{final}", note


def build_override_candidate(row: dict[str, str], ipa_body: str, code: str, split: str, note: str) -> str:
    if not code or not split:
        return ""
    initial, final = split.split("\t", 1)
    fields = [
        row.get("page", ""),
        row.get("row", ""),
        row.get("hanzi", ""),
        ipa_body,
        initial,
        final,
        row.get("tone_number", ""),
        code,
        f"Qwen-VL cell candidate; {note}".strip("; "),
    ]
    return "\t".join(fields)


def make_error_row(row: dict[str, str], crop_path: Path | None, message: str) -> dict[str, str]:
    return {
        "page": row.get("page", ""),
        "row": row.get("row", ""),
        "hanzi": row.get("hanzi", ""),
        "cell_image": str(crop_path or ""),
        "raw_ipa_vlm": row.get("ruian_ipa_raw_vlm", ""),
        "tone_from_glyph": row.get("tone_number", ""),
        "qwen_ipa_body": "",
        "qwen_tone_glyph": "",
        "qwen_tone_register": "",
        "qwen_confidence": "0.0",
        "status": "needs_human_check",
        "note": message,
        "suggested_initial": "",
        "suggested_final": "",
        "suggested_code": "",
        "override_candidate": "",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", default="059")
    parser.add_argument("--rows", default="13,14,15")
    parser.add_argument("--clean-rows", type=Path, default=DEFAULT_OUT_DIR / "page_059_clean_rows.tsv")
    parser.add_argument("--image-dir", type=Path, default=table_ocr.DEFAULT_IMAGE_DIR)
    parser.add_argument("--start-after", type=int, default=58)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ruian-pinyin-dict", type=Path, default=ROOT / "ruian_pinyin.dict.yaml")
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--base-url", default="https://dashscope-us.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--min-row-fraction", type=float, default=0.18)
    parser.add_argument("--min-col-fraction", type=float, default=0.24)
    parser.add_argument("--crop-pad", type=int, default=5)
    args = parser.parse_args(argv)

    page = int(args.page)
    target_rows = parse_rows(args.rows)
    clean_rows = {
        row.get("row", ""): row
        for row in read_tsv(args.clean_rows)
        if row.get("row", "") in target_rows
    }
    missing = sorted(target_rows - set(clean_rows), key=int)
    if missing:
        raise SystemExit(f"Rows not found in {args.clean_rows}: {', '.join(missing)}")

    image_path = find_image(page, args.image_dir, args.start_after)
    crops = crop_ipa_cells(
        page,
        image_path,
        target_rows,
        args.out_dir,
        args.min_row_fraction,
        args.min_col_fraction,
        args.crop_pad,
    )
    valid_syllables = build_clean_vlm_rows.load_valid_syllables(args.ruian_pinyin_dict)
    raw_jsonl = args.out_dir / "qwen_vl_cell_reviews_raw.jsonl"
    if raw_jsonl.exists():
        raw_jsonl.unlink()

    try:
        client = openai_client(args.api_key_env, args.base_url)
    except RuntimeError as exc:
        raise SystemExit(str(exc))

    output_rows: list[dict[str, str]] = []
    for row_number in sorted(target_rows, key=int):
        source_row = clean_rows[row_number]
        crop_path = crops.get(row_number)
        if not crop_path:
            output_rows.append(make_error_row(source_row, None, "missing_ipa_cell_crop"))
            continue
        try:
            review = call_qwen_vl(client, args.model, crop_path, args.timeout)
            append_jsonl(
                raw_jsonl,
                {
                    "page": f"{page:03d}",
                    "row": row_number,
                    "hanzi": source_row.get("hanzi", ""),
                    "cell_image": str(crop_path),
                    "review": review,
                },
            )
        except Exception as exc:
            output_rows.append(make_error_row(source_row, crop_path, f"api_error:{type(exc).__name__}"))
            continue

        status, validation_note = validate_review(review)
        ipa_body = normalize_cell(review.get("ipa_body"))
        suggested_code, split, code_note = suggest_code(ipa_body, source_row.get("tone_number", ""), valid_syllables)
        initial, final = ("", "")
        if split:
            initial, final = split.split("\t", 1)
        note = "; ".join(
            item
            for item in [
                normalize_cell(review.get("note")),
                validation_note,
                code_note,
            ]
            if item
        )
        output_rows.append(
            {
                "page": f"{page:03d}",
                "row": row_number,
                "hanzi": source_row.get("hanzi", ""),
                "cell_image": str(crop_path),
                "raw_ipa_vlm": source_row.get("ruian_ipa_raw_vlm", ""),
                "tone_from_glyph": source_row.get("tone_number", ""),
                "qwen_ipa_body": ipa_body,
                "qwen_tone_glyph": normalize_cell(review.get("tone_glyph_position")),
                "qwen_tone_register": normalize_cell(review.get("tone_register")),
                "qwen_confidence": str(review.get("confidence", "")),
                "status": status,
                "note": note,
                "suggested_initial": initial,
                "suggested_final": final,
                "suggested_code": suggested_code,
                "override_candidate": build_override_candidate(source_row, ipa_body, suggested_code, split, note),
            }
        )

    out_path = args.out_dir / "qwen_vl_cell_reviews.tsv"
    write_tsv(out_path, output_rows, OUT_FIELDS)
    print(f"Wrote Qwen-VL cell reviews: {out_path}")
    print(f"Wrote cell crops: {args.out_dir / 'qwen_vl_cell_crops'}")
    print(f"Wrote raw model JSONL: {raw_jsonl}")


if __name__ == "__main__":
    main()
