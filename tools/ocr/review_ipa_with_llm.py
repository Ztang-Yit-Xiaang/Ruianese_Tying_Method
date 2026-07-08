#!/usr/bin/env python3
"""IPA-aware review layer for Jie Yong Ki clean rows.

This script does not promote model suggestions into the dictionary or override
table. It prepares strict row prompts, validates JSON reviewer output, and
clusters conservative review candidates for human confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_clean_vlm_rows


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "output" / "jie_yong_ki" / "vlm_backend_trials"
REVIEW_MARKERS = {
    "ipa_artifact_theta",
    "code_not_in_scheme",
    "ruian_code_unresolved",
    "tone_glyph_unclear",
    "tone_glyph_from_raw_marker",
    "tone_category_conflict",
}
KNOWN_OCR_CONFUSIONS = [
    r"\theta may be an OCR artifact for ny/y/ɥ-like glyphs; do not map it automatically.",
    r"\alpha may be an OCR artifact for ɔ/o in some IPA cells; require image or override evidence.",
    r"\epsilon and ɛ should be preserved as eh at the IPA-value layer; the formal Rime code may simplify eh to e.",
    r"\eta may represent ŋ/ng.",
    "Tone glyphs may be misread as 2, ², ₂, _3, ^2, or ). Tone must come from tone_from_glyph.",
    "Empty hanzi cells may inherit the previous hanzi and should not by itself imply a phonetic issue.",
]
CANDIDATE_FIELDS = [
    "page",
    "row",
    "hanzi",
    "raw_ipa",
    "current_initial",
    "current_final",
    "tone_from_glyph",
    "mandarin_pinyin",
    "rime_code",
    "llm_status",
    "ipa_analysis",
    "issue",
    "suggested_clean_ipa",
    "suggested_initial",
    "suggested_final",
    "tone",
    "confidence",
    "should_add_override",
    "override_candidate",
    "validation_status",
    "validation_note",
]
CLUSTER_FIELDS = [
    "issue",
    "raw_ipa",
    "suggested_initial",
    "suggested_final",
    "tone",
    "suggested_code",
    "count",
    "rows",
    "override_candidates",
]
REVIEW_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ok", "suspicious", "needs_human_check"]},
        "ipa_analysis": {"type": "string"},
        "issue": {"type": "string"},
        "suggested_clean_ipa": {"type": "string"},
        "suggested_initial": {"type": "string"},
        "suggested_final": {"type": "string"},
        "tone": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "should_add_override": {"type": "boolean"},
        "override_candidate": {"type": "string"},
    },
    "required": [
        "status",
        "ipa_analysis",
        "issue",
        "suggested_clean_ipa",
        "suggested_initial",
        "suggested_final",
        "tone",
        "confidence",
        "should_add_override",
        "override_candidate",
    ],
}


PROMPT_TEMPLATE = """You are an IPA-aware LLM assistant specializing in phonetic transcription, IPA symbols, and OCR error review for Sinitic dialect materials.

You are reviewing one OCR/VLM-extracted row from a Rui'anese phonology table.

Your job is NOT to invent pronunciations. Your job is to check whether the OCR-extracted IPA-like text is plausible, based on:
1. IPA symbol knowledge,
2. the provided Rui'anese initial/final inventory,
3. known OCR confusions,
4. the deterministic parser output.

Rules:
1. The IPA/text phonetic field may contain OCR errors.
2. Use IPA expertise to detect suspicious symbols, spacing, segmentation, or diacritics.
3. Tone must come from tone_from_glyph only.
4. Mandarin pinyin is context only; do not use it to infer tone, initial, final, or a correction candidate.
5. Do not invent a pronunciation unsupported by the row or inventory.
6. If the row is plausible, return status = "ok".
7. If suspicious, propose one conservative correction.
8. If uncertain, return status = "needs_human_check".
9. Output JSON only.
10. If raw_ipa contains \\theta, treat it as an OCR artifact unless row evidence strongly supports a known IPA symbol.
11. If suggesting a correction, suggested_initial + suggested_final + tone must exist in the legal Rime syllable set; otherwise return needs_human_check.
12. If raw_ipa contains \\epsilon or ɛ, preserve that value as eh in suggested_clean_ipa; suggested_final may still be e when the formal Rime scheme simplifies eh to e.
13. If the only evidence for a correction is Mandarin pinyin, return needs_human_check.

Known initials:
{known_initials}

Known finals:
{known_finals}

Known IPA-to-Rime mapping:
{ipa_to_rime_mapping}

Known OCR confusions:
{known_ocr_confusions}

Row:
page: {page}
row: {row}
hanzi: {hanzi}
raw_ipa: {raw_ipa}
current_initial: {current_initial}
current_final: {current_final}
tone_from_glyph: {tone_from_glyph}
mandarin_pinyin: {mandarin_pinyin}
rime_code: {rime_code}

Return JSON with this schema:
{{
  "status": "ok | suspicious | needs_human_check",
  "ipa_analysis": "...",
  "issue": "...",
  "suggested_clean_ipa": "...",
  "suggested_initial": "...",
  "suggested_final": "...",
  "tone": "...",
  "confidence": 0.0,
  "should_add_override": true_or_false,
  "override_candidate": "..."
}}
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def legal_inventory(path: Path) -> tuple[list[str], list[str], set[str]]:
    initials: set[str] = set()
    finals: set[str] = set()
    syllables: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            initial = "" if parts[0] == "Ø" else parts[0]
            final = parts[1]
            initials.add(initial)
            finals.add(final)
            syllables.add(initial + final)
            for tone in range(1, 9):
                syllables.add(f"{initial}{final}{tone}")
    return sorted(initials), sorted(finals), syllables


def ipa_mapping_text() -> str:
    replacements = [f"{old} -> {new}" for old, new in build_clean_vlm_rows.IPA_TO_CODE_REPLACEMENTS]
    protected = [
        f"{old} -> {build_clean_vlm_rows.PROTECTED_INITIAL_RESTORE[value]}"
        for old, value in build_clean_vlm_rows.PROTECTED_INITIAL_REPLACEMENTS
    ]
    protected.append("z -> ss")
    latex = [fr"\{key} -> {value}" for key, value in build_clean_vlm_rows.LATEX_SYMBOLS.items()]
    return "; ".join(latex + protected + replacements)


def should_review(row: dict[str, str], include_ok: bool) -> bool:
    if include_ok:
        return True
    if row.get("status") != "ok":
        return True
    notes = set(filter(None, row.get("review_note", "").split(",")))
    if notes & REVIEW_MARKERS:
        return True
    try:
        return bool(row.get("tone_glyph_confidence")) and float(row["tone_glyph_confidence"]) < 0.60
    except ValueError:
        return False


def build_prompt(row: dict[str, str], initials: list[str], finals: list[str]) -> str:
    return PROMPT_TEMPLATE.format(
        known_initials=", ".join(initial or "Ø" for initial in initials),
        known_finals=", ".join(finals),
        ipa_to_rime_mapping=ipa_mapping_text(),
        known_ocr_confusions="; ".join(KNOWN_OCR_CONFUSIONS),
        page=row.get("page", ""),
        row=row.get("row", ""),
        hanzi=row.get("hanzi", ""),
        raw_ipa=row.get("ruian_ipa_raw_vlm", ""),
        current_initial=row.get("ruian_initial", ""),
        current_final=row.get("ruian_final", ""),
        tone_from_glyph=row.get("tone_number", ""),
        mandarin_pinyin=row.get("mandarin_pinyin", ""),
        rime_code=row.get("ruian_code", ""),
    )


def rule_review(row: dict[str, str], legal_syllables: set[str]) -> dict[str, Any]:
    raw = row.get("ruian_ipa_raw_vlm", "")
    tone = row.get("tone_number", "")
    code = row.get("ruian_code", "")
    if r"\theta" in raw or "θ" in raw:
        return {
            "status": "needs_human_check",
            "ipa_analysis": "The row contains theta-like OCR text, which is known to be unreliable for this source.",
            "issue": "ipa_artifact_theta",
            "suggested_clean_ipa": "",
            "suggested_initial": "",
            "suggested_final": "",
            "tone": tone,
            "confidence": 0.55,
            "should_add_override": False,
            "override_candidate": "",
        }
    if not code:
        return {
            "status": "needs_human_check",
            "ipa_analysis": "The deterministic parser did not produce a legal Rime code.",
            "issue": "ruian_code_unresolved",
            "suggested_clean_ipa": "",
            "suggested_initial": "",
            "suggested_final": "",
            "tone": tone,
            "confidence": 0.50,
            "should_add_override": False,
            "override_candidate": "",
        }
    if code not in legal_syllables:
        return {
            "status": "needs_human_check",
            "ipa_analysis": "The current code is not in the legal syllable dictionary.",
            "issue": "code_not_in_scheme",
            "suggested_clean_ipa": "",
            "suggested_initial": "",
            "suggested_final": "",
            "tone": tone,
            "confidence": 0.50,
            "should_add_override": False,
            "override_candidate": "",
        }
    if row.get("tone_glyph_source") == "raw_marker_fallback":
        return {
            "status": "needs_human_check",
            "ipa_analysis": "Tone was derived from a raw OCR marker fallback rather than the image glyph detector.",
            "issue": "tone_glyph_from_raw_marker",
            "suggested_clean_ipa": row.get("ruian_ipa_corrected", ""),
            "suggested_initial": row.get("ruian_initial", ""),
            "suggested_final": row.get("ruian_final", ""),
            "tone": tone,
            "confidence": 0.65,
            "should_add_override": False,
            "override_candidate": "",
        }
    return {
        "status": "ok",
        "ipa_analysis": "The deterministic parser output is legal and no configured suspicious marker is present.",
        "issue": "",
        "suggested_clean_ipa": row.get("ruian_ipa_corrected", ""),
        "suggested_initial": row.get("ruian_initial", ""),
        "suggested_final": row.get("ruian_final", ""),
        "tone": tone,
        "confidence": 0.90,
        "should_add_override": False,
        "override_candidate": "",
    }


def parse_review_json(value: str) -> dict[str, Any]:
    value = value.strip()
    if not value:
        raise ValueError("empty response")
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("review response must be a JSON object")
    return data


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    try:
        data = response.model_dump()
    except Exception:
        data = response
    chunks: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("type") in {"output_text", "text"} and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(data)
    return "\n".join(chunks)


def openai_client(api_key_env: str = "OPENAI_API_KEY", base_url: str | None = None):
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


def call_openai_reviewer(
    prompt: str,
    model: str,
    *,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    use_chat_completions: bool = False,
) -> dict[str, Any]:
    client = openai_client(api_key_env=api_key_env, base_url=base_url)

    if use_chat_completions:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return parse_review_json(content)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ipa_review",
                "schema": REVIEW_JSON_SCHEMA,
                "strict": True,
            }
        },
    )
    return parse_review_json(response_text(response))


def api_error_review(row: dict[str, str], exc: Exception) -> dict[str, Any]:
    return {
        "status": "needs_human_check",
        "ipa_analysis": f"OpenAI API review failed: {type(exc).__name__}",
        "issue": "api_error",
        "suggested_clean_ipa": "",
        "suggested_initial": "",
        "suggested_final": "",
        "tone": row.get("tone_number", ""),
        "confidence": 0.0,
        "should_add_override": False,
        "override_candidate": "",
    }


def validate_review(review: dict[str, Any], row: dict[str, str], legal_syllables: set[str]) -> tuple[str, str]:
    status = normalize_cell(review.get("status"))
    if status not in {"ok", "suspicious", "needs_human_check"}:
        return "invalid", "invalid_status"
    tone = normalize_cell(review.get("tone"))
    tone_from_glyph = row.get("tone_number", "")
    if tone != tone_from_glyph:
        return "invalid", "tone_mismatch"
    analysis = normalize_cell(review.get("ipa_analysis")).lower()
    issue = normalize_cell(review.get("issue")).lower()
    mandarin_pinyin = normalize_cell(row.get("mandarin_pinyin")).lower()
    if mandarin_pinyin and mandarin_pinyin in analysis:
        return "invalid", "mandarin_pinyin_used_as_evidence"
    if re.search(r"\b(mandarin|pinyin|普通话|普通話)\b", analysis) and re.search(
        r"\b(infer|derive|correspond|typical|unlikely|because|based)\b|推|推出|对应|對應",
        analysis,
    ):
        return "invalid", "mandarin_pinyin_used_as_evidence"
    if "mandarin" in issue or "pinyin" in issue:
        return "invalid", "mandarin_pinyin_used_as_evidence"
    raw_ipa = row.get("ruian_ipa_raw_vlm", "")
    if (r"\theta" in raw_ipa or "θ" in raw_ipa) and status == "suspicious":
        return "invalid", "theta_requires_image_or_override"
    suggested_clean_ipa = normalize_cell(review.get("suggested_clean_ipa"))
    if (r"\epsilon" in raw_ipa or "ɛ" in raw_ipa) and suggested_clean_ipa:
        if "eh" not in suggested_clean_ipa and "ɛ" not in suggested_clean_ipa:
            return "invalid", "epsilon_not_preserved_as_eh"
    suggested_code = normalize_cell(review.get("suggested_initial")) + normalize_cell(review.get("suggested_final")) + tone
    if status == "suspicious" and suggested_code not in legal_syllables:
        return "invalid", "suggested_code_not_in_scheme"
    return "ok", ""


def flatten_candidate(row: dict[str, str], review: dict[str, Any], legal_syllables: set[str]) -> dict[str, str]:
    validation_status, validation_note = validate_review(review, row, legal_syllables)
    return {
        "page": row.get("page", ""),
        "row": row.get("row", ""),
        "hanzi": row.get("hanzi", ""),
        "raw_ipa": row.get("ruian_ipa_raw_vlm", ""),
        "current_initial": row.get("ruian_initial", ""),
        "current_final": row.get("ruian_final", ""),
        "tone_from_glyph": row.get("tone_number", ""),
        "mandarin_pinyin": row.get("mandarin_pinyin", ""),
        "rime_code": row.get("ruian_code", ""),
        "llm_status": normalize_cell(review.get("status")),
        "ipa_analysis": normalize_cell(review.get("ipa_analysis")),
        "issue": normalize_cell(review.get("issue")),
        "suggested_clean_ipa": normalize_cell(review.get("suggested_clean_ipa")),
        "suggested_initial": normalize_cell(review.get("suggested_initial")),
        "suggested_final": normalize_cell(review.get("suggested_final")),
        "tone": normalize_cell(review.get("tone")),
        "confidence": str(review.get("confidence", "")),
        "should_add_override": str(bool(review.get("should_add_override"))).lower(),
        "override_candidate": normalize_cell(review.get("override_candidate")),
        "validation_status": validation_status,
        "validation_note": validation_note,
    }


def cluster_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        if row["validation_status"] != "ok" or row["llm_status"] == "ok":
            continue
        key = (row["issue"], row["raw_ipa"], row["suggested_initial"], row["suggested_final"], row["tone"])
        groups[key].append(row)
    clustered = []
    for (issue, raw_ipa, initial, final, tone), rows in groups.items():
        suggested_code = f"{initial}{final}{tone}" if initial or final else ""
        override_candidates = [row["override_candidate"] for row in rows if row["override_candidate"]]
        clustered.append(
            {
                "issue": issue,
                "raw_ipa": raw_ipa,
                "suggested_initial": initial,
                "suggested_final": final,
                "tone": tone,
                "suggested_code": suggested_code,
                "count": str(len(rows)),
                "rows": ",".join(f"{row['page']}:{row['row']}:{row['hanzi']}" for row in rows),
                "override_candidates": " || ".join(dict.fromkeys(override_candidates)),
            }
        )
    return sorted(clustered, key=lambda row: (-int(row["count"]), row["issue"], row["raw_ipa"]))


def load_external_reviews(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    reviews = {}
    if not path.exists():
        return reviews
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            page = normalize_cell(item.get("page"))
            row = normalize_cell(item.get("row"))
            review = item.get("review", item)
            if page and row and isinstance(review, dict):
                reviews[(page, row)] = review
    return reviews


def run_api_smoke_test(
    model: str,
    *,
    provider: str = "openai",
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str | None = None,
    use_chat_completions: bool = False,
) -> None:
    prompt = """Return JSON only with this exact schema. This is a connectivity test.
{
  "status": "ok",
  "ipa_analysis": "API smoke test succeeded.",
  "issue": "",
  "suggested_clean_ipa": "",
  "suggested_initial": "",
  "suggested_final": "",
  "tone": "1",
  "confidence": 1.0,
  "should_add_override": false,
  "override_candidate": ""
}
"""
    review = call_openai_reviewer(
        prompt,
        model,
        api_key_env=api_key_env,
        base_url=base_url,
        use_chat_completions=use_chat_completions,
    )
    if normalize_cell(review.get("tone")) != "1":
        raise SystemExit("API smoke test returned invalid tone.")
    print(
        f"API smoke test ok: provider={provider}, "
        f"model={model}, status={normalize_cell(review.get('status'))}"
    )

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-rows", type=Path, default=DEFAULT_OUT_DIR / "page_059_clean_rows.tsv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--legal-pairs", type=Path, default=ROOT / "ruian_legal_pairs.tsv")
    parser.add_argument("--ruian-pinyin-dict", type=Path, default=ROOT / "ruian_pinyin.dict.yaml")
    parser.add_argument("--include-ok", action="store_true")
    parser.add_argument("--external-reviews-jsonl", type=Path)
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--use-openai-compatible", action="store_true")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit rows reviewed. With API review, 0 defaults to 1 for safety.",
    )
    parser.add_argument("--api-smoke-test", action="store_true")
    args = parser.parse_args(argv)
    use_api = args.use_openai or args.use_openai_compatible
    if args.api_smoke_test:
        if not use_api:
            raise SystemExit("--api-smoke-test requires --use-openai or --use-openai-compatible")
        try:
            run_api_smoke_test(
                args.model,
                provider=args.provider,
                api_key_env=args.api_key_env,
                base_url=args.base_url,
                use_chat_completions=args.use_openai_compatible,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        return

    if use_api:
        try:
            openai_client(api_key_env=args.api_key_env, base_url=args.base_url)
        except RuntimeError as exc:
            raise SystemExit(str(exc))

    initials, finals, legal_syllables = legal_inventory(args.legal_pairs)
    valid_dict_syllables = build_clean_vlm_rows.load_valid_syllables(args.ruian_pinyin_dict)
    legal_syllables |= valid_dict_syllables
    rows = [row for row in read_tsv(args.clean_rows) if should_review(row, args.include_ok)]
    if use_api:
        limit = args.limit or 1
        rows = rows[:limit]
    elif args.limit:
        rows = rows[: args.limit]
    prompts = [
        {
            "page": row.get("page", ""),
            "row": row.get("row", ""),
            "hanzi": row.get("hanzi", ""),
            "prompt": build_prompt(row, initials, finals),
        }
        for row in rows
    ]
    write_jsonl(args.out_dir / "llm_ipa_review_prompts.jsonl", prompts)

    external_reviews = load_external_reviews(args.external_reviews_jsonl) if args.external_reviews_jsonl else {}
    candidates: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("page", ""), row.get("row", ""))
        if key in external_reviews:
            review = external_reviews[key]
        elif use_api:
            try:
                review = call_openai_reviewer(
                    build_prompt(row, initials, finals),
                    args.model,
                    api_key_env=args.api_key_env,
                    base_url=args.base_url,
                    use_chat_completions=args.use_openai_compatible,
                )
            except Exception as exc:
                review = api_error_review(row, exc)
        else:
            review = rule_review(row, legal_syllables)
        candidates.append(flatten_candidate(row, review, legal_syllables))

    clusters = cluster_candidates(candidates)
    write_tsv(args.out_dir / "llm_ipa_review_candidates.tsv", candidates, CANDIDATE_FIELDS)
    write_tsv(args.out_dir / "llm_ipa_review_clusters.tsv", clusters, CLUSTER_FIELDS)

    status_counts = Counter(row["llm_status"] for row in candidates)
    print(f"Wrote prompts: {args.out_dir / 'llm_ipa_review_prompts.jsonl'}")
    print(f"Wrote candidates: {args.out_dir / 'llm_ipa_review_candidates.tsv'}")
    print(f"Wrote clusters: {args.out_dir / 'llm_ipa_review_clusters.tsv'}")
    print("Review statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))


if __name__ == "__main__":
    main()
