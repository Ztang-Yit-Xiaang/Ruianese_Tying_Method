from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

from .inventory import Inventory
from .io_utils import append_jsonl, read_jsonl
from .label_semantics import canonicalize_authoritative_label
from .visual_labels import ipa_final_inventory, ipa_initial_inventory, normalize_ipa_final, normalize_ipa_initial


API_LABEL_STATUSES = (
    "labeled",
    "uncertain",
    "mixed_cluster",
    "needs_split",
    "unreadable",
    "insufficient_evidence",
)


LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": list(API_LABEL_STATUSES)},
        "ipa_initial": {"type": ["string", "null"]},
        "ipa_final": {"type": ["string", "null"]},
        "tone": {"type": ["integer", "null"], "minimum": 1, "maximum": 8},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": [
        "status",
        "ipa_initial",
        "ipa_final",
        "tone",
        "confidence",
        "notes",
    ],
}


@dataclass(frozen=True)
class ApiCredential:
    value: str
    source: str


def label_clusters_with_openai(
    cluster_manifest_path: Path,
    output_path: Path,
    inventory: Inventory,
    model: str,
    limit: int | None = None,
    use_api: bool = False,
    api_key_file: Path | None = None,
    base_url: str | None = None,
    allow_custom_endpoint: bool = False,
    allow_official_key_to_custom_endpoint: bool = False,
) -> list[dict[str, Any]]:
    clusters = read_jsonl(cluster_manifest_path)
    already = {row.get("cluster_id") for row in read_jsonl(output_path)}
    todo = [row for row in clusters if row.get("cluster_id") not in already and row.get("cluster_id") != "noise"]
    if limit is not None:
        todo = todo[:limit]
    if not use_api:
        templates = [_template_label(row) for row in todo]
        append_jsonl(output_path, templates)
        return templates

    resolved_base_url = load_base_url(base_url)
    custom_endpoint = is_custom_endpoint(resolved_base_url)
    custom_allowed = allow_custom_endpoint or _truthy_env("ALLOW_CUSTOM_OPENAI_ENDPOINT")
    if custom_endpoint and not custom_allowed:
        host = endpoint_host(resolved_base_url)
        raise RuntimeError(
            f"Custom API endpoint {host!r} is blocked. Pass --allow-custom-endpoint "
            "or set ALLOW_CUSTOM_OPENAI_ENDPOINT=1 after verifying the host."
        )
    credential = load_api_key_with_source(
        api_key_file,
        custom_endpoint=custom_endpoint,
        allow_official_key_to_custom_endpoint=(
            allow_official_key_to_custom_endpoint or _truthy_env("ALLOW_OPENAI_KEY_TO_CUSTOM_ENDPOINT")
        ),
    )
    client_kwargs: dict[str, str] = {"api_key": credential.value}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    print(f"Endpoint host: {endpoint_host(resolved_base_url)}")
    print(f"Model: {model}")
    print(f"Number of clusters: {len(todo)}")
    print(f"Number of images: {len(todo)}")
    print(f"API key source: {credential.source}")
    print(f"Custom endpoint allowed: {custom_endpoint and custom_allowed}")
    client = OpenAI(**client_kwargs)
    labels: list[dict[str, Any]] = []
    for row in todo:
        sheet_path = _resolve_path(cluster_manifest_path, row["contact_sheet"])
        prompt = build_prompt(row, inventory)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _image_data_url(sheet_path)},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ruian_ipa_cluster_label",
                    "schema": LABEL_SCHEMA,
                    "strict": True,
                }
            },
        )
        payload = json.loads(response.output_text)
        payload = _normalize_payload(payload, inventory)
        payload.update(
            {
                "cluster_id": row["cluster_id"],
                "label_status": "weak",
                "source": "openai_api",
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "contact_sheet": row["contact_sheet"],
                "representative_cell_ids": row.get("representative_cell_ids", []),
                "needs_review": payload["status"] != "labeled" or not payload.get("romanization"),
            }
        )
        labels.append(payload)
        append_jsonl(output_path, [payload])
    return labels


def load_api_key(api_key_file: Path | None = None) -> str:
    return load_api_key_with_source(api_key_file).value


def load_api_key_with_source(
    api_key_file: Path | None = None,
    *,
    custom_endpoint: bool = False,
    allow_official_key_to_custom_endpoint: bool = False,
) -> ApiCredential:
    if custom_endpoint:
        custom_key = os.environ.get("CUSTOM_OPENAI_API_KEY")
        if custom_key:
            return ApiCredential(custom_key.strip(), "CUSTOM_OPENAI_API_KEY")
    else:
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return ApiCredential(env_key.strip(), "OPENAI_API_KEY")

    candidates = _api_key_candidates(api_key_file or Path("API_KEY.txt"))
    for path in candidates:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return ApiCredential(key, path.name)
    env_key = os.environ.get("OPENAI_API_KEY")
    if custom_endpoint and env_key and allow_official_key_to_custom_endpoint:
        return ApiCredential(env_key.strip(), "OPENAI_API_KEY (explicitly allowed)")
    if custom_endpoint:
        raise RuntimeError(
            "No custom-endpoint credential found. Set CUSTOM_OPENAI_API_KEY, provide API_KEY.txt, "
            "or explicitly allow OPENAI_API_KEY with --allow-official-key-to-custom-endpoint."
        )
    raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or create API_KEY.txt.")


def load_base_url(base_url: str | None = None, base_url_file: Path | None = None) -> str | None:
    if base_url:
        return _validate_base_url(base_url)
    env_url = os.environ.get("OPENAI_BASE_URL")
    if env_url:
        return _validate_base_url(env_url)
    candidates = _api_key_candidates(base_url_file or Path("API_BASE_URL.txt"))
    for path in candidates:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return _validate_base_url(value)
    return None


def endpoint_host(base_url: str | None) -> str:
    return (urlparse(base_url).hostname or "api.openai.com") if base_url else "api.openai.com"


def is_custom_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    return host != "api.openai.com"


def _api_key_candidates(api_key_file: Path) -> list[Path]:
    if api_key_file.is_absolute():
        return [api_key_file]
    candidates: list[Path] = []
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        candidate = parent / api_key_file
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def build_prompt(cluster_row: dict[str, Any], inventory: Inventory) -> str:
    initials = ", ".join(["(zero initial)", *inventory.initials])
    finals = ", ".join(inventory.finals)
    ipa_initials = ", ".join(["(zero initial)", *ipa_initial_inventory()])
    ipa_finals = ", ".join(ipa_final_inventory())
    tones = ", ".join(str(t) for t in inventory.tones)
    return (
        "You are labeling printed IPA cells from a Rui'an dialect dictionary. "
        "This is not open OCR. Identify the syllable class and return only JSON.\n\n"
        f"Cluster: {cluster_row.get('cluster_id')} with {cluster_row.get('size')} similar crops.\n"
        "Return the authoritative visual IPA-layer labels only. Rime fields are derived by code. "
        "If two IPA symbols collapse to the same Rime romanization, still distinguish them.\n\n"
        "Important final mapping: IPA əʉ is romanized as final ou; IPA iəʉ is romanized as final iou. "
        "IPA ɛ is eh; IPA ɔ is oe; IPA æ is ae; IPA ʉ and y̟u both romanize as yu but must stay "
        "distinct in ipa_final when visible. Return ASCII romanization finals only in final.\n\n"
        "Important initial mapping: IPA z̠ or ʑ is romanized as initial zs; IPA dz is zz; IPA dʑ is jj.\n\n"
        "IPA ts is z; IPA tsʰ is c; IPA z is ss. Return ASCII romanization initials only in initial.\n\n"
        f"Allowed IPA initials: {ipa_initials}\n"
        f"Allowed IPA finals: {ipa_finals}\n"
        f"Derived Rime initials (reference only): {initials}\n"
        f"Derived Rime finals (reference only): {finals}\n"
        f"Allowed tones: {tones}\n\n"
        "Compare medoid/core with diverse and boundary samples. Return fields: status, ipa_initial, "
        "ipa_final, tone, confidence, notes. For mixed_cluster, needs_split, unreadable, or "
        "insufficient_evidence, return null IPA/tone fields instead of forcing a label."
    )


def _normalize_payload(payload: dict[str, Any], inventory: Inventory) -> dict[str, Any]:
    normalized = dict(payload)
    status = str(normalized.get("status", "uncertain"))
    if status not in API_LABEL_STATUSES:
        status = "uncertain"
    normalized["status"] = status
    normalized["ipa_initial"] = normalize_ipa_initial(normalized.get("ipa_initial"))
    normalized["ipa_final"] = normalize_ipa_final(normalized.get("ipa_final"))
    tone = normalized.get("tone")
    normalized["tone"] = str(tone) if tone is not None else ""
    normalized["ipa_label_source"] = "api"
    normalized["tone_label_source"] = "api"
    normalized["rime_label_source"] = "derived"
    if status != "labeled":
        normalized["ipa_initial"] = ""
        normalized["ipa_final"] = ""
        normalized["tone"] = ""
        normalized.update({"rime_initial": "", "rime_final": "", "rime_syllable": "", "initial": "", "final": "", "romanization": ""})
        return normalized
    result = canonicalize_authoritative_label(
        normalized,
        inventory,
        default_ipa_source="api",
        default_tone_source="api",
    )
    if result.label is None:
        normalized["status"] = "uncertain"
        normalized["normalization_error"] = result.reject_reason
        normalized.update({"rime_initial": "", "rime_final": "", "rime_syllable": "", "initial": "", "final": "", "romanization": ""})
        return normalized
    return result.label


def _template_label(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": row["cluster_id"],
        "status": "insufficient_evidence",
        "ipa_initial": "",
        "ipa_final": "",
        "initial": "",
        "final": "",
        "tone": 0,
        "rime_initial": "",
        "rime_final": "",
        "rime_syllable": "",
        "romanization": "",
        "confidence": 0.0,
        "notes": "Template only. Re-run with --use-api or fill manually, then set reviewed/gold.",
        "label_status": "weak",
        "source": "template",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contact_sheet": row.get("contact_sheet"),
        "representative_cell_ids": row.get("representative_cell_ids", []),
        "needs_review": True,
        "ipa_label_source": "unknown",
        "rime_label_source": "unknown",
        "tone_label_source": "unknown",
    }


def _validate_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API base URL must be an absolute HTTP or HTTPS URL")
    return normalized


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _resolve_path(anchor: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (anchor.parent / path, anchor.parent.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return anchor.parent / path
