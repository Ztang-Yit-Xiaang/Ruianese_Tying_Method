from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from .decoding import decode_ipa_pair, topk_probabilities
from .hashing import mapping_hash, sha256_file
from .inventory import Inventory
from .io_utils import read_jsonl, write_jsonl
from .modeling import load_checkpoint
from .visual_labels import roman_from_ipa


@torch.no_grad()
def predict_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    inventory: Inventory,
    image_key: str = "ipa_clean",
    image_size: int = 160,
    confidence_threshold: float = 0.85,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    schema_path: Path | None = None,
    legal_pairs_path: Path | None = None,
    allow_mapping_mismatch: bool = False,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    model, metadata = load_checkpoint(checkpoint_path, map_location=device)
    mismatches = _check_checkpoint_hashes(
        metadata,
        schema_path=schema_path,
        legal_pairs_path=legal_pairs_path,
        allow_mapping_mismatch=allow_mapping_mismatch,
    )
    model.to(device)
    model.eval()
    rows = [row for row in read_jsonl(manifest_path) if row.get("paths", {}).get(image_key)]
    out: list[dict[str, Any]] = []
    for row in rows:
        image = _load_image(_resolve_path(manifest_path, row["paths"][image_key]), image_size).unsqueeze(0).to(device)
        logits = model(image)
        probs = {key: torch.softmax(value, dim=1)[0].detach().cpu().numpy() for key, value in logits.items()}
        tone_idx = int(np.argmax(probs["tone"]))
        tone = model.classes.tones[tone_idx]
        class_space = _checkpoint_class_space(metadata)
        raw_topk_tone = topk_probabilities(probs["tone"], model.classes.tones, top_k)
        decoding: dict[str, Any] = {}
        if class_space == "ipa":
            allowed_pairs = _checkpoint_ipa_pairs(metadata, model, inventory)
            decoding = decode_ipa_pair(
                probs["initial"],
                probs["final"],
                model.classes.initials,
                model.classes.finals,
                allowed_pairs,
                top_k=top_k,
            )
            ipa_initial = decoding["predicted_ipa_initial"]
            ipa_final = decoding["predicted_ipa_final"]
            mapped = roman_from_ipa(ipa_initial, ipa_final)
            if mapped is None:
                rime_initial = ""
                rime_final = ""
                romanization = ""
            else:
                rime_initial, rime_final = mapped
                romanization = inventory.compose(rime_initial, rime_final, tone)
        else:
            initial_idx = int(np.argmax(probs["initial"]))
            final_idx = int(np.argmax(probs["final"]))
            rime_initial = model.classes.initials[initial_idx]
            rime_final = model.classes.finals[final_idx]
            romanization = inventory.compose(rime_initial, rime_final, tone)
            ipa_initial = ""
            ipa_final = ""
            decoding = {
                "raw_topk_initial": topk_probabilities(probs["initial"], model.classes.initials, top_k),
                "raw_topk_final": topk_probabilities(probs["final"], model.classes.finals, top_k),
                "raw_ipa_initial": "",
                "raw_ipa_final": "",
                "raw_pair_valid": False,
                "constraint_changed_prediction": False,
                "constrained_topk_pairs": [],
                "pair_score": None,
            }
        initial_probability = float(probs["initial"][model.classes.initials.index(ipa_initial if class_space == "ipa" else rime_initial)])
        final_probability = float(probs["final"][model.classes.finals.index(ipa_final if class_space == "ipa" else rime_final)])
        confidence = float(min(initial_probability, final_probability, probs["tone"][tone_idx]))
        valid = inventory.is_valid_romanization(romanization)
        out.append(
            {
                "cell_id": row["id"],
                "page_no": row.get("page_no"),
                "row_index": row.get("row_index"),
                "predicted_class_space": class_space,
                "raw_ipa_initial": decoding["raw_ipa_initial"],
                "raw_ipa_final": decoding["raw_ipa_final"],
                "raw_pair_valid": decoding["raw_pair_valid"],
                "predicted_ipa_initial": ipa_initial,
                "predicted_ipa_final": ipa_final,
                "constraint_changed_prediction": decoding["constraint_changed_prediction"],
                "raw_topk_initial": decoding["raw_topk_initial"],
                "raw_topk_final": decoding["raw_topk_final"],
                "raw_topk_tone": raw_topk_tone,
                "constrained_topk_pairs": decoding["constrained_topk_pairs"],
                "pair_score": decoding["pair_score"],
                "ipa_initial": ipa_initial,
                "ipa_final": ipa_final,
                "mapped_rime_initial": rime_initial,
                "mapped_rime_final": rime_final,
                "mapped_rime_syllable": romanization,
                "rime_initial": rime_initial,
                "rime_final": rime_final,
                "rime_syllable": romanization,
                "initial": rime_initial,
                "final": rime_final,
                "tone": tone,
                "romanization": romanization,
                "confidence": round(confidence, 6),
                "valid_romanization": valid,
                "ipa_label_source": "model" if class_space == "ipa" else "unknown",
                "rime_label_source": "derived" if class_space == "ipa" else "model",
                "tone_label_source": "model",
                "needs_review": (not valid) or confidence < confidence_threshold,
                "mapping_mismatch": bool(mismatches),
                "mapping_mismatch_details": mismatches,
                "model_checkpoint": str(checkpoint_path),
                "model_metadata": metadata,
            }
        )
    write_jsonl(output_path, out)
    return out


def _checkpoint_class_space(metadata: dict[str, Any]) -> str:
    if metadata.get("class_space") in {"ipa", "rime"}:
        return str(metadata["class_space"])
    config = metadata.get("config", {})
    if isinstance(config, dict) and config.get("class_space") in {"ipa", "rime"}:
        return str(config["class_space"])
    return "rime"


def _checkpoint_ipa_pairs(
    metadata: dict[str, Any],
    model: Any,
    inventory: Inventory,
) -> set[tuple[str, str]]:
    pairs = {
        (str(value[0]), str(value[1]))
        for value in metadata.get("ipa_pair_vocabulary", [])
        if isinstance(value, (list, tuple)) and len(value) == 2
    }
    if pairs:
        return pairs
    for initial in model.classes.initials:
        for final in model.classes.finals:
            mapped = roman_from_ipa(initial, final)
            if mapped and (inventory.legal_pairs is None or mapped in inventory.legal_pairs):
                pairs.add((initial, final))
    return pairs


def _check_checkpoint_hashes(
    metadata: dict[str, Any],
    *,
    schema_path: Path | None,
    legal_pairs_path: Path | None,
    allow_mapping_mismatch: bool,
) -> list[str]:
    current = {
        "mapping_hash": mapping_hash(),
        "schema_hash": sha256_file(schema_path),
        "legal_pairs_hash": sha256_file(legal_pairs_path),
    }
    mismatches: list[str] = []
    for key, current_value in current.items():
        checkpoint_value = metadata.get(key)
        if checkpoint_value is None:
            mismatches.append(f"{key}:missing_in_checkpoint")
        elif current_value is not None and checkpoint_value != current_value:
            mismatches.append(f"{key}:checkpoint={checkpoint_value};current={current_value}")
    if mismatches and not allow_mapping_mismatch:
        raise RuntimeError(
            "Checkpoint mapping/schema metadata does not match the current pipeline: "
            + ", ".join(mismatches)
            + ". Re-train or pass --allow-mapping-mismatch for an explicitly marked legacy run."
        )
    return mismatches


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    canvas = Image.new("L", (image_size, image_size), 255)
    img.thumbnail((image_size - 12, image_size - 12), Image.Resampling.LANCZOS)
    x = (image_size - img.width) // 2
    y = (image_size - img.height) // 2
    canvas.paste(img, (x, y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.stack([arr, arr, arr], axis=0)
    return torch.from_numpy(arr)


def _resolve_path(anchor: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for candidate in (anchor.parent / path, anchor.parent.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return anchor.parent / path
