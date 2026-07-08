from __future__ import annotations

import csv
import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .decoding import decode_ipa_pair
from .hashing import current_git_commit, mapping_hash, sha256_file
from .inventory import Inventory
from .io_utils import read_jsonl, write_jsonl
from .label_semantics import canonicalize_authoritative_label
from .modeling import ClassSets, MultiHeadClassifier, save_checkpoint
from .visual_labels import (
    IPA_FINAL_TO_ROMAN,
    IPA_INITIAL_TO_ROMAN,
    IPA_RIME_MAPPING_VERSION,
    ipa_final_inventory,
    ipa_initial_inventory,
    roman_from_ipa,
)


@dataclass
class TrainConfig:
    arch: str = "resnet18"
    class_space: str = "ipa"
    image_size: int = 160
    batch_size: int = 32
    epochs: int = 12
    lr: float = 1e-3
    val_ratio: float = 0.2
    seed: int = 17
    split_mode: str = "group_cluster"
    allow_small_dataset: bool = False
    split_by_page: bool | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.split_by_page is not None:
            self.split_mode = "group_page" if self.split_by_page else "random_image"


class IpaLabelDataset(Dataset):
    def __init__(
        self,
        labels: list[dict[str, Any]],
        classes: ClassSets,
        anchor: Path,
        image_size: int,
        initial_key: str = "initial",
        final_key: str = "final",
        augment: bool = False,
    ) -> None:
        self.labels = labels
        self.classes = classes
        self.anchor = anchor
        self.image_size = image_size
        self.initial_key = initial_key
        self.final_key = final_key
        self.augment = augment
        self.initial_to_idx = {value: idx for idx, value in enumerate(classes.initials)}
        self.final_to_idx = {value: idx for idx, value in enumerate(classes.finals)}
        self.tone_to_idx = {value: idx for idx, value in enumerate(classes.tones)}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.labels[idx]
        image = self._load_image(row["image"])
        target = {
            "initial": torch.tensor(self.initial_to_idx[str(row[self.initial_key])], dtype=torch.long),
            "final": torch.tensor(self.final_to_idx[str(row[self.final_key])], dtype=torch.long),
            "tone": torch.tensor(self.tone_to_idx[str(row["tone"])], dtype=torch.long),
        }
        return image, target

    def _load_image(self, value: str) -> torch.Tensor:
        path = Path(value)
        if not path.is_absolute():
            path = self.anchor / path
            if not path.exists():
                path = self.anchor.parent / value
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)
        if self.augment:
            angle = random.uniform(-1.4, 1.4)
            img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
        canvas = Image.new("L", (self.image_size, self.image_size), 255)
        img.thumbnail((self.image_size - 12, self.image_size - 12), Image.Resampling.LANCZOS)
        x = (self.image_size - img.width) // 2
        y = (self.image_size - img.height) // 2
        canvas.paste(img, (x, y))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.stack([arr, arr, arr], axis=0)
        return torch.from_numpy(arr)


def train_classifier(
    labels_path: Path,
    output_dir: Path,
    inventory: Inventory,
    config: TrainConfig,
    *,
    schema_path: Path | None = None,
    legal_pairs_path: Path | None = None,
) -> dict[str, Any]:
    source_rows = read_jsonl(labels_path)
    labels, rejected, initial_key, final_key = _prepare_labels_with_rejections(
        source_rows, inventory, config.class_space
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rejected_reports(output_dir, rejected)
    print(f"Total label rows: {len(source_rows)}")
    print(f"Accepted for {config.class_space.upper()} training: {len(labels)}")
    print(f"Rejected: {len(rejected)}")
    for reason, count in sorted(Counter(row["reject_reason"] for row in rejected).items()):
        print(f"- {reason}: {count}")
    minimum = 2 if config.allow_small_dataset else 8
    if len(labels) < minimum:
        raise RuntimeError(
            f"Need at least {minimum} valid labels; use --allow-small-dataset only for an explicit smoke run."
        )
    if len(labels) < 300:
        warnings.warn(
            f"Only {len(labels)} valid labels are available; this run is a baseline, not a quality model.",
            stacklevel=2,
        )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_rows, val_rows, split_report = _split_labels(labels, config.val_ratio, config.seed, config.split_mode)
    classes = ClassSets.from_labels(labels, initial_key=initial_key, final_key=final_key)
    allowed_ipa_pairs = sorted(
        {(str(row["ipa_initial"]), str(row["ipa_final"])) for row in train_rows}
        if config.class_space == "ipa"
        else set()
    )
    coverage = _build_class_coverage(train_rows, inventory, config.class_space, initial_key, final_key)
    _write_coverage_reports(output_dir, coverage)
    unlearnable_prediction_classes = {
        "initials": sorted(set(classes.initials) - {str(row[initial_key]) for row in train_rows}),
        "finals": sorted(set(classes.finals) - {str(row[final_key]) for row in train_rows}),
        "tones": sorted(set(classes.tones) - {str(row["tone"]) for row in train_rows}),
    }
    for kind, values in unlearnable_prediction_classes.items():
        if values:
            warnings.warn(
                f"Prediction {kind} have no training samples and cannot be learned: {values}",
                stacklevel=2,
            )
    train_ds = IpaLabelDataset(
        train_rows,
        classes,
        labels_path.parent,
        config.image_size,
        initial_key=initial_key,
        final_key=final_key,
        augment=True,
    )
    val_ds = IpaLabelDataset(
        val_rows,
        classes,
        labels_path.parent,
        config.image_size,
        initial_key=initial_key,
        final_key=final_key,
        augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device(config.device)
    model = MultiHeadClassifier(config.arch, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_path = output_dir / f"{config.arch}_best.pt"
    metadata_base = {
        "config": config.__dict__,
        "class_space": config.class_space,
        "initial_key": initial_key,
        "final_key": final_key,
        "ipa_initial_vocabulary": classes.initials if config.class_space == "ipa" else [],
        "ipa_final_vocabulary": classes.finals if config.class_space == "ipa" else [],
        "rime_initial_vocabulary": (
            sorted(set(IPA_INITIAL_TO_ROMAN.values())) if config.class_space == "ipa" else classes.initials
        ),
        "rime_final_vocabulary": (
            sorted(set(IPA_FINAL_TO_ROMAN.values())) if config.class_space == "ipa" else classes.finals
        ),
        "tone_vocabulary": classes.tones,
        "ipa_pair_vocabulary": [list(pair) for pair in allowed_ipa_pairs],
        "unlearnable_prediction_classes": unlearnable_prediction_classes,
        "ipa_rime_mapping_version": IPA_RIME_MAPPING_VERSION,
        "mapping_hash": mapping_hash(),
        "schema_hash": sha256_file(schema_path),
        "legal_pairs_hash": sha256_file(legal_pairs_path),
        "label_manifest_hash": sha256_file(labels_path),
        "split_mode": config.split_mode,
        "random_seed": config.seed,
        "training_cluster_ids": split_report["train_cluster_ids"],
        "validation_cluster_ids": split_report["validation_cluster_ids"],
        "split_report": split_report,
        "git_commit": current_git_commit(),
    }

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            logits = model(images)
            loss = sum(criterion(logits[key], targets[key]) for key in logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(images)
        metrics = evaluate(
            model,
            val_loader,
            device,
            allowed_ipa_pairs=allowed_ipa_pairs if config.class_space == "ipa" else None,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(1, len(train_ds))
        history.append(metrics)
        mean_acc = float(np.mean([metrics["initial_acc"], metrics["final_acc"], metrics["tone_acc"]]))
        if mean_acc >= best_metric:
            best_metric = mean_acc
            save_checkpoint(
                best_path,
                model,
                {
                    **metadata_base,
                    "metrics": metrics,
                    "train_label_count": len(train_rows),
                    "val_label_count": len(val_rows),
                },
            )

    write_jsonl(output_dir / "train_history.jsonl", history)
    summary = {
        "best_checkpoint": str(best_path),
        "best_mean_acc": best_metric,
        "class_space": config.class_space,
        "initial_key": initial_key,
        "final_key": final_key,
        "total_label_count": len(source_rows),
        "trainable_label_count": len(labels),
        "rejected_label_count": len(rejected),
        "classes": classes.as_dict(),
        "unlearnable_prediction_classes": unlearnable_prediction_classes,
        "split_report": split_report,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _prepare_labels(
    rows: list[dict[str, Any]],
    inventory: Inventory,
    class_space: str,
) -> tuple[list[dict[str, Any]], str, str]:
    labels, _, initial_key, final_key = _prepare_labels_with_rejections(rows, inventory, class_space)
    return labels, initial_key, final_key


def _prepare_labels_with_rejections(
    rows: list[dict[str, Any]],
    inventory: Inventory,
    class_space: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    labels: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if class_space not in {"ipa", "rime"}:
        raise ValueError(f"Unsupported class_space: {class_space}")
    for row in rows:
        if class_space == "ipa":
            result = canonicalize_authoritative_label(
                row,
                inventory,
                default_ipa_source=str(row.get("ipa_label_source", "unknown")),
                default_tone_source=str(row.get("tone_label_source", "unknown")),
            )
            if result.label is None:
                rejected.append(_rejected_row(row, result.reject_reason or "invalid_ipa_label"))
                continue
            item = result.label
        else:
            romanization = str(row.get("rime_syllable") or row.get("romanization") or "")
            parsed = inventory.parse_romanization(romanization)
            if parsed is None or not inventory.is_valid_romanization(romanization):
                rejected.append(_rejected_row(row, "invalid_rime_label"))
                continue
            item = dict(row)
            item["initial"], item["final"], item["tone"] = parsed
        if not item.get("image"):
            rejected.append(_rejected_row(row, "missing_image"))
            continue
        labels.append(item)
    if class_space == "ipa":
        return labels, rejected, "ipa_initial", "ipa_final"
    return labels, rejected, "initial", "final"


@torch.no_grad()
def evaluate(
    model: MultiHeadClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    allowed_ipa_pairs: Iterable[tuple[str, str]] | None = None,
) -> dict[str, float]:
    model.eval()
    correct = {
        "initial": 0,
        "final": 0,
        "tone": 0,
        "pair": 0,
        "unconstrained_full": 0,
        "constrained_pair": 0,
        "constrained_full": 0,
    }
    total = 0
    pair_set = set(allowed_ipa_pairs or [])
    for images, targets in loader:
        images = images.to(device)
        targets = {key: value.to(device) for key, value in targets.items()}
        logits = model(images)
        probabilities = {
            key: torch.softmax(value, dim=1).detach().cpu().numpy() for key, value in logits.items()
        }
        predictions = {key: value.argmax(dim=1) for key, value in logits.items()}
        total += len(images)
        for key in ("initial", "final", "tone"):
            correct[key] += int((predictions[key] == targets[key]).sum().detach().cpu())
        pair_matches = (predictions["initial"] == targets["initial"]) & (
            predictions["final"] == targets["final"]
        )
        full_matches = pair_matches & (predictions["tone"] == targets["tone"])
        correct["pair"] += int(pair_matches.sum().detach().cpu())
        correct["unconstrained_full"] += int(full_matches.sum().detach().cpu())
        if pair_set:
            for index in range(len(images)):
                decoded = decode_ipa_pair(
                    probabilities["initial"][index],
                    probabilities["final"][index],
                    model.classes.initials,
                    model.classes.finals,
                    pair_set,
                    top_k=1,
                )
                target_pair = (
                    model.classes.initials[int(targets["initial"][index].detach().cpu())],
                    model.classes.finals[int(targets["final"][index].detach().cpu())],
                )
                pair_ok = (
                    decoded["predicted_ipa_initial"], decoded["predicted_ipa_final"]
                ) == target_pair
                tone_ok = bool(predictions["tone"][index] == targets["tone"][index])
                correct["constrained_pair"] += int(pair_ok)
                correct["constrained_full"] += int(pair_ok and tone_ok)
    denom = max(1, total)
    return {
        "initial_acc": correct["initial"] / denom,
        "final_acc": correct["final"] / denom,
        "tone_acc": correct["tone"] / denom,
        "ipa_pair_exact_acc": correct["pair"] / denom,
        "unconstrained_full_syllable_acc": correct["unconstrained_full"] / denom,
        "constrained_ipa_pair_acc": correct["constrained_pair"] / denom if pair_set else 0.0,
        "constrained_full_syllable_acc": correct["constrained_full"] / denom if pair_set else 0.0,
        "val_count": total,
    }


def _split_labels(
    labels: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    split_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if split_mode not in {"random_image", "group_cluster", "group_page"}:
        raise ValueError(f"Unsupported split mode: {split_mode}")
    if split_mode == "random_image":
        indices = list(range(len(labels)))
        random.Random(seed).shuffle(indices)
        cut = max(1, min(len(indices) - 1, int(round(len(indices) * val_ratio))))
        val_indices, train_indices = indices[:cut], indices[cut:]
    else:
        group_key = "cluster_id" if split_mode == "group_cluster" else "page_no"
        missing = [row.get("cell_id", index) for index, row in enumerate(labels) if row.get(group_key) in {None, ""}]
        if missing:
            raise ValueError(f"{split_mode} requires {group_key}; missing on {len(missing)} rows")
        groups = np.asarray([str(row[group_key]) for row in labels])
        if len(set(groups)) < 2:
            raise ValueError(f"{split_mode} requires at least two distinct {group_key} groups")
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_array, val_array = next(splitter.split(np.zeros(len(labels)), groups=groups))
        train_indices, val_indices = train_array.tolist(), val_array.tolist()
    train_rows = [labels[index] for index in train_indices]
    val_rows = [labels[index] for index in val_indices]
    train_clusters = {str(row.get("cluster_id")) for row in train_rows if row.get("cluster_id") not in {None, ""}}
    val_clusters = {str(row.get("cluster_id")) for row in val_rows if row.get("cluster_id") not in {None, ""}}
    train_pages = {str(row.get("page_no")) for row in train_rows if row.get("page_no") not in {None, ""}}
    val_pages = {str(row.get("page_no")) for row in val_rows if row.get("page_no") not in {None, ""}}
    report = {
        "split_mode": split_mode,
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "train_cluster_count": len(train_clusters),
        "validation_cluster_count": len(val_clusters),
        "cluster_overlap_count": len(train_clusters & val_clusters),
        "page_overlap_count": len(train_pages & val_pages),
        "train_cluster_ids": sorted(train_clusters),
        "validation_cluster_ids": sorted(val_clusters),
    }
    if split_mode == "group_cluster" and report["cluster_overlap_count"]:
        raise AssertionError("group_cluster split produced cluster leakage")
    return train_rows, val_rows, report


def _build_class_coverage(
    train_rows: list[dict[str, Any]],
    inventory: Inventory,
    class_space: str,
    initial_key: str,
    final_key: str,
) -> dict[str, Any]:
    initial_counts = Counter(str(row[initial_key]) for row in train_rows)
    final_counts = Counter(str(row[final_key]) for row in train_rows)
    tone_counts = Counter(str(row["tone"]) for row in train_rows)
    if class_space == "ipa":
        expected_initials = ["", *ipa_initial_inventory()]
        expected_finals = ipa_final_inventory()
    else:
        expected_initials = ["", *inventory.initials]
        expected_finals = list(inventory.finals)
    observed_pairs = sorted({(str(row[initial_key]), str(row[final_key])) for row in train_rows})
    allowed_pairs = _allowed_ipa_pairs(inventory) if class_space == "ipa" else []
    return {
        "class_space": class_space,
        "initials": _coverage_section(expected_initials, initial_counts),
        "finals": _coverage_section(expected_finals, final_counts),
        "tones": _coverage_section(list(inventory.tones), tone_counts),
        "ipa_pairs": {
            "total_allowed": len(allowed_pairs),
            "observed_count": len(observed_pairs),
            "observed": [list(pair) for pair in observed_pairs],
            "unobserved_allowed": [list(pair) for pair in sorted(set(allowed_pairs) - set(observed_pairs))],
        },
    }


def _allowed_ipa_pairs(inventory: Inventory) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for initial in ["", *ipa_initial_inventory()]:
        for final in ipa_final_inventory():
            mapped = roman_from_ipa(initial, final)
            if mapped and (inventory.legal_pairs is None or mapped in inventory.legal_pairs):
                pairs.append((initial, final))
    return pairs


def _coverage_section(expected: list[str], counts: Counter[str]) -> dict[str, Any]:
    observed = sorted(counts)
    missing = sorted(set(expected) - set(observed))
    return {
        "schema_total": len(set(expected)),
        "observed": observed,
        "missing": missing,
        "count_per_class": dict(sorted(counts.items())),
        "warning": "These classes cannot be learned by the current supervised classifier." if missing else "",
    }


def _write_coverage_reports(output_dir: Path, coverage: dict[str, Any]) -> None:
    (output_dir / "class_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for kind in ("initials", "finals", "tones"):
        section = coverage[kind]
        for value in sorted(set(section["observed"]) | set(section["missing"])):
            rows.append(
                {
                    "kind": kind,
                    "class": value,
                    "count": section["count_per_class"].get(value, 0),
                    "observed": value in section["observed"],
                    "missing": value in section["missing"],
                }
            )
    _write_tsv(output_dir / "class_coverage.tsv", rows, ["kind", "class", "count", "observed", "missing"])


def _write_rejected_reports(output_dir: Path, rejected: list[dict[str, Any]]) -> None:
    write_jsonl(output_dir / "training_rejected_labels.jsonl", rejected)
    _write_tsv(
        output_dir / "training_rejected_labels.tsv",
        rejected,
        ["cell_id", "cluster_id", "reject_reason", "romanization", "ipa_initial", "ipa_final", "tone"],
    )


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rejected_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "cell_id": row.get("cell_id", ""),
        "cluster_id": row.get("cluster_id", ""),
        "reject_reason": reason,
        "romanization": row.get("rime_syllable", row.get("romanization", "")),
        "ipa_initial": row.get("ipa_initial", ""),
        "ipa_final": row.get("ipa_final", ""),
        "tone": row.get("tone", ""),
    }
