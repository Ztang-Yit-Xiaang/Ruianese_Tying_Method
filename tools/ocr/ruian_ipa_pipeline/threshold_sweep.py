from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .cluster import _project_features, build_feature_table
from .cluster_diagnostics import diagnose_cluster
from .clustering import cluster_features
from .image_features import FeatureConfig, ImageQualityConfig
from .io_utils import write_jsonl


def run_threshold_sweep(
    manifest_path: Path,
    output_dir: Path,
    thresholds: list[float],
    image_key: str = "ipa_clean",
    task: str = "ipa_body",
    feature_type: str = "hog",
    canvas_width: int = 256,
    canvas_height: int = 64,
    alignment: str = "baseline",
    padding: int = 8,
    linkage: str = "complete",
    pca_components: int = 48,
    random_state: int = 17,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_config = FeatureConfig(
        task=task,
        feature_type=feature_type,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        padding=padding,
        alignment=alignment,
        random_state=random_state,
    )
    accepted_rows, raw_features, rejected_rows = build_feature_table(
        manifest_path,
        output_dir,
        image_key,
        feature_config,
        ImageQualityConfig(),
        root_for_paths=output_dir,
    )
    write_jsonl(output_dir / "rejected_cells.jsonl", rejected_rows)
    if not accepted_rows:
        raise RuntimeError("No usable images for threshold sweep.")
    raw_matrix = np.stack(raw_features).astype(np.float32)
    features, _ = _project_features(raw_matrix, pca_components, task, None, random_state)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        result = cluster_features(
            features,
            method="agglomerative",
            distance_threshold=threshold,
            linkage=linkage,
            random_state=random_state,
        )
        rows.append(_sweep_metrics(accepted_rows, features, result.labels, threshold))
    (output_dir / "threshold_sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_thresholds(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(part) for part in value.split(":")]
        thresholds: list[float] = []
        current = start
        while current <= stop + step / 2:
            thresholds.append(round(current, 10))
            current += step
        return thresholds
    return [float(part) for part in value.split(",") if part.strip()]


def _sweep_metrics(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    by_cluster: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_cluster.setdefault(int(label), []).append(idx)
    sizes = [len(indices) for label, indices in by_cluster.items() if label >= 0]
    singleton_count = sum(1 for size in sizes if size == 1)
    noise_count = len(by_cluster.get(-1, []))
    mean_distances: list[float] = []
    max_distances: list[float] = []
    suspicious = 0
    for label, indices in by_cluster.items():
        if label < 0:
            continue
        diag = diagnose_cluster([rows[i] for i in indices], features[indices], distance_threshold=threshold)
        mean_distances.append(diag.mean_distance)
        max_distances.append(diag.max_distance)
        if diag.warnings:
            suspicious += 1
    total = len(rows)
    return {
        "threshold": threshold,
        "cluster_count": len(sizes),
        "singleton_count": singleton_count,
        "singleton_ratio": round(singleton_count / max(1, total), 6),
        "max_cluster_size": max(sizes) if sizes else 0,
        "avg_cluster_size": round(float(np.mean(sizes)), 6) if sizes else 0.0,
        "noise_count": noise_count,
        "mean_intra_cluster_distance": round(float(np.mean(mean_distances)), 6) if mean_distances else 0.0,
        "max_intra_cluster_distance": round(float(np.max(max_distances)), 6) if max_distances else 0.0,
        "suspected_mixed_cluster_count": suspicious,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run threshold sweep diagnostics for Rui'an IPA clustering.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thresholds", default="0.12:0.34:0.02")
    parser.add_argument("--image-key", default="ipa_clean")
    parser.add_argument("--task", choices=["ipa_body", "tone_spatial"], default="ipa_body")
    parser.add_argument("--feature-type", choices=["pixel_pca", "hog"], default="hog")
    parser.add_argument("--canvas-width", type=int, default=256)
    parser.add_argument("--canvas-height", type=int, default=64)
    parser.add_argument("--alignment", choices=["center", "baseline"], default="baseline")
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--linkage", choices=["average", "complete"], default="complete")
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--random-state", type=int, default=17)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.manifest.parent / "threshold_sweep")
    rows = run_threshold_sweep(
        manifest_path=args.manifest,
        output_dir=output_dir,
        thresholds=parse_thresholds(args.thresholds),
        image_key=args.image_key,
        task=args.task,
        feature_type=args.feature_type,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        alignment=args.alignment,
        padding=args.padding,
        linkage=args.linkage,
        pca_components=args.pca_components,
        random_state=args.random_state,
    )
    print(f"Wrote threshold sweep for {len(rows)} thresholds into {output_dir}")


if __name__ == "__main__":
    main()
