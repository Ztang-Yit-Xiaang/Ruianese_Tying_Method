from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .cluster_diagnostics import diagnose_cluster
from .clustering import cluster_features
from .contact_sheet import ContactItem, make_contact_sheet as make_v2_contact_sheet, make_overview_sheet, select_sample_indices
from .image_features import (
    FeatureConfig,
    ImageQualityConfig,
    assess_image_quality,
    extract_feature,
    load_grayscale_image,
    normalize_ipa_image,
    pixel_feature,
)
from .io_utils import read_jsonl, to_repo_path, write_jsonl
from .tone_features import ToneSpatialConfig, extract_tone_spatial_feature


def cluster_manifest(
    manifest_path: Path,
    output_dir: Path,
    image_key: str = "ipa_clean",
    image_size: int = 96,
    pca_components: int = 48,
    method: str = "auto",
    distance_threshold: float = 0.22,
    kmeans_clusters: int | None = None,
    max_sheet_items: int = 24,
    root_for_paths: Path | None = None,
    task: str = "ipa_body",
    feature_type: str = "pixel_pca",
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    alignment: str = "center",
    padding: int = 4,
    linkage: str = "average",
    random_state: int = 17,
    use_pca: bool | None = None,
    min_ink_pixels: int = 12,
    min_ink_ratio: float = 0.0008,
    max_line_like_component_ratio: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Cluster IPA cell crops while preserving old output fields."""
    root_for_paths = root_for_paths or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_config = _feature_config(
        task=task,
        feature_type=feature_type,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        image_size=image_size,
        alignment=alignment,
        padding=padding,
        random_state=random_state,
    )
    quality_config = ImageQualityConfig(
        min_ink_pixels=min_ink_pixels,
        min_ink_ratio=min_ink_ratio,
        max_line_like_component_ratio=max_line_like_component_ratio,
    )
    accepted_rows, raw_features, rejected_rows = build_feature_table(
        manifest_path=manifest_path,
        output_dir=output_dir,
        image_key=image_key,
        feature_config=feature_config,
        quality_config=quality_config,
        root_for_paths=root_for_paths,
    )
    write_jsonl(output_dir / "rejected_cells.jsonl", rejected_rows)
    if not accepted_rows:
        raise RuntimeError("No manifest rows with usable IPA crops after v2 quality filtering.")

    raw_matrix = np.stack(raw_features).astype(np.float32)
    features, pca_metadata = _project_features(raw_matrix, pca_components, task, use_pca, random_state)
    cluster_result = cluster_features(
        features,
        method=method,
        distance_threshold=distance_threshold,
        linkage=linkage,
        kmeans_clusters=kmeans_clusters,
        random_state=random_state,
    )
    labels = cluster_result.labels
    np.savez_compressed(
        output_dir / "features.npz",
        ids=np.array([row["id"] for row in accepted_rows]),
        features=features,
        raw_features=raw_matrix,
        labels=labels,
        probabilities=_optional_array(cluster_result.probabilities),
        outlier_scores=_optional_array(cluster_result.outlier_scores),
        pca_components=pca_metadata.get("components", np.array([], dtype=np.float32)),
        pca_mean=pca_metadata.get("mean", np.array([], dtype=np.float32)),
    )

    cell_rows, cluster_rows = _write_cluster_outputs(
        manifest_path=manifest_path,
        output_dir=output_dir,
        root_for_paths=root_for_paths,
        image_key=image_key,
        rows=accepted_rows,
        features=features,
        labels=labels,
        probabilities=cluster_result.probabilities,
        outlier_scores=cluster_result.outlier_scores,
        distance_threshold=distance_threshold,
        max_sheet_items=max_sheet_items,
        random_state=random_state,
    )
    _write_metadata(
        output_dir,
        manifest_path,
        image_key,
        feature_config,
        quality_config,
        raw_matrix,
        features,
        pca_components,
        pca_metadata,
        cluster_result.method_used,
        cluster_result.metadata or {},
        method,
        distance_threshold,
        linkage,
        random_state,
        len(rejected_rows),
    )
    return cell_rows, cluster_rows


def build_feature_table(
    manifest_path: Path,
    output_dir: Path,
    image_key: str,
    feature_config: FeatureConfig,
    quality_config: ImageQualityConfig,
    root_for_paths: Path | None = None,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
    rows = [row for row in read_jsonl(manifest_path) if row.get("paths", {}).get(image_key)]
    normalized_dir = output_dir / "normalized_images"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        image_path = _resolve_path(manifest_path, row["paths"][image_key])
        if not image_path.exists():
            rejected.append(_reject_row(row, image_path, ["missing_image"]))
            continue
        image = load_grayscale_image(image_path)
        quality = assess_image_quality(image, quality_config)
        if not quality["accepted"]:
            rejected.append(_reject_row(row, image_path, quality["reject_reasons"], quality))
            continue
        if feature_config.task == "tone_spatial":
            raw_feature, tone_metadata = extract_tone_spatial_feature(image, ToneSpatialConfig())
            norm = float(np.linalg.norm(raw_feature))
            if norm > 0:
                raw_feature = raw_feature / norm
            normalized = normalize_ipa_image(image, feature_config, crop_to_ink=False)
            extra = {"tone_spatial": tone_metadata}
        else:
            raw_feature, normalized = extract_feature(image, feature_config)
            extra = {}
        norm_path = normalized_dir / f"{row['id']}.png"
        normalized.save(norm_path)
        item = dict(row)
        item["computed_quality"] = quality
        item["normalized_image"] = to_repo_path(norm_path, root_for_paths or output_dir)
        item["resolved_normalized_path"] = str(norm_path)
        item["resolved_image_path"] = str(image_path)
        item.update(extra)
        accepted.append(item)
        features.append(raw_feature.astype(np.float32))
    return accepted, features, rejected


def image_vector(path: Path, size: int) -> np.ndarray:
    """Backward-compatible pixel feature used by the original cluster.py."""
    config = FeatureConfig(feature_type="pixel_pca", canvas_width=size, canvas_height=size, padding=4)
    image = load_grayscale_image(path)
    normalized = normalize_ipa_image(image, config, crop_to_ink=True)
    return pixel_feature(normalized)


def make_contact_sheet(items: list[tuple[str, Path]], out_path: Path, tile_size: int = 112, columns: int = 6) -> None:
    """Backward-compatible contact sheet wrapper for old callers."""
    contact_items = [
        ContactItem(cell_id=cell_id, page_no=None, row_index=None, raw_path=path, normalized_path=None)
        for cell_id, path in items
    ]
    make_v2_contact_sheet(contact_items, out_path, tile_width=tile_size, tile_height=tile_size, columns=columns)


def _write_cluster_outputs(
    manifest_path: Path,
    output_dir: Path,
    root_for_paths: Path,
    image_key: str,
    rows: list[dict[str, Any]],
    features: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    outlier_scores: np.ndarray | None,
    distance_threshold: float,
    max_sheet_items: int,
    random_state: int,
) -> tuple[list[dict], list[dict]]:
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_cluster[int(label)].append(idx)

    pre_diagnostics: dict[int, Any] = {}
    medoid_vectors: list[np.ndarray] = []
    for cluster_label, indices in by_cluster.items():
        cluster_features = features[indices]
        cluster_rows = [rows[idx] for idx in indices]
        diag = diagnose_cluster(cluster_rows, cluster_features, distance_threshold=distance_threshold)
        pre_diagnostics[cluster_label] = diag
        medoid_vectors.append(cluster_features[diag.medoid_index])
    all_medoids = np.stack(medoid_vectors) if medoid_vectors else None

    cell_rows: list[dict] = []
    for idx, (row, label) in enumerate(zip(rows, labels, strict=True)):
        cluster_id = _cluster_id(int(label))
        item = {
            "cell_id": row["id"],
            "cluster_id": cluster_id,
            "page_no": row.get("page_no"),
            "row_index": row.get("row_index"),
            "ipa_clean": row["paths"][image_key],
            "normalized_image": row.get("normalized_image"),
            "label_status": "unlabeled",
        }
        if probabilities is not None and len(probabilities) == len(rows):
            item["membership_probability"] = round(float(probabilities[idx]), 6)
        if outlier_scores is not None and len(outlier_scores) == len(rows):
            item["outlier_score"] = round(float(outlier_scores[idx]), 6)
        cell_rows.append(item)

    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    cluster_rows: list[dict] = []
    for cluster_label, indices in sorted(by_cluster.items(), key=lambda pair: (pair[0] < 0, pair[0])):
        cluster_features = features[indices]
        cluster_items = [rows[idx] for idx in indices]
        diag = diagnose_cluster(cluster_items, cluster_features, all_medoids, distance_threshold)
        samples = select_sample_indices(
            cluster_features,
            diag.distances_to_medoid,
            diag.medoid_index,
            max(1, min(max_sheet_items, 8)),
            random_state=random_state,
        )
        cluster_id = _cluster_id(cluster_label)
        contact_sets = {
            key: _contact_items(
                manifest_path,
                cluster_items,
                samples[key],
                diag.distances_to_medoid,
                probabilities,
                indices,
            )
            for key in ("core", "boundary", "diverse", "random")
        }
        prefix = "noise" if cluster_label < 0 else cluster_id
        core_path = sheet_dir / f"{prefix}_core.png"
        boundary_path = sheet_dir / f"{prefix}_boundary.png"
        overview_path = sheet_dir / f"{prefix}_overview.png"
        make_v2_contact_sheet(contact_sets["core"], core_path, title=f"{cluster_id} core")
        make_v2_contact_sheet(contact_sets["boundary"], boundary_path, title=f"{cluster_id} boundary")
        make_overview_sheet(
            [
                ("core", contact_sets["core"]),
                ("diverse", contact_sets["diverse"]),
                ("boundary", contact_sets["boundary"]),
            ],
            overview_path,
        )
        representative_ids = [cluster_items[idx]["id"] for idx in samples["core"]]
        manifest_row = {
            "cluster_id": cluster_id,
            "size": len(indices),
            "representative_cell_ids": representative_ids,
            "contact_sheet": to_repo_path(overview_path, root_for_paths),
            "core_contact_sheet": to_repo_path(core_path, root_for_paths),
            "boundary_contact_sheet": to_repo_path(boundary_path, root_for_paths),
            "overview_contact_sheet": to_repo_path(overview_path, root_for_paths),
            "label_status": "unlabeled",
            **diag.as_manifest_fields(),
        }
        cluster_rows.append(manifest_row)

    write_jsonl(output_dir / "cell_clusters.jsonl", cell_rows)
    write_jsonl(output_dir / "cluster_manifest.jsonl", cluster_rows)
    return cell_rows, cluster_rows


def _contact_items(
    manifest_path: Path,
    rows: list[dict[str, Any]],
    local_indices: list[int],
    distances: np.ndarray,
    probabilities: np.ndarray | None,
    global_indices: list[int],
) -> list[ContactItem]:
    items: list[ContactItem] = []
    for local_idx in local_indices:
        row = rows[local_idx]
        norm_value = row.get("resolved_normalized_path") or row.get("normalized_image")
        norm_path = _resolve_path(manifest_path, norm_value) if norm_value else None
        raw_path = _resolve_path(manifest_path, row["paths"].get("ipa_clean", ""))
        prob = None
        global_idx = global_indices[local_idx]
        if probabilities is not None and len(probabilities) > global_idx:
            prob = float(probabilities[global_idx])
        items.append(
            ContactItem(
                cell_id=row["id"],
                page_no=row.get("page_no"),
                row_index=row.get("row_index"),
                raw_path=raw_path,
                normalized_path=norm_path,
                distance=float(distances[local_idx]),
                probability=prob,
            )
        )
    return items


def _project_features(
    raw_matrix: np.ndarray,
    pca_components: int,
    task: str,
    use_pca: bool | None,
    random_state: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | bool | int]]:
    if use_pca is None:
        use_pca = task != "tone_spatial" and pca_components > 0 and raw_matrix.shape[0] > 2
    if use_pca:
        n_components = max(2, min(pca_components, raw_matrix.shape[0] - 1, raw_matrix.shape[1]))
        pca = PCA(n_components=n_components, random_state=random_state)
        features = pca.fit_transform(raw_matrix)
        features = normalize(features)
        return features.astype(np.float32), {
            "enabled": True,
            "n_components": n_components,
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
        }
    return normalize(raw_matrix).astype(np.float32), {"enabled": False, "n_components": 0}


def _write_metadata(
    output_dir: Path,
    manifest_path: Path,
    image_key: str,
    feature_config: FeatureConfig,
    quality_config: ImageQualityConfig,
    raw_matrix: np.ndarray,
    features: np.ndarray,
    requested_pca_components: int,
    pca_metadata: dict[str, Any],
    method_used: str,
    method_metadata: dict[str, Any],
    requested_method: str,
    distance_threshold: float,
    linkage: str,
    random_state: int,
    rejected_count: int,
) -> None:
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "program_version": "ruian_ipa_pipeline.cluster_v2",
        "manifest_path": str(manifest_path),
        "image_key": image_key,
        "image_count": int(raw_matrix.shape[0]),
        "rejected_count": int(rejected_count),
        "feature_shape": list(raw_matrix.shape),
        "projected_feature_shape": list(features.shape),
        "feature_config": feature_config.metadata(),
        "quality_config": quality_config.__dict__,
        "pca": {
            "enabled": bool(pca_metadata.get("enabled", False)),
            "requested_components": requested_pca_components,
            "components": int(pca_metadata.get("n_components", 0)),
            "normalization": "l2_after_projection",
        },
        "clustering": {
            "requested_method": requested_method,
            "method_used": method_used,
            "distance_threshold": distance_threshold,
            "linkage": linkage,
            "random_state": random_state,
            **method_metadata,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _feature_config(
    task: str,
    feature_type: str,
    canvas_width: int | None,
    canvas_height: int | None,
    image_size: int,
    alignment: str,
    padding: int,
    random_state: int,
) -> FeatureConfig:
    if canvas_width is None:
        canvas_width = image_size if feature_type == "pixel_pca" else 256
    if canvas_height is None:
        canvas_height = image_size if feature_type == "pixel_pca" else 64
    return FeatureConfig(
        task=task,
        feature_type=feature_type,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        padding=padding,
        alignment=alignment,
        random_state=random_state,
    )


def _reject_row(
    row: dict[str, Any],
    image_path: Path,
    reasons: list[str],
    quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cell_id": row.get("id"),
        "page_no": row.get("page_no"),
        "row_index": row.get("row_index"),
        "image_path": str(image_path),
        "reject_reasons": reasons,
        "quality": quality or {},
    }


def _optional_array(values: np.ndarray | None) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) if values is not None else np.array([], dtype=np.float32)


def _cluster_id(label: int) -> str:
    return f"cluster_{label:04d}" if label >= 0 else "noise"


def _resolve_path(anchor: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        anchor.parent / path,
        anchor.parent.parent / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster Rui'an IPA cell crops with v2 local features.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", "--out", dest="output_dir", type=Path, required=True)
    parser.add_argument("--image-key", default="ipa_clean")
    parser.add_argument("--task", choices=["ipa_body", "tone_spatial"], default="ipa_body")
    parser.add_argument("--feature-type", choices=["pixel_pca", "hog"], default="pixel_pca")
    parser.add_argument("--canvas-width", type=int)
    parser.add_argument("--canvas-height", type=int)
    parser.add_argument("--alignment", choices=["center", "baseline"], default="center")
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--method", choices=["auto", "agglomerative", "dbscan", "hdbscan", "kmeans"], default="auto")
    parser.add_argument("--linkage", choices=["average", "complete"], default="average")
    parser.add_argument("--distance-threshold", type=float, default=0.22)
    parser.add_argument("--kmeans-clusters", type=int)
    parser.add_argument("--max-sheet-items", type=int, default=24)
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--min-ink-pixels", type=int, default=12)
    parser.add_argument("--min-ink-ratio", type=float, default=0.0008)
    parser.add_argument("--max-line-like-component-ratio", type=float, default=0.85)
    args = parser.parse_args()
    _, clusters = cluster_manifest(
        args.manifest,
        args.output_dir,
        image_key=args.image_key,
        pca_components=args.pca_components,
        method=args.method,
        distance_threshold=args.distance_threshold,
        kmeans_clusters=args.kmeans_clusters,
        max_sheet_items=args.max_sheet_items,
        task=args.task,
        feature_type=args.feature_type,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        alignment=args.alignment,
        padding=args.padding,
        linkage=args.linkage,
        random_state=args.random_state,
        use_pca=False if args.no_pca else None,
        min_ink_pixels=args.min_ink_pixels,
        min_ink_ratio=args.min_ink_ratio,
        max_line_like_component_ratio=args.max_line_like_component_ratio,
    )
    print(f"Wrote {len(clusters)} clusters into {args.output_dir}")


if __name__ == "__main__":
    main()
