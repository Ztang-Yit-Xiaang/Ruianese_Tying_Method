from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClusterDiagnostics:
    medoid_index: int
    medoid_cell_id: str
    distances_to_medoid: np.ndarray
    mean_distance: float
    max_distance: float
    p90_distance: float
    compactness: float
    silhouette_like_score: float | None
    quality_status: str
    warnings: list[str]

    def as_manifest_fields(self) -> dict[str, Any]:
        return {
            "medoid_cell_id": self.medoid_cell_id,
            "mean_distance": round(self.mean_distance, 6),
            "max_distance": round(self.max_distance, 6),
            "p90_distance": round(self.p90_distance, 6),
            "compactness": round(self.compactness, 6),
            "silhouette_like_score": None
            if self.silhouette_like_score is None
            else round(self.silhouette_like_score, 6),
            "quality_status": self.quality_status,
            "warnings": self.warnings,
        }


def cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
    sim = np.clip(features @ features.T, -1.0, 1.0)
    return 1.0 - sim


def diagnose_cluster(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    all_cluster_medoids: np.ndarray | None = None,
    distance_threshold: float = 0.22,
) -> ClusterDiagnostics:
    if len(rows) == 0:
        raise ValueError("Cannot diagnose an empty cluster.")
    if len(rows) == 1:
        medoid_index = 0
        distances = np.zeros(1, dtype=np.float32)
    else:
        distances_matrix = cosine_distance_matrix(features)
        medoid_index = int(np.argmin(distances_matrix.mean(axis=1)))
        distances = distances_matrix[medoid_index].astype(np.float32)

    mean_distance = float(np.mean(distances))
    max_distance = float(np.max(distances))
    p90_distance = float(np.percentile(distances, 90))
    compactness = float(1.0 / (1.0 + mean_distance))
    silhouette_like = _silhouette_like(features, distances, medoid_index, all_cluster_medoids)

    warnings = _quality_warnings(rows, max_distance, p90_distance, distance_threshold)
    quality_status = "review" if warnings else "ok"
    return ClusterDiagnostics(
        medoid_index=medoid_index,
        medoid_cell_id=str(rows[medoid_index].get("id", "")),
        distances_to_medoid=distances,
        mean_distance=mean_distance,
        max_distance=max_distance,
        p90_distance=p90_distance,
        compactness=compactness,
        silhouette_like_score=silhouette_like,
        quality_status=quality_status,
        warnings=warnings,
    )


def _silhouette_like(
    features: np.ndarray,
    distances_to_medoid: np.ndarray,
    medoid_index: int,
    all_cluster_medoids: np.ndarray | None,
) -> float | None:
    if all_cluster_medoids is None or len(all_cluster_medoids) <= 1:
        return None
    own = float(np.mean(distances_to_medoid))
    medoid = features[medoid_index]
    other_distances = 1.0 - np.clip(all_cluster_medoids @ medoid, -1.0, 1.0)
    other_distances = other_distances[other_distances > 1e-8]
    if other_distances.size == 0:
        return None
    nearest_other = float(np.min(other_distances))
    denom = max(own, nearest_other, 1e-8)
    return (nearest_other - own) / denom


def _quality_warnings(
    rows: list[dict[str, Any]],
    max_distance: float,
    p90_distance: float,
    distance_threshold: float,
) -> list[str]:
    warnings: list[str] = []
    if max_distance > distance_threshold:
        warnings.append("large_internal_distance")
    if p90_distance > distance_threshold * 0.85:
        warnings.append("large_p90_distance")
    aspect_ratios: list[float] = []
    low_ink = 0
    for row in rows:
        quality = row.get("computed_quality") or row.get("quality") or {}
        width = float(quality.get("bbox_width") or quality.get("row_height") or 0)
        height = float(quality.get("bbox_height") or quality.get("row_height") or 0)
        if width > 0 and height > 0:
            aspect_ratios.append(width / height)
        if float(quality.get("ink_ratio") or 0.0) < 0.003:
            low_ink += 1
    if len(aspect_ratios) >= 3:
        ratios = np.asarray(aspect_ratios, dtype=np.float32)
        if float(ratios.max() / max(1e-6, ratios.min())) > 3.0:
            warnings.append("aspect_ratio_outlier")
    if low_ink:
        warnings.append("contains_low_ink_samples")
    return warnings
