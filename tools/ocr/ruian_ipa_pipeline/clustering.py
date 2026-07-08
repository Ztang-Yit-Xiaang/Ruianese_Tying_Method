from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray | None = None
    outlier_scores: np.ndarray | None = None
    method_used: str = ""
    metadata: dict[str, Any] | None = None


def cluster_features(
    features: np.ndarray,
    method: str = "auto",
    distance_threshold: float = 0.22,
    linkage: str = "average",
    kmeans_clusters: int | None = None,
    random_state: int = 17,
) -> ClusterResult:
    """Cluster normalized features with conservative defaults."""
    if method == "auto":
        if len(features) <= 2500:
            method = "agglomerative"
        elif _hdbscan_available():
            method = "hdbscan"
        else:
            raise RuntimeError(
                "method=auto refuses to switch large data to kmeans. "
                "Install hdbscan or explicitly pass --method agglomerative/dbscan/kmeans."
            )

    if method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=True,
        )
        labels = model.fit_predict(features)
        return ClusterResult(
            labels=np.asarray(labels, dtype=int),
            method_used="agglomerative",
            metadata={"metric": "cosine", "linkage": linkage, "distance_threshold": distance_threshold},
        )

    if method == "dbscan":
        model = DBSCAN(eps=distance_threshold, min_samples=2, metric="cosine")
        labels = model.fit_predict(features)
        return ClusterResult(
            labels=np.asarray(labels, dtype=int),
            method_used="dbscan",
            metadata={"metric": "cosine", "eps": distance_threshold, "min_samples": 2},
        )

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("hdbscan is not installed. Install with: python -m pip install hdbscan") from exc
        model = hdbscan.HDBSCAN(metric="euclidean", min_cluster_size=2, prediction_data=False)
        labels = model.fit_predict(features)
        probabilities = getattr(model, "probabilities_", None)
        outlier_scores = getattr(model, "outlier_scores_", None)
        return ClusterResult(
            labels=np.asarray(labels, dtype=int),
            probabilities=np.asarray(probabilities, dtype=np.float32) if probabilities is not None else None,
            outlier_scores=np.asarray(outlier_scores, dtype=np.float32) if outlier_scores is not None else None,
            method_used="hdbscan",
            metadata={"metric": "euclidean", "min_cluster_size": 2},
        )

    if method == "kmeans":
        clusters = kmeans_clusters or max(8, int(np.sqrt(len(features)) * 2))
        model = MiniBatchKMeans(n_clusters=clusters, random_state=random_state, n_init="auto")
        labels = model.fit_predict(features)
        return ClusterResult(
            labels=np.asarray(labels, dtype=int),
            method_used="kmeans",
            metadata={"n_clusters": clusters, "random_state": random_state},
        )

    raise ValueError(f"Unknown cluster method: {method}")


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
