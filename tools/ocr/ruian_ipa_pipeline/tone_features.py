from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


REGION_NAMES = ("left_top", "right_top", "left_bottom", "right_bottom")


@dataclass(frozen=True)
class ToneSpatialConfig:
    ink_threshold: int = 24
    include_local_hog: bool = False


def extract_tone_spatial_feature(
    image: Image.Image,
    config: ToneSpatialConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract spatial tone features from the full cell without recentering tone ink."""
    config = config or ToneSpatialConfig()
    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    fg = (255 - arr) > config.ink_threshold
    body_center = _estimate_body_center(fg)
    h, w = fg.shape[:2]
    cx, cy = body_center
    regions = {
        "left_top": (0, 0, int(round(cx)), int(round(cy))),
        "right_top": (int(round(cx)), 0, w, int(round(cy))),
        "left_bottom": (0, int(round(cy)), int(round(cx)), h),
        "right_bottom": (int(round(cx)), int(round(cy)), w, h),
    }

    feature_values: list[float] = []
    region_metrics: dict[str, Any] = {}
    for name in REGION_NAMES:
        x0, y0, x1, y1 = _nonempty_region(regions[name], w, h)
        mask = fg[y0:y1, x0:x1]
        metrics = _region_metrics(mask, x0, y0, w, h, cx, cy)
        region_metrics[name] = metrics
        feature_values.extend(
            [
                metrics["ink_ratio"],
                metrics["component_count"],
                metrics["centroid_x"],
                metrics["centroid_y"],
                metrics["largest_component_area_norm"],
                metrics["largest_aspect_ratio"],
                metrics["relative_centroid_x"],
                metrics["relative_centroid_y"],
            ]
        )
    metadata = {
        "body_center": [round(float(cx), 4), round(float(cy), 4)],
        "regions": region_metrics,
    }
    return np.asarray(feature_values, dtype=np.float32), metadata


def _estimate_body_center(fg: np.ndarray) -> tuple[float, float]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    components: list[tuple[int, int, int, int, int, float, float]] = []
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        if area > 0:
            components.append((x, y, w, h, area, cx, cy))
    if not components:
        h, w = fg.shape[:2]
        return w / 2.0, h / 2.0
    largest = max(c[4] for c in components)
    body = [c for c in components if c[4] >= max(4, largest * 0.12)]
    xs0 = [c[0] for c in body]
    ys0 = [c[1] for c in body]
    xs1 = [c[0] + c[2] for c in body]
    ys1 = [c[1] + c[3] for c in body]
    return (min(xs0) + max(xs1)) / 2.0, (min(ys0) + max(ys1)) / 2.0


def _region_metrics(
    mask: np.ndarray,
    x_offset: int,
    y_offset: int,
    full_w: int,
    full_h: int,
    body_cx: float,
    body_cy: float,
) -> dict[str, float]:
    area = max(1, int(mask.size))
    ink_pixels = int(mask.sum())
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components: list[dict[str, float]] = []
    for idx in range(1, count):
        x, y, w, h, comp_area = [int(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        if comp_area <= 0:
            continue
        components.append(
            {
                "area": float(comp_area),
                "w": float(w),
                "h": float(h),
                "cx": x_offset + cx,
                "cy": y_offset + cy,
            }
        )
    if components:
        weighted_area = sum(c["area"] for c in components)
        centroid_x = sum(c["cx"] * c["area"] for c in components) / weighted_area
        centroid_y = sum(c["cy"] * c["area"] for c in components) / weighted_area
        largest = max(components, key=lambda c: c["area"])
        largest_area_norm = largest["area"] / area
        largest_aspect = largest["w"] / max(1.0, largest["h"])
    else:
        centroid_x = 0.0
        centroid_y = 0.0
        largest_area_norm = 0.0
        largest_aspect = 0.0
    return {
        "ink_ratio": float(ink_pixels / area),
        "component_count": float(len(components)),
        "centroid_x": float(centroid_x / max(1, full_w)),
        "centroid_y": float(centroid_y / max(1, full_h)),
        "largest_component_area_norm": float(largest_area_norm),
        "largest_aspect_ratio": float(largest_aspect),
        "relative_centroid_x": float((centroid_x - body_cx) / max(1, full_w)) if components else 0.0,
        "relative_centroid_y": float((centroid_y - body_cy) / max(1, full_h)) if components else 0.0,
    }


def _nonempty_region(region: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = region
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1
