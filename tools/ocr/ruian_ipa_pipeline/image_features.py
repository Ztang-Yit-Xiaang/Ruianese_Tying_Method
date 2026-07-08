from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageQualityConfig:
    min_ink_pixels: int = 12
    min_ink_ratio: float = 0.0008
    max_line_like_component_ratio: float = 0.85
    ink_threshold: int = 24


@dataclass(frozen=True)
class FeatureConfig:
    task: str = "ipa_body"
    feature_type: str = "pixel_pca"
    canvas_width: int = 96
    canvas_height: int = 96
    padding: int = 4
    alignment: str = "center"
    bbox_padding: int = 3
    random_state: int = 17
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2
    prefer_skimage_hog: bool = True

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureResult:
    row: dict[str, Any]
    image_path: Path
    raw_feature: np.ndarray
    normalized_image: Image.Image | None
    quality: dict[str, Any]


def load_grayscale_image(path: str | Path) -> Image.Image:
    """Load a grayscale image without changing geometry."""
    return Image.open(path).convert("L")


def find_ink_bbox(
    image: Image.Image | np.ndarray,
    ink_threshold: int = 24,
    padding: int = 0,
) -> BBox | None:
    """Return x0, y0, x1, y1 for foreground ink, or None if no ink exists."""
    arr = _as_array(image)
    fg = 255 - arr
    ys, xs = np.where(fg > ink_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(arr.shape[1], int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(arr.shape[0], int(ys.max()) + padding + 1)
    return x0, y0, x1, y1


def estimate_baseline(image: Image.Image | np.ndarray, ink_threshold: int = 24) -> int:
    """Estimate a baseline from the lower body ink projection."""
    arr = _as_array(image)
    fg = (255 - arr) > ink_threshold
    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return arr.shape[0] // 2
    projection = fg.sum(axis=1)
    lower_start = int(max(0, np.percentile(ys, 45)))
    lower_projection = projection[lower_start:]
    if lower_projection.size and lower_projection.max() > 0:
        return lower_start + int(np.argmax(lower_projection))
    return int(ys.max())


def normalize_ipa_image(
    image: Image.Image,
    config: FeatureConfig,
    crop_to_ink: bool = True,
) -> Image.Image:
    """Normalize an IPA body crop to a fixed canvas without stretching."""
    img = image.copy()
    if crop_to_ink:
        bbox = find_ink_bbox(img, padding=config.bbox_padding)
        if bbox is not None:
            img = img.crop(bbox)

    max_w = max(1, config.canvas_width - config.padding * 2)
    max_h = max(1, config.canvas_height - config.padding * 2)
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (config.canvas_width, config.canvas_height), 255)

    x = (config.canvas_width - img.width) // 2
    if config.alignment == "baseline":
        baseline = estimate_baseline(img)
        target = config.canvas_height - config.padding - max(3, config.canvas_height // 9)
        y = target - baseline
        y = max(config.padding, min(config.canvas_height - config.padding - img.height, y))
    elif config.alignment == "center":
        y = (config.canvas_height - img.height) // 2
    else:
        raise ValueError(f"Unsupported alignment: {config.alignment}")
    canvas.paste(img, (x, y))
    return ImageOps.autocontrast(canvas)


def assess_image_quality(
    image: Image.Image,
    config: ImageQualityConfig | None = None,
) -> dict[str, Any]:
    """Compute local image quality metrics and reject obvious blanks/noise/lines."""
    config = config or ImageQualityConfig()
    arr = _as_array(image)
    fg = (255 - arr) > config.ink_threshold
    ink_pixels = int(fg.sum())
    ink_ratio = float(ink_pixels / max(1, fg.size))
    bbox = find_ink_bbox(arr, config.ink_threshold, padding=0)
    component_stats = _component_stats(fg)

    reasons: list[str] = []
    if ink_pixels < config.min_ink_pixels:
        reasons.append("too_few_ink_pixels")
    if ink_ratio < config.min_ink_ratio:
        reasons.append("low_ink_ratio")
    if component_stats["component_count"] == 0:
        reasons.append("no_components")
    if bbox is None:
        bbox_width = 0
        bbox_height = 0
    else:
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        if bbox_width <= 1 or bbox_height <= 1:
            reasons.append("tiny_ink_bbox")
    if component_stats["line_like_component_ratio"] >= config.max_line_like_component_ratio:
        reasons.append("line_like_component")

    return {
        "accepted": not reasons,
        "reject_reasons": reasons,
        "ink_pixels": ink_pixels,
        "ink_ratio": round(ink_ratio, 8),
        "ink_bbox": list(bbox) if bbox else None,
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        **component_stats,
    }


def extract_feature(
    image: Image.Image,
    config: FeatureConfig,
) -> tuple[np.ndarray, Image.Image]:
    """Extract a deterministic local feature vector and its normalized view."""
    normalized = normalize_ipa_image(image, config, crop_to_ink=config.task == "ipa_body")
    if config.feature_type == "pixel_pca":
        feature = pixel_feature(normalized)
    elif config.feature_type == "hog":
        feature = hog_feature(normalized, config)
    else:
        raise ValueError(f"Unsupported feature_type: {config.feature_type}")
    return _l2_normalize(feature), normalized


def pixel_feature(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return ((255.0 - arr).reshape(-1) / 255.0).astype(np.float32)


def hog_feature(image: Image.Image, config: FeatureConfig) -> np.ndarray:
    if config.prefer_skimage_hog:
        try:
            from skimage.feature import hog as skimage_hog  # type: ignore

            arr = np.asarray(image, dtype=np.float32) / 255.0
            return np.asarray(
                skimage_hog(
                    arr,
                    orientations=config.hog_orientations,
                    pixels_per_cell=(config.hog_pixels_per_cell, config.hog_pixels_per_cell),
                    cells_per_block=(config.hog_cells_per_block, config.hog_cells_per_block),
                    block_norm="L2-Hys",
                    feature_vector=True,
                ),
                dtype=np.float32,
            )
        except ModuleNotFoundError:
            pass
    return _opencv_hog_feature(image, config)


def _opencv_hog_feature(image: Image.Image, config: FeatureConfig) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    width, height = image.size
    cell = max(1, int(config.hog_pixels_per_cell))
    block = max(1, int(config.hog_cells_per_block)) * cell
    win_w = max(cell, (width // cell) * cell)
    win_h = max(cell, (height // cell) * cell)
    if (win_w, win_h) != (width, height):
        arr = cv2.resize(arr, (win_w, win_h), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=(win_w, win_h),
        _blockSize=(block, block),
        _blockStride=(cell, cell),
        _cellSize=(cell, cell),
        _nbins=int(config.hog_orientations),
    )
    feature = hog.compute(arr)
    if feature is None:
        raise RuntimeError("OpenCV HOG failed to compute features.")
    return feature.reshape(-1).astype(np.float32)


def _component_stats(fg: np.ndarray) -> dict[str, Any]:
    mask = fg.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[dict[str, Any]] = []
    height, width = mask.shape[:2]
    line_like_ratio = 0.0
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        cx, cy = [float(v) for v in centroids[idx]]
        components.append({"x": x, "y": y, "w": w, "h": h, "area": area, "cx": cx, "cy": cy})
        if h <= 3 and w > 0:
            line_like_ratio = max(line_like_ratio, w / max(1, width))
        if w <= 3 and h > 0:
            line_like_ratio = max(line_like_ratio, h / max(1, height))
    largest = max((c["area"] for c in components), default=0)
    return {
        "component_count": len(components),
        "largest_component_area": int(largest),
        "line_like_component_ratio": round(float(line_like_ratio), 6),
    }


def _as_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
    else:
        arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8, copy=False)


def _l2_normalize(feature: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature
