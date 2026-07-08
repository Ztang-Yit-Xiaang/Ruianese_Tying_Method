from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_utils import imread, imwrite, page_number_from_path, stable_cell_id, to_repo_path, write_jsonl


@dataclass
class ExtractConfig:
    hanzi_col: int = 1
    ipa_col: int = 7
    header_rows: int = 1
    inner_margin: int = 3
    min_ink_ratio: float = 0.001
    include_empty: bool = False
    tone_zone_ratio: float = 0.48
    edge_cleanup_px: int = 6
    min_vertical_lines: int = 8
    min_horizontal_lines: int = 8


def extract_pages(
    page_paths: list[Path],
    output_dir: Path,
    config: ExtractConfig,
    root_for_paths: Path | None = None,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_root = output_dir / "crops"
    rows: list[dict] = []
    failures: list[dict] = []
    for page_path in page_paths:
        try:
            rows.extend(extract_page(page_path, crop_root, config, root_for_paths or output_dir))
        except Exception as exc:  # noqa: BLE001 - collect failures for review.
            failures.append(
                {
                    "page_path": to_repo_path(page_path, root_for_paths),
                    "page_no": page_number_from_path(page_path),
                    "error": repr(exc),
                }
            )
    write_jsonl(output_dir / "ipa_cells_manifest.jsonl", rows)
    if failures:
        write_jsonl(output_dir / "extract_failures.jsonl", failures)
    return rows


def extract_page(page_path: Path, crop_root: Path, config: ExtractConfig, root_for_paths: Path) -> list[dict]:
    page_no = page_number_from_path(page_path)
    img = imread(page_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = binarize_foreground(gray)
    horiz, vert, grid = detect_grid_masks(fg)
    x_lines = line_positions(vert, axis=0)
    y_lines = line_positions(horiz, axis=1)
    if len(x_lines) < config.min_vertical_lines or len(y_lines) < config.min_horizontal_lines:
        raise RuntimeError(f"Not enough table lines: x={len(x_lines)} y={len(y_lines)}")
    x_lines = _trim_sparse_edge_lines(x_lines)
    y_lines = _trim_sparse_edge_lines(y_lines)
    col_count = len(x_lines) - 1
    if col_count <= max(config.hanzi_col, config.ipa_col):
        raise RuntimeError(f"Not enough table columns: {col_count}")

    line_mask = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    fg_without_lines = cv2.bitwise_and(fg, cv2.bitwise_not(line_mask))

    table_bbox = [int(x_lines[0]), int(y_lines[0]), int(x_lines[-1] - x_lines[0]), int(y_lines[-1] - y_lines[0])]
    out: list[dict] = []
    row_index = 0
    for interval_index in range(config.header_rows, len(y_lines) - 1):
        y0, y1 = y_lines[interval_index], y_lines[interval_index + 1]
        if y1 - y0 < 18:
            continue
        hanzi_bbox = _cell_bbox(x_lines, y0, y1, config.hanzi_col, config.inner_margin, gray.shape)
        ipa_bbox = _cell_bbox(x_lines, y0, y1, config.ipa_col, config.inner_margin, gray.shape)
        ipa_clean = _clean_crop(fg_without_lines, ipa_bbox)
        ipa_clean = _clear_border_artifacts(ipa_clean, config.edge_cleanup_px)
        ink_ratio = float(np.count_nonzero(255 - ipa_clean) / max(1, ipa_clean.size))
        has_ipa_ink = ink_ratio >= config.min_ink_ratio
        if not has_ipa_ink and not config.include_empty:
            continue

        cell_id = stable_cell_id(page_no, row_index)
        cell_dir = crop_root / cell_id
        hanzi_raw = _raw_crop(gray, hanzi_bbox)
        ipa_raw = _raw_crop(gray, ipa_bbox)
        tone_clean = _tone_crop(ipa_clean, config.tone_zone_ratio)

        paths = {
            "hanzi_raw": cell_dir / "hanzi_raw.png",
            "ipa_raw": cell_dir / "ipa_raw.png",
            "ipa_clean": cell_dir / "ipa_clean.png",
            "tone_clean": cell_dir / "tone_clean.png",
        }
        imwrite(paths["hanzi_raw"], hanzi_raw)
        imwrite(paths["ipa_raw"], ipa_raw)
        imwrite(paths["ipa_clean"], ipa_clean)
        imwrite(paths["tone_clean"], tone_clean)

        row = {
            "id": cell_id,
            "page_no": page_no,
            "page_path": to_repo_path(page_path, root_for_paths),
            "row_index": row_index,
            "source_row_interval_index": interval_index,
            "table_bbox": table_bbox,
            "hanzi_bbox": hanzi_bbox,
            "ipa_bbox": ipa_bbox,
            "paths": {key: to_repo_path(value, root_for_paths) for key, value in paths.items()},
            "quality": {
                "has_ipa_ink": has_ipa_ink,
                "ink_ratio": round(ink_ratio, 6),
                "row_height": int(y1 - y0),
                "column_count": col_count,
            },
            "label_status": "unlabeled",
        }
        out.append(row)
        row_index += 1
    return out


def binarize_foreground(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )


def detect_grid_masks(fg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = fg.shape[:2]
    horizontal_len = max(50, int(w * 0.055))
    vertical_len = max(50, int(h * 0.035))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_len))
    horiz = cv2.morphologyEx(fg, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(fg, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    grid = cv2.bitwise_or(horiz, vert)
    return horiz, vert, grid


def line_positions(mask: np.ndarray, axis: int) -> list[int]:
    projection = np.count_nonzero(mask, axis=axis)
    if projection.size == 0 or projection.max() == 0:
        return []
    threshold = max(20, int(projection.max() * 0.22))
    indices = np.flatnonzero(projection >= threshold)
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) <= groups[-1][-1] + 3:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [int(round(sum(group) / len(group))) for group in groups if len(group) >= 1]


def _trim_sparse_edge_lines(lines: list[int]) -> list[int]:
    if len(lines) <= 2:
        return lines
    diffs = np.diff(lines)
    if len(diffs) == 0:
        return lines
    median = float(np.median(diffs))
    trimmed = list(lines)
    if diffs[0] > median * 3:
        trimmed = trimmed[1:]
    if len(trimmed) > 2 and trimmed[-1] - trimmed[-2] > median * 3:
        trimmed = trimmed[:-1]
    return trimmed


def _cell_bbox(
    x_lines: list[int],
    y0: int,
    y1: int,
    col: int,
    margin: int,
    shape: tuple[int, int],
) -> list[int]:
    h, w = shape
    x0 = max(0, x_lines[col] + margin)
    x1 = min(w, x_lines[col + 1] - margin)
    yy0 = max(0, y0 + margin)
    yy1 = min(h, y1 - margin)
    return [int(x0), int(yy0), int(max(1, x1 - x0)), int(max(1, yy1 - yy0))]


def _raw_crop(gray: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    return gray[y : y + h, x : x + w].copy()


def _clean_crop(fg_without_lines: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    crop_fg = fg_without_lines[y : y + h, x : x + w]
    return 255 - crop_fg


def _tone_crop(ipa_clean: np.ndarray, ratio: float) -> np.ndarray:
    width = max(1, int(ipa_clean.shape[1] * ratio))
    return ipa_clean[:, :width].copy()


def _clear_border_artifacts(crop: np.ndarray, px: int) -> np.ndarray:
    cleaned = crop.copy()
    if cleaned.size == 0:
        return cleaned
    px = max(1, int(px))
    cleaned[: min(2, cleaned.shape[0]), :] = 255
    cleaned[max(0, cleaned.shape[0] - 2) :, :] = 255
    cleaned[:, max(0, cleaned.shape[1] - px) :] = 255
    cleaned[:, :1] = 255
    return cleaned
