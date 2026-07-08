from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


@dataclass(frozen=True)
class ContactItem:
    cell_id: str
    page_no: int | None
    row_index: int | None
    raw_path: Path
    normalized_path: Path | None
    distance: float = 0.0
    probability: float | None = None


def select_sample_indices(
    features: np.ndarray,
    distances_to_medoid: np.ndarray,
    medoid_index: int,
    limit: int,
    random_state: int = 17,
) -> dict[str, list[int]]:
    limit = max(1, int(limit))
    n = len(distances_to_medoid)
    all_indices = list(range(n))
    core = _unique([medoid_index, *np.argsort(distances_to_medoid).astype(int).tolist()])[:limit]
    boundary = _unique(np.argsort(-distances_to_medoid).astype(int).tolist())[:limit]
    diverse = _farthest_first(features, medoid_index, limit)
    rng = np.random.default_rng(random_state)
    random_order = rng.permutation(all_indices).astype(int).tolist()
    random = _unique(random_order)[:limit]
    return {"core": core, "boundary": boundary, "diverse": diverse, "random": random}


def make_contact_sheet(
    items: list[ContactItem],
    out_path: Path,
    title: str = "",
    tile_width: int = 172,
    tile_height: int = 150,
    columns: int = 5,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        Image.new("RGB", (tile_width, tile_height), "white").save(out_path)
        return
    rows = int(np.ceil(len(items) / columns))
    header = 24 if title else 0
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height + header), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((6, 5), title, fill=(0, 0, 0))
    for idx, item in enumerate(items):
        x = (idx % columns) * tile_width
        y = (idx // columns) * tile_height + header
        _draw_tile(sheet, draw, item, x, y, tile_width, tile_height)
    sheet.save(out_path)


def make_overview_sheet(
    sections: list[tuple[str, list[ContactItem]]],
    out_path: Path,
    tile_width: int = 172,
    tile_height: int = 150,
    columns: int = 5,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nonempty = [(name, items) for name, items in sections if items]
    if not nonempty:
        Image.new("RGB", (tile_width, tile_height), "white").save(out_path)
        return
    section_heights = [24 + int(np.ceil(len(items) / columns)) * tile_height for _, items in nonempty]
    sheet = Image.new("RGB", (columns * tile_width, sum(section_heights)), "white")
    draw = ImageDraw.Draw(sheet)
    y0 = 0
    for (name, items), section_height in zip(nonempty, section_heights, strict=True):
        draw.rectangle((0, y0, columns * tile_width, y0 + 23), fill=(235, 235, 235))
        draw.text((6, y0 + 5), name, fill=(0, 0, 0))
        for idx, item in enumerate(items):
            x = (idx % columns) * tile_width
            y = y0 + 24 + (idx // columns) * tile_height
            _draw_tile(sheet, draw, item, x, y, tile_width, tile_height)
        y0 += section_height
    sheet.save(out_path)


def _draw_tile(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    item: ContactItem,
    x: int,
    y: int,
    tile_width: int,
    tile_height: int,
) -> None:
    draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline=(210, 210, 210))
    raw = _load_tile_image(item.raw_path, tile_width - 10, 44)
    sheet.paste(raw, (x + (tile_width - raw.width) // 2, y + 4))
    if item.normalized_path and item.normalized_path.exists():
        normalized = _load_tile_image(item.normalized_path, tile_width - 10, 44)
        sheet.paste(normalized, (x + (tile_width - normalized.width) // 2, y + 52))
    meta = f"{item.cell_id[-10:]}"
    row = "" if item.page_no is None or item.row_index is None else f" p{item.page_no} r{item.row_index}"
    draw.text((x + 4, y + tile_height - 44), meta + row, fill=(0, 0, 0))
    dist = f"d={item.distance:.3f}"
    if item.probability is not None:
        dist += f" p={item.probability:.2f}"
    draw.text((x + 4, y + tile_height - 26), dist, fill=(40, 40, 40))


def _load_tile_image(path: Path, max_width: int, max_height: int) -> Image.Image:
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return Image.merge("RGB", (img, img, img))


def _farthest_first(features: np.ndarray, medoid_index: int, limit: int) -> list[int]:
    n = len(features)
    if n == 0:
        return []
    selected = [int(medoid_index)]
    remaining = set(range(n))
    remaining.discard(int(medoid_index))
    while remaining and len(selected) < limit:
        selected_features = features[selected]
        best_idx = None
        best_distance = -1.0
        for idx in remaining:
            distances = 1.0 - np.clip(selected_features @ features[idx], -1.0, 1.0)
            min_distance = float(np.min(distances))
            if min_distance > best_distance:
                best_distance = min_distance
                best_idx = idx
        if best_idx is None:
            break
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    return selected


def _unique(indices: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out
