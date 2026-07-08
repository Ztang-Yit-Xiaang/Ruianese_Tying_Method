from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tools.ocr.ruian_ipa_pipeline.cluster import cluster_manifest
from tools.ocr.ruian_ipa_pipeline.cluster_diagnostics import diagnose_cluster
from tools.ocr.ruian_ipa_pipeline.clustering import cluster_features
from tools.ocr.ruian_ipa_pipeline.contact_sheet import ContactItem, make_contact_sheet, select_sample_indices
from tools.ocr.ruian_ipa_pipeline.image_features import (
    FeatureConfig,
    ImageQualityConfig,
    assess_image_quality,
    extract_feature,
    find_ink_bbox,
    hog_feature,
    load_grayscale_image,
    normalize_ipa_image,
)
from tools.ocr.ruian_ipa_pipeline.syllable_inventory import SyllableInventory
from tools.ocr.ruian_ipa_pipeline.threshold_sweep import run_threshold_sweep
from tools.ocr.ruian_ipa_pipeline.tone_features import extract_tone_spatial_feature


@pytest.fixture
def work_tmp(request: pytest.FixtureRequest) -> Path:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name)
    path = Path(".codex_tmp_pytest_runtime") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_same_image_same_feature_is_deterministic(work_tmp: Path) -> None:
    path = _make_cell(work_tmp / "a.png", body=True, tone="left_top")
    image = load_grayscale_image(path)
    config = FeatureConfig(feature_type="pixel_pca", canvas_width=256, canvas_height=64, alignment="baseline")
    first, _ = extract_feature(image, config)
    second, _ = extract_feature(image, config)
    assert np.array_equal(first, second)


def test_ipa_body_normalization_preserves_aspect_ratio(work_tmp: Path) -> None:
    path = work_tmp / "wide.png"
    img = Image.new("L", (220, 50), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 20, 210, 25), fill=0)
    img.save(path)
    image = load_grayscale_image(path)
    config = FeatureConfig(feature_type="pixel_pca", canvas_width=256, canvas_height=64, padding=8)
    normalized = normalize_ipa_image(image, config)
    bbox = find_ink_bbox(normalized)
    assert bbox is not None
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    assert width / height > 4.0


def test_tone_spatial_keeps_corner_position(work_tmp: Path) -> None:
    left_top = load_grayscale_image(_make_cell(work_tmp / "lt.png", body=True, tone="left_top"))
    right_bottom = load_grayscale_image(_make_cell(work_tmp / "rb.png", body=True, tone="right_bottom"))
    lt_feature, _ = extract_tone_spatial_feature(left_top)
    rb_feature, _ = extract_tone_spatial_feature(right_bottom)
    assert np.linalg.norm(lt_feature - rb_feature) > 0.1


def test_left_top_and_left_bottom_tone_features_differ(work_tmp: Path) -> None:
    left_top = load_grayscale_image(_make_cell(work_tmp / "lt.png", body=True, tone="left_top"))
    left_bottom = load_grayscale_image(_make_cell(work_tmp / "lb.png", body=True, tone="left_bottom"))
    lt_feature, _ = extract_tone_spatial_feature(left_top)
    lb_feature, _ = extract_tone_spatial_feature(left_bottom)
    assert np.linalg.norm(lt_feature - lb_feature) > 0.05


def test_blank_and_tiny_noise_are_rejected(work_tmp: Path) -> None:
    blank = Image.new("L", (80, 40), 255)
    noise = Image.new("L", (80, 40), 255)
    noise.putpixel((5, 5), 0)
    config = ImageQualityConfig(min_ink_pixels=5)
    blank_quality = assess_image_quality(blank, config)
    noise_quality = assess_image_quality(noise, config)
    assert not blank_quality["accepted"]
    assert "too_few_ink_pixels" in blank_quality["reject_reasons"]
    assert not noise_quality["accepted"]


def test_contact_sheet_small_cluster_does_not_fail(work_tmp: Path) -> None:
    path = _make_cell(work_tmp / "cell.png", body=True)
    out = work_tmp / "sheet.png"
    make_contact_sheet(
        [ContactItem("p0001_r0001", 1, 1, path, None, distance=0.0)],
        out,
    )
    assert out.exists()


def test_sample_selection_unique_when_possible() -> None:
    features = np.eye(5, dtype=np.float32)
    distances = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    samples = select_sample_indices(features, distances, medoid_index=0, limit=4, random_state=17)
    for values in samples.values():
        assert len(values) == len(set(values))


def test_cluster_diagnostics_reproducible() -> None:
    features = np.asarray([[1, 0], [0.95, 0.05], [0, 1]], dtype=np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    rows = [{"id": f"cell_{idx}", "quality": {"ink_ratio": 0.1, "bbox_width": 20, "bbox_height": 10}} for idx in range(3)]
    first = diagnose_cluster(rows, features, distance_threshold=0.5)
    second = diagnose_cluster(rows, features, distance_threshold=0.5)
    assert first.medoid_cell_id == second.medoid_cell_id
    assert np.array_equal(first.distances_to_medoid, second.distances_to_medoid)


def test_pixel_pca_cluster_mode_still_runs(work_tmp: Path) -> None:
    manifest = _write_manifest(work_tmp, absolute=False)
    out = work_tmp / "clusters"
    cell_rows, cluster_rows = cluster_manifest(
        manifest,
        out,
        feature_type="pixel_pca",
        method="agglomerative",
        distance_threshold=0.4,
    )
    assert cell_rows
    assert cluster_rows
    assert (out / "features.npz").exists()
    assert (out / "cluster_manifest.jsonl").exists()


def test_hog_fallback_runs_without_skimage_requirement(work_tmp: Path) -> None:
    image = load_grayscale_image(_make_cell(work_tmp / "hog.png", body=True))
    config = FeatureConfig(
        feature_type="hog",
        canvas_width=128,
        canvas_height=64,
        prefer_skimage_hog=False,
    )
    feature = hog_feature(normalize_ipa_image(image, config), config)
    assert feature.size > 0


def test_hdbscan_missing_does_not_break_dbscan() -> None:
    features = np.asarray([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    result = cluster_features(features, method="dbscan", distance_threshold=0.2)
    assert len(result.labels) == 3


def test_relative_and_absolute_manifest_paths_resolve(work_tmp: Path) -> None:
    rel_manifest = _write_manifest(work_tmp / "rel", absolute=False)
    abs_manifest = _write_manifest(work_tmp / "abs", absolute=True)
    for manifest in [rel_manifest, abs_manifest]:
        out = manifest.parent / "clusters"
        cell_rows, _ = cluster_manifest(manifest, out, method="agglomerative", distance_threshold=0.4)
        assert len(cell_rows) == 3


def test_threshold_sweep_writes_json_and_csv(work_tmp: Path) -> None:
    manifest = _write_manifest(work_tmp, absolute=False)
    out = work_tmp / "sweep"
    rows = run_threshold_sweep(
        manifest,
        out,
        thresholds=[0.12, 0.2],
        feature_type="pixel_pca",
        canvas_width=96,
        canvas_height=96,
        alignment="center",
    )
    assert len(rows) == 2
    assert (out / "threshold_sweep.json").exists()
    assert (out / "threshold_sweep.csv").exists()


def test_syllable_inventory_basic_interface(work_tmp: Path) -> None:
    schema = work_tmp / "schema.yaml"
    legal = work_tmp / "legal.tsv"
    schema.write_text(
        "alphabet:\n  initials:\n    - z\n  finals:\n    - a\n\ntone_marks:\n  - \"1\"\n  - \"2\"\n",
        encoding="utf-8",
    )
    legal.write_text("z\ta\ttest\n", encoding="utf-8")
    inventory = SyllableInventory.load_from_yaml(schema, legal)
    assert inventory.is_valid_syllable("za1")
    assert inventory.nearest_valid_candidates("za3")


def _make_cell(path: Path, body: bool = True, tone: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (120, 64), 255)
    draw = ImageDraw.Draw(img)
    if body:
        draw.rectangle((36, 28, 96, 36), fill=0)
        draw.rectangle((48, 18, 52, 42), fill=0)
        draw.rectangle((76, 18, 80, 42), fill=0)
    if tone:
        coords = {
            "left_top": (18, 12, 24, 18),
            "left_bottom": (18, 46, 24, 52),
            "right_top": (102, 12, 108, 18),
            "right_bottom": (102, 46, 108, 52),
        }[tone]
        draw.ellipse(coords, fill=0)
    img.save(path)
    return path


def _write_manifest(root: Path, absolute: bool) -> Path:
    crop_dir = root / "crops"
    paths = [
        _make_cell(crop_dir / "a.png", body=True, tone="left_top"),
        _make_cell(crop_dir / "b.png", body=True, tone="left_top"),
        _make_cell(crop_dir / "c.png", body=True, tone="right_bottom"),
    ]
    rows = []
    for idx, path in enumerate(paths):
        value = str(path.resolve()) if absolute else str(path.relative_to(root)).replace("\\", "/")
        rows.append(
            {
                "id": f"p0001_r{idx:04d}",
                "page_no": 1,
                "row_index": idx,
                "paths": {"ipa_clean": value},
                "quality": {"has_ipa_ink": True},
            }
        )
    manifest = root / "ipa_cells_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest
