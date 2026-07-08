from __future__ import annotations

import csv
import json
import re
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from tools.ocr.ruian_ipa_pipeline.api_label import (
    _normalize_payload,
    label_clusters_with_openai,
    load_base_url,
)
from tools.ocr.ruian_ipa_pipeline.decoding import decode_ipa_pair
from tools.ocr.ruian_ipa_pipeline.hashing import mapping_hash
from tools.ocr.ruian_ipa_pipeline.inventory import Inventory, LegalPairRecord
from tools.ocr.ruian_ipa_pipeline.label_semantics import canonicalize_authoritative_label
from tools.ocr.ruian_ipa_pipeline.labels import build_cell_labels
from tools.ocr.ruian_ipa_pipeline.predict import _check_checkpoint_hashes
from tools.ocr.ruian_ipa_pipeline.review import build_review_queue
from tools.ocr.ruian_ipa_pipeline.review_suggestions import generate_review_fix_suggestions
from tools.ocr.ruian_ipa_pipeline.train import _prepare_labels_with_rejections, _split_labels
from tools.ocr.ruian_ipa_pipeline.visual_labels import infer_ipa_final, roman_from_ipa


@pytest.fixture
def inventory() -> Inventory:
    records = (
        LegalPairRecord("b", "a", human_reviewed=True, source="human_review"),
        LegalPairRecord("p", "a", attested_in_dictionary=True, source="dictionary"),
        LegalPairRecord("", "yu", phonologically_allowed=True, source="phonology"),
    )
    return Inventory(
        initials=("b", "p"),
        finals=("a", "yu"),
        tones=tuple(str(value) for value in range(1, 9)),
        legal_pairs=frozenset((record.initial, record.final) for record in records),
        legal_pair_records=records,
    )


@pytest.fixture
def work_tmp(request: pytest.FixtureRequest) -> Path:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", request.node.name)
    path = Path(".codex_tmp_pytest_runtime") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_ipa_to_rime_is_deterministic_and_rime_is_not_training_truth(inventory: Inventory) -> None:
    assert roman_from_ipa("p", "a") == ("b", "a")
    result = canonicalize_authoritative_label({"romanization": "ba1", "tone": "1"}, inventory)
    assert result.label is None
    assert result.reject_reason == "missing_ipa_initial"


def test_many_to_one_rime_final_is_not_reverse_inferred() -> None:
    assert roman_from_ipa("", "ʉ") == ("", "yu")
    assert roman_from_ipa("", "y̟u") == ("", "yu")
    assert infer_ipa_final("yu") is None


def test_api_mixed_cluster_cannot_enter_ipa_training(inventory: Inventory) -> None:
    payload = _normalize_payload(
        {
            "status": "mixed_cluster",
            "ipa_initial": None,
            "ipa_final": None,
            "tone": None,
            "confidence": 0.2,
            "notes": "boundary differs",
        },
        inventory,
    )
    payload.update({"image": "unused.png", "cluster_id": "cluster_1"})
    accepted, rejected, _, _ = _prepare_labels_with_rejections([payload], inventory, "ipa")
    assert not accepted
    assert rejected[0]["reject_reason"] == "api_status:mixed_cluster"


def test_missing_or_unknown_ipa_is_rejected(inventory: Inventory) -> None:
    rows = [
        {"ipa_initial": "p", "tone": "1", "image": "x"},
        {"ipa_initial": "unknown", "ipa_final": "a", "tone": "1", "image": "x"},
    ]
    accepted, rejected, _, _ = _prepare_labels_with_rejections(rows, inventory, "ipa")
    assert not accepted
    assert {row["reject_reason"] for row in rejected} == {"missing_ipa_final", "invalid_ipa_initial"}


def test_group_cluster_split_has_no_leakage_and_is_reproducible() -> None:
    rows = [
        {"cell_id": f"c{cluster}_{index}", "cluster_id": f"cluster_{cluster}", "page_no": index}
        for cluster in range(5)
        for index in range(2)
    ]
    first_train, first_val, first_report = _split_labels(rows, 0.4, 17, "group_cluster")
    second_train, second_val, second_report = _split_labels(rows, 0.4, 17, "group_cluster")
    assert [row["cell_id"] for row in first_train] == [row["cell_id"] for row in second_train]
    assert [row["cell_id"] for row in first_val] == [row["cell_id"] for row in second_val]
    assert first_report == second_report
    assert first_report["cluster_overlap_count"] == 0


def test_group_cluster_requires_cluster_id() -> None:
    with pytest.raises(ValueError, match="requires cluster_id"):
        _split_labels([{"cell_id": "a"}, {"cell_id": "b"}], 0.5, 17, "group_cluster")


def test_constrained_decoding_keeps_raw_and_selects_only_allowed_pair() -> None:
    decoded = decode_ipa_pair(
        np.asarray([0.6, 0.4]),
        np.asarray([0.4, 0.6]),
        ["p", "t"],
        ["a", "i"],
        {("p", "a"), ("t", "i")},
    )
    assert (decoded["raw_ipa_initial"], decoded["raw_ipa_final"]) == ("p", "i")
    assert not decoded["raw_pair_valid"]
    assert (decoded["predicted_ipa_initial"], decoded["predicted_ipa_final"]) in {("p", "a"), ("t", "i")}
    assert decoded["constraint_changed_prediction"]
    assert decoded["raw_topk_initial"] and decoded["constrained_topk_pairs"]


def test_legal_pair_statuses_are_not_boolean(inventory: Inventory) -> None:
    assert inventory.validate_romanization("ba1")["status"] == "reviewed"
    assert inventory.validate_romanization("pa1")["status"] == "attested_unreviewed"
    assert inventory.validate_romanization("yu1")["status"] == "allowed_unattested"
    assert inventory.validate_romanization("byu1")["status"] == "invalid"


def test_rime_override_requires_reason(inventory: Inventory) -> None:
    result = canonicalize_authoritative_label(
        {
            "ipa_initial": "p",
            "ipa_final": "a",
            "tone": "1",
            "rime_initial_override": "p",
            "rime_final_override": "a",
        },
        inventory,
    )
    assert result.label is None
    assert result.reject_reason == "rime_override_missing_reason"


def test_ipa_rime_conflict_is_written(work_tmp: Path, inventory: Inventory) -> None:
    manifest = work_tmp / "manifest.jsonl"
    clusters = work_tmp / "cells.jsonl"
    labels = work_tmp / "labels.jsonl"
    output = work_tmp / "cell_labels.jsonl"
    manifest.write_text("", encoding="utf-8")
    clusters.write_text("", encoding="utf-8")
    labels.write_text(
        json.dumps(
            {
                "cluster_id": "cluster_1",
                "label_status": "reviewed",
                "ipa_initial": "p",
                "ipa_final": "a",
                "tone": "1",
                "romanization": "pa1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert build_cell_labels(manifest, clusters, labels, output, inventory) == []
    conflict = (work_tmp / "label_conflicts.tsv").read_text(encoding="utf-8-sig")
    assert "provided_rime_differs_from_ipa_mapping" in conflict


def test_checkpoint_hash_mismatch_is_detected(work_tmp: Path) -> None:
    schema = work_tmp / "schema.yaml"
    legal = work_tmp / "legal.tsv"
    schema.write_text("schema", encoding="utf-8")
    legal.write_text("legal", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        _check_checkpoint_hashes(
            {"mapping_hash": "old", "schema_hash": "old", "legal_pairs_hash": "old"},
            schema_path=schema,
            legal_pairs_path=legal,
            allow_mapping_mismatch=False,
        )
    mismatches = _check_checkpoint_hashes(
        {"mapping_hash": mapping_hash(), "schema_hash": "old", "legal_pairs_hash": "old"},
        schema_path=schema,
        legal_pairs_path=legal,
        allow_mapping_mismatch=True,
    )
    assert mismatches


def test_custom_endpoint_requires_explicit_permission(work_tmp: Path, inventory: Inventory) -> None:
    manifest = work_tmp / "clusters.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="allow-custom-endpoint"):
        label_clusters_with_openai(
            manifest,
            work_tmp / "labels.jsonl",
            inventory,
            model="gpt-5.4",
            use_api=True,
            base_url="https://example.invalid/v1",
        )


def test_api_preflight_never_logs_key(
    work_tmp: Path,
    inventory: Inventory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = work_tmp / "clusters.jsonl"
    manifest.write_text("", encoding="utf-8")
    secret = "unit-test-secret-never-print"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    label_clusters_with_openai(
        manifest,
        work_tmp / "labels.jsonl",
        inventory,
        model="gpt-5.4",
        use_api=True,
    )
    captured = capsys.readouterr().out
    assert secret not in captured
    assert "API key source: OPENAI_API_KEY" in captured


def test_base_url_priority_and_scheme_validation(work_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = work_tmp / "API_BASE_URL.txt"
    file_path.write_text("https://file.example/v1\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    assert load_base_url("https://cli.example/v1", file_path) == "https://cli.example/v1"
    assert load_base_url(None, file_path) == "https://env.example/v1"
    monkeypatch.delenv("OPENAI_BASE_URL")
    assert load_base_url(None, file_path) == "https://file.example/v1"
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        load_base_url("ftp://bad.example/v1", file_path)


def test_review_suggestions_do_not_modify_review_tsv(work_tmp: Path) -> None:
    review = work_tmp / "review.tsv"
    original = "cluster_id\tcurrent_romanization\ncluster_0101\tnyi7\ncluster_0073\tkung1\n"
    review.write_text(original, encoding="utf-8")
    output = work_tmp / "review_fix_suggestions.tsv"
    rows = generate_review_fix_suggestions(review, output)
    assert len(rows) == 2
    assert review.read_text(encoding="utf-8") == original
    assert next(row for row in rows if row["cluster_id"] == "cluster_0101")["derived_rime"] == "nji7"
    assert next(row for row in rows if row["cluster_id"] == "cluster_0073")["derived_rime"] == "gong1"
    assert all(row["requires_visual_review"] for row in rows)


def test_review_queue_regeneration_preserves_human_fields(work_tmp: Path, inventory: Inventory) -> None:
    labels = work_tmp / "labels.jsonl"
    clusters = work_tmp / "clusters.jsonl"
    review_dir = work_tmp / "review"
    review_dir.mkdir()
    labels.write_text(
        json.dumps(
            {
                "cluster_id": "cluster_1",
                "label_status": "weak",
                "ipa_initial": "p",
                "ipa_final": "a",
                "tone": "1",
                "romanization": "ba1",
                "confidence": 0.8,
                "needs_review": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    clusters.write_text(json.dumps({"cluster_id": "cluster_1", "size": 1}) + "\n", encoding="utf-8")
    queue = review_dir / "review_queue.tsv"
    queue.write_text(
        "cluster_id\tdecision\tcorrect_romanization\treview_note\n"
        "cluster_1\tincorrect\tba1\thuman kept\n",
        encoding="utf-8",
    )
    build_review_queue(labels, clusters, review_dir, inventory)
    with queue.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["decision"] == "incorrect"
    assert row["correct_romanization"] == "ba1"
    assert row["review_note"] == "human kept"
    assert row["current_ipa_initial"] == "p"
    assert "correct_tone" in row
