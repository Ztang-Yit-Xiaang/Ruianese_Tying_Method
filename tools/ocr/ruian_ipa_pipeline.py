#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ruian_ipa_pipeline.api_label import label_clusters_with_openai
from ruian_ipa_pipeline.cluster import cluster_manifest
from ruian_ipa_pipeline.extract import ExtractConfig, extract_pages
from ruian_ipa_pipeline.inventory import load_inventory
from ruian_ipa_pipeline.label_management import promote_labels, summarize_labels
from ruian_ipa_pipeline.labels import build_cell_labels
from ruian_ipa_pipeline.predict import predict_manifest
from ruian_ipa_pipeline.review import apply_review_decisions, build_review_queue
from ruian_ipa_pipeline.review_suggestions import generate_review_fix_suggestions
from ruian_ipa_pipeline.train import TrainConfig, train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Rui'an IPA-cell closed-set recognition pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract hanzi/IPA crops from body table pages")
    extract.add_argument("--pages-dir", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--page-start", type=int)
    extract.add_argument("--page-end", type=int)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--hanzi-col", type=int, default=1)
    extract.add_argument("--ipa-col", type=int, default=7)
    extract.add_argument("--header-rows", type=int, default=1)
    extract.add_argument("--include-empty", action="store_true")
    extract.add_argument("--min-ink-ratio", type=float, default=0.001)

    cluster = sub.add_parser("cluster", help="Cluster extracted IPA crops and make contact sheets")
    cluster.add_argument("--manifest", type=Path, required=True)
    cluster.add_argument("--out", type=Path, required=True)
    cluster.add_argument("--image-key", default="ipa_clean")
    cluster.add_argument("--task", choices=["ipa_body", "tone_spatial"], default="ipa_body")
    cluster.add_argument("--feature-type", choices=["pixel_pca", "hog"], default="pixel_pca")
    cluster.add_argument("--canvas-width", type=int)
    cluster.add_argument("--canvas-height", type=int)
    cluster.add_argument("--alignment", choices=["center", "baseline"], default="center")
    cluster.add_argument("--padding", type=int, default=4)
    cluster.add_argument("--method", choices=["auto", "agglomerative", "dbscan", "hdbscan", "kmeans"], default="auto")
    cluster.add_argument("--linkage", choices=["average", "complete"], default="average")
    cluster.add_argument("--distance-threshold", type=float, default=0.22)
    cluster.add_argument("--kmeans-clusters", type=int)
    cluster.add_argument("--max-sheet-items", type=int, default=24)
    cluster.add_argument("--pca-components", type=int, default=48)
    cluster.add_argument("--no-pca", action="store_true")
    cluster.add_argument("--random-state", type=int, default=17)
    cluster.add_argument("--min-ink-pixels", type=int, default=12)
    cluster.add_argument("--min-ink-ratio", type=float, default=0.0008)
    cluster.add_argument("--max-line-like-component-ratio", type=float, default=0.85)

    api = sub.add_parser("api-label", help="Create weak cluster labels with OpenAI, or templates without --use-api")
    api.add_argument("--cluster-manifest", type=Path, required=True)
    api.add_argument("--schema", type=Path, required=True)
    api.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    api.add_argument("--out", type=Path, required=True)
    api.add_argument("--model", default="gpt-5.4")
    api.add_argument("--limit", type=int)
    api.add_argument("--use-api", action="store_true")
    api.add_argument("--api-key-file", type=Path, default=Path("API_KEY.txt"))
    api.add_argument("--base-url")
    api.add_argument("--allow-custom-endpoint", action="store_true")
    api.add_argument("--allow-official-key-to-custom-endpoint", action="store_true")

    promote = sub.add_parser("promote-labels", help="Promote safe OpenAI weak labels to reviewed")
    promote.add_argument("--labels", type=Path, required=True)
    promote.add_argument("--schema", type=Path, required=True)
    promote.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    promote.add_argument("--out", type=Path, required=True)
    promote.add_argument("--confidence-min", type=float, default=0.95)

    summarize = sub.add_parser("summarize-labels", help="Summarize label status and validity counts")
    summarize.add_argument("--labels", type=Path, required=True)
    summarize.add_argument("--schema", type=Path, required=True)
    summarize.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))

    review = sub.add_parser("review-labels", help="Build a TSV/manual review pack for weak labels")
    review.add_argument("--labels", type=Path, required=True)
    review.add_argument("--schema", type=Path, required=True)
    review.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    review.add_argument("--cluster-manifest", type=Path, required=True)
    review.add_argument("--out-dir", type=Path, required=True)
    review.add_argument("--confidence-min", type=float, default=0.95)

    apply_review = sub.add_parser("apply-review", help="Apply manual TSV review decisions to cluster labels")
    apply_review.add_argument("--labels", type=Path, required=True)
    apply_review.add_argument("--review", type=Path, required=True)
    apply_review.add_argument("--schema", type=Path, required=True)
    apply_review.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    apply_review.add_argument("--out", type=Path, required=True)

    suggestions = sub.add_parser("review-suggestions", help="Generate non-applying visual review suggestions")
    suggestions.add_argument("--review", type=Path, required=True)
    suggestions.add_argument("--out", type=Path, required=True)
    suggestions.add_argument("--cluster-manifest", type=Path)
    suggestions.add_argument("--cluster-labels", type=Path)

    labels = sub.add_parser("build-labels", help="Propagate reviewed/gold cluster labels to cells")
    labels.add_argument("--manifest", type=Path, required=True)
    labels.add_argument("--cell-clusters", type=Path, required=True)
    labels.add_argument("--cluster-labels", type=Path, required=True)
    labels.add_argument("--schema", type=Path, required=True)
    labels.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    labels.add_argument("--out", type=Path, required=True)

    train = sub.add_parser("train", help="Train a multi-head CNN classifier")
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--schema", type=Path, required=True)
    train.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--arch", choices=["resnet18", "resnet34", "convnext_tiny"], default="resnet18")
    train.add_argument("--class-space", choices=["ipa", "rime"], default="ipa")
    train.add_argument("--image-size", type=int, default=160)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--device", default=None)
    train.add_argument(
        "--split-mode",
        choices=["group_cluster", "group_page", "random_image"],
        default="group_cluster",
    )
    train.add_argument("--allow-small-dataset", action="store_true")
    train.add_argument("--no-page-split", action="store_true", help=argparse.SUPPRESS)

    predict = sub.add_parser("predict", help="Predict romanized syllables from a trained classifier")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--manifest", type=Path, required=True)
    predict.add_argument("--schema", type=Path, required=True)
    predict.add_argument("--legal-pairs", type=Path, default=Path("ruian_legal_pairs.tsv"))
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--image-size", type=int, default=160)
    predict.add_argument("--confidence-threshold", type=float, default=0.85)
    predict.add_argument("--device", default=None)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--allow-mapping-mismatch", action="store_true")

    args = parser.parse_args()
    if args.command == "extract":
        paths = sorted(args.pages_dir.glob("*.png"))
        paths = _filter_pages(paths, args.page_start, args.page_end)
        if args.limit:
            paths = paths[: args.limit]
        config = ExtractConfig(
            hanzi_col=args.hanzi_col,
            ipa_col=args.ipa_col,
            header_rows=args.header_rows,
            include_empty=args.include_empty,
            min_ink_ratio=args.min_ink_ratio,
        )
        rows = extract_pages(paths, args.out, config, root_for_paths=args.out)
        print(f"Extracted {len(rows)} IPA cell rows into {args.out}")
    elif args.command == "cluster":
        _, clusters = cluster_manifest(
            args.manifest,
            args.out,
            image_key=args.image_key,
            task=args.task,
            feature_type=args.feature_type,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            alignment=args.alignment,
            padding=args.padding,
            method=args.method,
            linkage=args.linkage,
            distance_threshold=args.distance_threshold,
            kmeans_clusters=args.kmeans_clusters,
            max_sheet_items=args.max_sheet_items,
            pca_components=args.pca_components,
            use_pca=False if args.no_pca else None,
            random_state=args.random_state,
            min_ink_pixels=args.min_ink_pixels,
            min_ink_ratio=args.min_ink_ratio,
            max_line_like_component_ratio=args.max_line_like_component_ratio,
            root_for_paths=args.out,
        )
        print(f"Wrote {len(clusters)} clusters into {args.out}")
    elif args.command == "api-label":
        inventory = load_inventory(args.schema, args.legal_pairs)
        rows = label_clusters_with_openai(
            args.cluster_manifest,
            args.out,
            inventory,
            model=args.model,
            limit=args.limit,
            use_api=args.use_api,
            api_key_file=args.api_key_file,
            base_url=args.base_url,
            allow_custom_endpoint=args.allow_custom_endpoint,
            allow_official_key_to_custom_endpoint=args.allow_official_key_to_custom_endpoint,
        )
        mode = "API labels" if args.use_api else "label templates"
        print(f"Wrote {len(rows)} {mode} to {args.out}")
    elif args.command == "promote-labels":
        inventory = load_inventory(args.schema, args.legal_pairs)
        _, summary = promote_labels(args.labels, args.out, inventory, confidence_min=args.confidence_min)
        print(
            "Promoted "
            f"{summary['promoted']} labels to reviewed; "
            f"wrote {summary['total']} labels to {args.out}"
        )
    elif args.command == "summarize-labels":
        inventory = load_inventory(args.schema, args.legal_pairs)
        summary = summarize_labels(args.labels, inventory)
        for key, value in summary.items():
            print(f"{key}: {value}")
    elif args.command == "review-labels":
        inventory = load_inventory(args.schema, args.legal_pairs)
        rows, summary = build_review_queue(
            args.labels,
            args.cluster_manifest,
            args.out_dir,
            inventory,
            confidence_min=args.confidence_min,
        )
        print(f"Wrote {len(rows)} review rows to {summary['queue']}")
        print(f"Review report: {summary['report']}")
    elif args.command == "apply-review":
        inventory = load_inventory(args.schema, args.legal_pairs)
        _, summary = apply_review_decisions(args.labels, args.review, args.out, inventory)
        print(
            f"Applied {summary['changed']} review decisions; "
            f"accepted={summary['accepted']}, incorrect={summary['incorrect']}, "
            f"gold={summary['gold']}, rejected={summary['rejected']}, mixed={summary['mixed']}; "
            f"wrote {summary['total']} labels to {args.out}"
        )
    elif args.command == "review-suggestions":
        rows = generate_review_fix_suggestions(
            args.review,
            args.out,
            cluster_manifest_path=args.cluster_manifest,
            cluster_labels_path=args.cluster_labels,
        )
        print(f"Wrote {len(rows)} non-applying review suggestions to {args.out}")
    elif args.command == "build-labels":
        inventory = load_inventory(args.schema, args.legal_pairs)
        rows = build_cell_labels(args.manifest, args.cell_clusters, args.cluster_labels, args.out, inventory)
        print(f"Wrote {len(rows)} propagated reviewed/gold cell labels to {args.out}")
    elif args.command == "train":
        inventory = load_inventory(args.schema, args.legal_pairs)
        config = TrainConfig(
            arch=args.arch,
            class_space=args.class_space,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            split_mode="random_image" if args.no_page_split else args.split_mode,
            allow_small_dataset=args.allow_small_dataset,
        )
        if args.device:
            config.device = args.device
        summary = train_classifier(
            args.labels,
            args.out,
            inventory,
            config,
            schema_path=args.schema,
            legal_pairs_path=args.legal_pairs,
        )
        print(f"Best checkpoint: {summary['best_checkpoint']}")
    elif args.command == "predict":
        inventory = load_inventory(args.schema, args.legal_pairs)
        rows = predict_manifest(
            args.checkpoint,
            args.manifest,
            args.out,
            inventory,
            image_size=args.image_size,
            confidence_threshold=args.confidence_threshold,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            schema_path=args.schema,
            legal_pairs_path=args.legal_pairs,
            allow_mapping_mismatch=args.allow_mapping_mismatch,
            top_k=args.top_k,
        )
        print(f"Wrote {len(rows)} predictions to {args.out}")


def _filter_pages(paths: list[Path], start: int | None, end: int | None) -> list[Path]:
    if start is None and end is None:
        return paths
    from ruian_ipa_pipeline.io_utils import page_number_from_path

    out: list[Path] = []
    for path in paths:
        page_no = page_number_from_path(path)
        if page_no is None:
            continue
        if start is not None and page_no < start:
            continue
        if end is not None and page_no > end:
            continue
        out.append(path)
    return out


if __name__ == "__main__":
    main()
