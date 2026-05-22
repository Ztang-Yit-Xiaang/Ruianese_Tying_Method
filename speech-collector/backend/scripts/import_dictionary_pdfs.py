import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import PDF_SOURCE_PATHS
from app.db import get_conn, init_db
from app.dictionary_importer import extract_source_entries, source_specs


def upsert_source_and_entries(sample_only: bool, limit_per_page: int) -> tuple[int, int]:
    source_count = 0
    entry_count = 0
    init_db()
    with get_conn() as conn:
        for source in source_specs(PDF_SOURCE_PATHS):
            if not source.pdf_path.exists():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dictionary_sources(
                        id, title, author, pdf_path, dialect_scope, processing_status, note, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'missing', 'PDF path not found', CURRENT_TIMESTAMP)
                    """,
                    (source.id, source.title, source.author, str(source.pdf_path), source.dialect_scope),
                )
                continue

            try:
                page_count, extractable_pages, entries = extract_source_entries(
                    source,
                    sample_only=sample_only,
                    limit_per_page=limit_per_page,
                )
                if extractable_pages == 0:
                    status = "needs_ocr"
                    note = "No extractable text layer found in sampled pages; queue this source for OCR."
                else:
                    status = "sampled" if sample_only else "extracted"
                    note = f"Generated {len(entries)} pending review entries"
            except Exception as exc:
                page_count = 0
                extractable_pages = 0
                entries = []
                status = "needs_ocr"
                note = str(exc)

            conn.execute(
                """
                INSERT OR REPLACE INTO dictionary_sources(
                    id, title, author, pdf_path, dialect_scope, processing_status,
                    page_count, extractable_pages, note, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    source.id,
                    source.title,
                    source.author,
                    str(source.pdf_path),
                    source.dialect_scope,
                    status,
                    page_count,
                    extractable_pages,
                    note,
                ),
            )
            source_count += 1
            if entries:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO dictionary_entries(
                        id, source_id, text, reading, ipa, gloss, source, page,
                        entry_type, dialect, review_status, review_note
                    )
                    VALUES (
                        :id, :source_id, :text, :reading, :ipa, :gloss, :source, :page,
                        :entry_type, :dialect, :review_status, :review_note
                    )
                    """,
                    entries,
                )
                entry_count += len(entries)
    return source_count, entry_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import semi-automatic dictionary entries from local PDFs.")
    parser.add_argument("--full", action="store_true", help="Extract all pages instead of a safe sample.")
    parser.add_argument("--limit-per-page", type=int, default=80)
    args = parser.parse_args()

    sources, entries = upsert_source_and_entries(sample_only=not args.full, limit_per_page=args.limit_per_page)
    print(f"Processed {sources} dictionary sources; queued {entries} pending entries")


if __name__ == "__main__":
    main()
