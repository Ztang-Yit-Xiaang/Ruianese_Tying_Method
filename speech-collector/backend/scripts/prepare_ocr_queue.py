import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import PDF_SOURCE_PATHS, STORAGE_DIR
from app.db import get_conn, init_db
from app.dictionary_importer import source_specs


def sample_page_indexes(page_count: int) -> list[int]:
    indexes = {0, 1, 2, max(page_count // 2 - 1, 0), page_count // 2, min(page_count // 2 + 1, page_count - 1), page_count - 1}
    return sorted(index for index in indexes if 0 <= index < page_count)


def render_queue(dpi: int) -> int:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("請先安裝 pymupdf：pip install pymupdf") from exc

    init_db()
    rendered = 0
    queue_root = STORAGE_DIR / "ocr_queue"
    queue_root.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        for source in source_specs(PDF_SOURCE_PATHS):
            if not source.pdf_path.exists():
                continue
            doc = fitz.open(str(source.pdf_path))
            source_dir = queue_root / source.id
            source_dir.mkdir(parents=True, exist_ok=True)
            for index in sample_page_indexes(len(doc)):
                page_no = index + 1
                out_path = source_dir / f"page_{page_no:04d}.png"
                if not out_path.exists():
                    pix = doc[index].get_pixmap(dpi=dpi, alpha=False)
                    pix.save(out_path)
                    rendered += 1
            doc.close()
            conn.execute(
                """
                UPDATE dictionary_sources
                SET processing_status = 'needs_ocr',
                    note = note || '; OCR sample images prepared under storage/ocr_queue',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (source.id,),
            )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sampled PDF pages into an OCR review queue.")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    count = render_queue(args.dpi)
    print(f"Rendered {count} OCR queue images")


if __name__ == "__main__":
    main()
