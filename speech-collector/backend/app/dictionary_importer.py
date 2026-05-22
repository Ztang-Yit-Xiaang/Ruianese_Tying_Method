import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .task_importer import stable_id


CJK_RE = re.compile(r"[\u3400-\u9fff]")
IPA_HINT_RE = re.compile(r"[ɑɒɔəɛɕɦɨɯɲŋœøʂʐʑʔʰˀː˥˦˧˨˩]")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9ʰŋɕɦɛɔəœøː˥˦˧˨˩'\-]*")


@dataclass(frozen=True)
class DictionarySourceSpec:
    title: str
    author: str
    pdf_path: Path
    dialect_scope: str

    @property
    def id(self) -> str:
        return stable_id(self.title, self.author, self.pdf_path)


def source_specs(paths: Iterable[Path]) -> list[DictionarySourceSpec]:
    specs: list[DictionarySourceSpec] = []
    for path in paths:
        name = path.stem
        if "瑞安" in name:
            dialect = "ruian"
        else:
            dialect = "wenzhou"
        author = ""
        if "郑张尚芳" in name:
            author = "郑张尚芳"
        elif "游汝杰" in name:
            author = "游汝杰, 杨乾明"
        elif "李荣" in name:
            author = "李荣"
        elif "沈克成" in name:
            author = "沈克成, 何克识"
        elif "张永恺" in name:
            author = "张永恺"
        specs.append(DictionarySourceSpec(name, author, path, dialect))
    return specs


def extract_pages(path: Path, sample_only: bool = True) -> tuple[int, dict[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("請先安裝 PDF 依賴：pip install pypdf pdfplumber pymupdf") from exc

    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    if sample_only:
        indexes = {0, 1, 2, max(page_count // 2 - 1, 0), page_count // 2, min(page_count // 2 + 1, page_count - 1), page_count - 1}
        page_indexes = sorted(index for index in indexes if 0 <= index < page_count)
    else:
        page_indexes = list(range(page_count))

    pages: dict[int, str] = {}
    for index in page_indexes:
        text = reader.pages[index].extract_text() or ""
        pages[index + 1] = text
    return page_count, pages


def guess_entry_type(text: str) -> str:
    return "sentence" if len(text) > 4 else "word"


def clean_candidate_text(value: str) -> str:
    value = re.sub(r"\s+", "", value)
    return value.strip("，。；：、（）()[]【】")


def entries_from_page(source: DictionarySourceSpec, page: int, text: str, limit_per_page: int = 80) -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not CJK_RE.search(line):
            continue
        cjk_chunks = re.findall(r"[\u3400-\u9fff]{1,12}", line)
        if not cjk_chunks:
            continue
        headword = clean_candidate_text(cjk_chunks[0])
        if not headword or headword in seen or len(headword) > 12:
            continue
        seen.add(headword)

        latin = LATIN_RE.findall(line)
        reading = " ".join(latin[:3])
        ipa = " ".join(token for token in latin if IPA_HINT_RE.search(token))[:80]
        gloss = line[:180]
        entry_type = guess_entry_type(headword)
        entry_id = stable_id(source.id, page, headword, reading, gloss)
        entries.append(
            {
                "id": entry_id,
                "source_id": source.id,
                "text": headword,
                "reading": reading,
                "ipa": ipa,
                "gloss": gloss,
                "source": source.title,
                "page": page,
                "entry_type": entry_type,
                "dialect": source.dialect_scope,
                "review_status": "pending",
                "review_note": "auto-extracted; verify against PDF before publishing",
            }
        )
        if len(entries) >= limit_per_page:
            break
    return entries


def extract_source_entries(source: DictionarySourceSpec, sample_only: bool = True, limit_per_page: int = 80) -> tuple[int, int, list[dict]]:
    page_count, pages = extract_pages(source.pdf_path, sample_only=sample_only)
    entries: list[dict] = []
    extractable_pages = 0
    for page, text in pages.items():
        if text.strip():
            extractable_pages += 1
        entries.extend(entries_from_page(source, page, text, limit_per_page=limit_per_page))
    return page_count, extractable_pages, entries
