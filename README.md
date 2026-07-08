# Ruianese Typing Method

瑞安話輸入法是一套基於 Rime 的瑞安話（浙南吳語溫州片）拼音輸入方案。方案以現有 schema 與人工校訂規則為準，參考 `ruian_pinyin_scheme_v2 (1).docx`，並參考張永愷《瑞安方言讀音字典》聲韻配合表整理可輸入音節。

Contact: [ztangyitxiaang@gmail.com](mailto:ztangyitxiaang@gmail.com)

## Features

- Rime schema for Rui'an dialect romanization
- Initial, final, and tone spelling based on the project pinyin scheme
- Syllable dictionary generated from legal initial-final pairs
- Expandable character dictionary workflow through TSV files

## Pinyin Policy

This repository uses the document scheme:

- IPA `o` is written `o`.
- IPA `ɔ` is written `oe`.
- IPA `ɛ` is written `eh`.
- IPA `uɔ` is written `uoe`.
- IPA `yɔ` is written `yoe`.
- IPA `ts` is written `z`.
- IPA `z` is written `ss`.
- IPA `ʑ` / `z̠` is written `zs`.
- The older repository note `ɔ -> o, o -> oo` is no longer the official spelling.
- Checked-tone codas `-p/-t/-k` are not required in ordinary input; tones are typed with suffix numbers `1` to `8`.

## Initials

| IPA | Code | Notes |
|---|---|---|
| p | b | unaspirated voiceless stop |
| pʰ | p | aspirated voiceless stop |
| b | bb | voiced stop |
| m | m | nasal |
| f | f | voiceless fricative |
| v | v | voiced fricative |
| t | d | unaspirated voiceless stop |
| tʰ | t | aspirated voiceless stop |
| d | dd | voiced stop |
| n | n | nasal |
| l | l | lateral |
| k | g | unaspirated voiceless stop |
| kʰ | k | aspirated voiceless stop |
| g | gg | voiced stop |
| ŋ | ng | nasal |
| h | h | voiceless fricative |
| ɦ | hh | voiced fricative |
| tɕ | j | unaspirated affricate |
| tɕʰ | q | aspirated affricate |
| dʑ | jj | voiced affricate |
| ȵ | nj | palatal nasal |
| ɕ | x | voiceless fricative |
| ts | z | unaspirated affricate |
| tsʰ | c | aspirated affricate |
| dz | zz | voiced affricate |
| s | s | voiceless fricative |
| z | ss | voiced fricative |
| z̠ / ʑ | zs | voiced fricative |

## Finals

- Simple finals: `a`, `o`, `oe`, `ae`, `e`, `eh`, `i`, `u`, `yu`
- Compound finals: `ao`, `ai`, `ou`, `ei`, `ia`, `iao`, `iou`, `ie`, `iae`, `io`, `uai`, `uo`, `uoe`, `yo`, `yue`, `yoe`
- Nasal finals: `ang`, `eng`, `ong`, `iang`, `iong`
- Special finals: `ng`, `i` for the apical vowel

## Tones

Tone numbers are typed after the syllable:

| Tone | Category |
|---|---|
| 1 | 陰平 |
| 2 | 陰上 |
| 3 | 陰去 |
| 4 | 陰入 |
| 5 | 陽平 |
| 6 | 陽上 |
| 7 | 陽去 |
| 8 | 陽入 |

Example: `boe`, `boe1`, `boe8`.

## Dictionary Generation

Run the full pipeline, including Jie Yong Ki / 張永愷 book-page OCR after page 058:

```powershell
python tools/run_all.py
```

Raw OCR text, OCR TSV confidence data, combined OCR text, and validation reports are written to `output/jie_yong_ki/book_ocr/`. Table-aware OCR writes cell-level review files to `output/jie_yong_ki/book_ocr_structured/`.

If PaddleOCR-VL raw output for page 059 already exists, `run_all.py` also builds clean VLM rows with Zhang Yongkai tone-glyph parsing under `output/jie_yong_ki/vlm_backend_trials/`.

For a fast smoke test on the first dictionary page only:

```powershell
python tools/run_all.py --limit 1 --structured-limit 1
```

`ruian_pinyin.dict.yaml` is generated from `ruian_legal_pairs.tsv`.

```powershell
python tools/generate_ruian_pinyin_dict.py
```

The generator creates one bare syllable plus eight tone-number forms for every legal initial-final pair.

## IPA Closed-Set Recognition

The Zhang Yongkai dictionary IPA cells are handled as a closed-set image recognition task, not as general OCR. The authoritative training labels are `ipa_initial`, `ipa_final`, and `tone`; Rime fields such as `rime_initial`, `rime_final`, and `romanization` are derived from the IPA layer unless a documented override is supplied.

The detailed workflow lives in `tools/ocr/README_ipa_pipeline.md`. The normal flow is:

```text
extract -> cluster -> api-label -> promote-labels -> review-labels -> apply-review -> build-labels -> train -> predict
```

OpenAI API labeling is only a weak-label assistant for cluster contact sheets. It does not perform image cropping, clustering, or final dictionary promotion. The local CNN training path uses reviewed or gold IPA-layer labels, keeps same-cluster images out of both train and validation splits, and records mapping/schema hashes in checkpoints.

API secrets should stay in environment variables or ignored local files such as `API_KEY.txt` and `API_BASE_URL.txt`. Do not copy keys into TSV files, prompts, README text, or committed output.

Character entries can be maintained in `ruianese_characters_template.tsv` with:

```text
字<TAB>code<TAB>weight
```

Then generate the character dictionary with:

```powershell
python tools/build_ruianese_dict_from_tsv.py ruianese_characters_template.tsv ruianese.character_01.dict.yaml
```

## Jie Yong Ki Workflow

Jie Yong Ki / 張永愷 review artifacts are written under `output/jie_yong_ki/`.

Seed the legal-pair review files from the curated table:

```powershell
python tools/ocr/ocr_ruian_table.py --from-official
```

Run OCR on `tools/ocr/p437.png`, `tools/ocr/p438.png`, and `tools/ocr/p439.png` after installing the optional OCR dependencies:

```powershell
python -m pip install opencv-python pytesseract
python tools/ocr/ocr_ruian_table.py
```

Run table-aware OCR on Jie Yong Ki / 張永愷 dictionary pages with the Tesseract CLI:

```powershell
python tools/ocr/ocr_jie_yong_ki_book_table.py --page 059
python tools/ocr/ocr_jie_yong_ki_book_table.py
```

The structured OCR keeps every low-confidence cell for review. Per-page files such as `page_059_cells.tsv` use `page`, `row`, `column`, `column_name`, `text`, `confidence`, `bbox`, `status`, and `note`; `dictionary_rows.tsv` reconstructs entry rows for proofreading.

Compare the current Tesseract cell OCR with a modern PaddleOCR backend on a single page:

```powershell
python tools/ocr/compare_ocr_backends.py --page 059
```

If PaddleOCR is not installed, the comparison script still writes the Tesseract baseline and reports the install command:

```powershell
python -m pip install paddleocr paddlepaddle
```

Run high-accuracy VLM trials on a single page. This is validity-first and may take a long time while models download or run:

```powershell
python tools/ocr/compare_vlm_backends.py --backend paddle --page 059
python tools/ocr/compare_vlm_backends.py --backend qwen --page 059
```

The VLM trial writes raw model output, validated row TSV files, and a comparison report under `output/jie_yong_ki/vlm_backend_trials/`.

Build the clean page-059 rows from PaddleOCR-VL's table output and the scanned IPA cell glyphs:

```powershell
python tools/ocr/build_clean_vlm_rows.py --page 059
```

The clean rows use `mandarin_pinyin` only as lookup metadata. Rui'an tone numbers come from the printed half-circle tone glyph in the `瑞安方音 / 国际音标` cell: left-lower, left-upper, right-upper, and right-lower map to 平、上、去、入; a lower bar marks 陽調, and no lower bar marks 陰調. OCR artifacts such as `2`, `²`, `₂`, and `)` are not accepted as tone numbers.

When VLM output contains OCR artifacts in the IPA field, keep the raw value in `ruian_ipa_raw_vlm` and correct it through `output/jie_yong_ki/vlm_backend_trials/ipa_overrides.tsv`. In particular, `\theta` is treated as an OCR artifact, not as a formal Zhang Yongkai IPA token. Empty hanzi cells inherit the nearest previous hanzi because they usually mark an additional reading of the same character.

Run the IPA-aware review layer after clean rows are generated:

```powershell
python tools/ocr/review_ipa_with_llm.py --clean-rows output/jie_yong_ki/vlm_backend_trials/page_059_clean_rows.tsv
```

This writes `llm_ipa_review_prompts.jsonl`, `llm_ipa_review_candidates.tsv`, and `llm_ipa_review_clusters.tsv`. The reviewer is advisory only: it never writes to `ipa_overrides.tsv` or the formal dictionaries. Human-confirmed suggestions can later be copied into `ipa_overrides.tsv`.

To use OpenAI as the reviewer, set the key in the current PowerShell session and start with a one-row smoke test:

```powershell
$env:OPENAI_API_KEY="..."
python tools/ocr/review_ipa_with_llm.py --api-smoke-test --use-openai --limit 1
python tools/ocr/review_ipa_with_llm.py --use-openai --limit 1
```

The script reads only `OPENAI_API_KEY` from the environment. Do not put API keys in TSV files, command history, README, or prompts.

Convert the existing Jie Yong Ki / 張永愷 character readings to the current pinyin policy:

```powershell
python tools/convert_jie_yong_ki_character_dict.py
```

Review `output/jie_yong_ki/unresolved_codes.tsv` before promoting converted readings into a formal Rime dictionary. The converter changes only readings that can be validated against `ruian_pinyin.dict.yaml`; uncertain readings are preserved for manual review.

Generate the formal validated Rime dictionary:

```powershell
python tools/generate_jie_yong_ki_dict.py
```

This writes `ruianese.jie_yong_ki.dict.yaml` with only `ok` and `converted` readings. Unresolved rows stay in `output/jie_yong_ki/unresolved_codes.tsv`.

## Installation

1. Install [Rime](https://rime.im/).
2. Copy `ruian_pinyin.schema.yaml` and `ruian_pinyin.dict.yaml` into your Rime config folder.
3. Deploy Rime.
4. Select `瑞安話拼音`.

## Contributing

Useful contributions include:

- Proofreading legal initial-final pairs against Jie Yong Ki / 張永愷's table
- Adding verified character readings to the TSV character dictionary
- Testing the Rime schema on different platforms

## Development Checks

Run lint and focused tests before committing pipeline changes:

```powershell
python -m ruff check tools tests
python -m pytest tests\ocr -q
```

If the default `python` command is not configured on Windows, use the installed interpreter directly, for example:

```powershell
& "C:\Path\To\Python\python.exe" -m ruff check tools tests
```

## License

MIT License. See [LICENSE](LICENSE).
