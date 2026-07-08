# 瑞安話拼音 Schema

本文件整理 `ruian_pinyin.schema.yaml` 的拼音輸入規則。規則以現有 Rime schema 為底，參考 `ruian_pinyin_scheme_v2 (1).docx`；若文件與人工確認規則衝突，以本文件列出的人工確認規則為準。

## 規則優先級

- `o` 寫作 `o`。
- `ɔ` 寫作 `oe`。
- IPA 擦音 `z` 寫作 `ss`。
- `ʑ` / `z̠` 寫作 `zs`。
- 本規則已用於 `ruian_legal_pairs.tsv`、`ruian_pinyin.dict.yaml` 與正式字表重生流程。

## 聲母

| IPA | 輸入碼 | 說明 |
|---|---|---|
| p | b | 不送氣清塞音 |
| pʰ | p | 送氣清塞音 |
| b | bb | 濁塞音 |
| m | m | 鼻音 |
| f | f | 清擦音 |
| v | v | 濁擦音 |
| t | d | 不送氣清塞音 |
| tʰ | t | 送氣清塞音 |
| d | dd | 濁塞音 |
| n | n | 鼻音 |
| l | l | 邊音 |
| k | g | 不送氣清塞音 |
| kʰ | k | 送氣清塞音 |
| g | gg | 濁塞音 |
| ŋ | ng | 鼻音 |
| h | h | 清擦音 |
| ɦ | hh | 濁擦音 |
| tɕ | j | 不送氣清塞擦音 |
| tɕʰ | q | 送氣清塞擦音 |
| dʑ | jj | 濁塞擦音 |
| ȵ | nj | 舌面鼻音 |
| ɕ | x | 清擦音 |
| ts | z | 不送氣清塞擦音 |
| tsʰ | c | 送氣清塞擦音 |
| dz | zz | 濁塞擦音 |
| s | s | 清擦音 |
| z | ss | 濁擦音 |
| ʑ / z̠ | zs | 濁擦音 |

## 韻母

### 單韻母

| IPA | 輸入碼 | 備註 |
|---|---|---|
| a | a |  |
| o | o | 人工確認規則；不改作 `oo` |
| ɔ | oe | 人工確認規則；不改作 `o` |
| æ | ae |  |
| ə | e |  |
| ɛ | eh | 音值層保留 `eh` |
| i | i |  |
| u | u |  |
| y / ʉ | yu | 合併為 `yu` |

### 複韻母

| IPA / 來源 | 輸入碼 |
|---|---|
| au | ao |
| ai | ai |
| əʉ / ou | ou |
| ei | ei |
| ia | ia |
| iau | iao |
| iəʉ / iou | iou |
| ie | ie |
| iæ | iae |
| iɔ | io |
| uai | uai |
| uo | uo |
| uɔ | uoe |
| yo | yo |
| yɛ / yə | yue |
| yɔ | yoe |

`ruian_pinyin_scheme_v2 (1).docx` 中出現的 `ua`、`uoo`、`yuo` 可作後續擴充候選；本次正式碼表仍以 `ruian_legal_pairs.tsv` 中列出的合法聲韻配合為準。

### 鼻韻母

| IPA | 輸入碼 |
|---|---|
| aŋ | ang |
| eŋ | eng |
| oŋ / uŋ | ong |
| iaŋ | iang |
| ioŋ | iong |

### 特殊韻母

| IPA | 輸入碼 |
|---|---|
| ŋ | ng |
| ɿ | i |

## 聲調

聲調以音節尾碼數字表示。

| 調類 | 輸入碼 |
|---|---|
| 陰平 | 1 |
| 陰上 | 2 |
| 陰去 | 3 |
| 陰入 | 4 |
| 陽平 | 5 |
| 陽上 | 6 |
| 陽去 | 7 |
| 陽入 | 8 |

普通輸入使用「聲母 + 韻母 + 聲調」格式，例如 `zoe1`。入聲不強制輸入歷史 `-p/-t/-k` 韻尾。

## Rime 實作說明

- `ruian_pinyin.schema.yaml` 定義輸入方案、聲母韻母清單與 Rime engine。
- `ruian_pinyin.dict.yaml` 是目前正式音節字典，由 `ruian_legal_pairs.tsv` 重生。
- `ss` 與 `eh` 已納入正式 schema 與合法音節生成流程；字表讀音仍需逐條校對，不把所有舊 `e` 自動改成 `eh`。
