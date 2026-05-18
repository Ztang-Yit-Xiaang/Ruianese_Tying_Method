# OCR Backend Comparison

- Page: 059
- Grid/cell segmentation: shared PIL/numpy detector
- PaddleOCR status: available

| Backend | Cells | Rows | Hanzi cells | Missing-hanzi rows | Review cells | Suspicious cells | Low-conf cells | OK rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 230 | 22 | 6 | 16 | 128 | 1 | 70 | 0 |
| paddleocr | 230 | 22 | 20 | 3 | 63 | 1 | 13 | 0 |

## Hanzi Column Samples

| Row | Tesseract | PaddleOCR |
|---:|---|---|
| 1 | o | 汉字 |
| 2 | Fi | 万 |
| 3 | 万(万) | 万（萬） |
| 4 | oe | 丰 |
| 5 | Ca) | 丰（豐） |
| 6 | je | 井 |
| 7 | FT CBD | 开（開 |
| 8 | Jf | 夫 |
| 9 |  |  |
| 10 | K | 天 |
| 11 | 元 | 元 |
| 12 | 无(无) | 无（無） |
| 13 |  |  |
| 14 | 韦(章) | 韦（韋） |
| 15 | ee | 专（專、 耑） * |
| 16 | ae | 丐 |
| 17 | th | 廿 |
| 18 | 五 | 五 |
| 19 | (i CT) | 匝（币） |
| 20 | oR | 丐 |
| 21 | th | 卅 |
| 22 | po |  |
| 23 | 不 | 不 |
