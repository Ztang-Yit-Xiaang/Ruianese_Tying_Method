# 瑞安話 IPA 格辨識流水線

這套工具把《瑞安方言讀音字典》正文中的「國際音標」欄當作封閉集合分類問題處理，不再讓通用 OCR 逐字符讀 IPA。

注意：權威真值是 `ipa_initial` / `ipa_final` / `tone`。`rime_initial` / `rime_final` / `rime_syllable` 由 IPA 正向映射；舊欄位 `initial` / `final` / `romanization` 只是相容 alias。比如 IPA 韻母 `əʉ` 在 Rime 層標為 `ou`，但圖像層仍保存 `ipa_final=əʉ`。production 流程不會從 Rime 反推 IPA 訓練真值。

核心分工：

- OpenCV：定位正文表格、裁出 `hanzi_crop`、`ipa_crop`、`tone_crop`。
- 聚類：把相似 IPA 格放在同一 cluster，只標代表樣本。
- OpenAI API：只做弱標註與疑難複核，不直接入庫。
- 人工：確認 cluster label，標成 `reviewed` 或 `gold`。
- CNN：用確認後標籤訓練本地多頭分類器，批量辨識全書。

## 1. 裁正文 IPA 格

```powershell
python tools\ocr\ruian_ipa_pipeline.py extract `
  --pages-dir "D:\瑞安文化研究\Ruianese\張永愷書之圖" `
  --page-start 80 `
  --page-end 120 `
  --out output\ipa_closed_set
```

主要輸出：

- `output/ipa_closed_set/ipa_cells_manifest.jsonl`
- `output/ipa_closed_set/crops/<cell_id>/hanzi_raw.png`
- `output/ipa_closed_set/crops/<cell_id>/ipa_raw.png`
- `output/ipa_closed_set/crops/<cell_id>/ipa_clean.png`
- `output/ipa_closed_set/crops/<cell_id>/tone_clean.png`

正文表格預設欄位為：

- `--hanzi-col 1`
- `--ipa-col 7`
- `--header-rows 1`

如果某些頁面裁錯，先調這三個參數，不要急著改模型。

## 2. 聚類與 contact sheet

```powershell
python tools\ocr\ruian_ipa_pipeline.py cluster `
  --manifest output\ipa_closed_set\ipa_cells_manifest.jsonl `
  --out output\ipa_closed_set\clusters `
  --distance-threshold 0.22
```

主要輸出：

- `output/ipa_closed_set/clusters/cell_clusters.jsonl`
- `output/ipa_closed_set/clusters/cluster_manifest.jsonl`
- `output/ipa_closed_set/clusters/contact_sheets/*.png`

若 cluster 太碎，逐步提高 `--distance-threshold`；若 cluster 混雜，降低它。第一輪寧可稍微碎一點，避免錯標籤大規模傳播。

v2 聚類建議先用 HOG + 橫向畫布：

```powershell
python tools\ocr\ruian_ipa_pipeline.py cluster `
  --manifest output\ipa_closed_set\ipa_cells_manifest.jsonl `
  --out output\ipa_closed_set\clusters_v2 `
  --task ipa_body `
  --feature-type hog `
  --canvas-width 256 `
  --canvas-height 64 `
  --alignment baseline `
  --method agglomerative `
  --linkage complete `
  --distance-threshold 0.18
```

`ipa_body` 會裁掉主體空白、保持長寬比，放入橫向白底畫布；`tone_spatial` 則使用完整 IPA cell，不會把聲調墨跡獨立 bbox crop 後重新置中，因為聲調可能在左上、右上、左下或右下。

```powershell
python tools\ocr\ruian_ipa_pipeline.py cluster `
  --manifest output\ipa_closed_set\ipa_cells_manifest.jsonl `
  --out output\ipa_closed_set\tone_features `
  --task tone_spatial `
  --image-key ipa_clean `
  --feature-type pixel_pca `
  --no-pca
```

v2 會額外輸出：

- `rejected_cells.jsonl`：空白、污點、疑似殘線等低品質 crop，不刪原圖。
- `metadata.json`：feature、normalization、PCA、clustering 參數，方便重現。
- `normalized_images/*.png`：模型實際看到的標準化圖。
- `cluster_xxxx_core.png`、`cluster_xxxx_boundary.png`、`cluster_xxxx_overview.png`：核心、邊界與混合概覽 contact sheet。

threshold sweep 只產生診斷，不自動選最佳值：

```powershell
python -m tools.ocr.ruian_ipa_pipeline.threshold_sweep `
  --manifest output\ipa_closed_set\ipa_cells_manifest.jsonl `
  --output-dir output\ipa_closed_set\threshold_sweep `
  --thresholds 0.12:0.34:0.02
```

## 3. API 弱標註

不加 `--use-api` 時只產生模板，不會呼叫 API：

```powershell
python tools\ocr\ruian_ipa_pipeline.py api-label `
  --cluster-manifest output\ipa_closed_set\clusters\cluster_manifest.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\cluster_labels.jsonl `
  --limit 20
```

確認要呼叫 OpenAI API 時：

```powershell
$env:OPENAI_API_KEY="..."
python tools\ocr\ruian_ipa_pipeline.py api-label `
  --cluster-manifest output\ipa_closed_set\clusters\cluster_manifest.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\cluster_labels.jsonl `
  --model gpt-5.4 `
  --limit 20 `
  --use-api
```

若沒有設定 `OPENAI_API_KEY`，工具會讀 repo 根目錄的 `API_KEY.txt`。base URL 優先級是 CLI > `OPENAI_BASE_URL` > `API_BASE_URL.txt` > 官方 endpoint。非官方 endpoint 必須加 `--allow-custom-endpoint`；建議使用 `CUSTOM_OPENAI_API_KEY`，避免把官方 key 靜默傳給第三方。只有明確加 `--allow-official-key-to-custom-endpoint` 才能把 `OPENAI_API_KEY` 傳給自訂地址。工具只列印 host 與 key 來源，不列印 secret。

API 輸出格式：

```json
{
  "cluster_id": "cluster_0017",
  "status": "labeled",
  "ipa_initial": "tsʰ",
  "ipa_final": "əʉ",
  "tone": 6,
  "confidence": 0.94,
  "notes": "Clear repeated syllable.",
  "label_status": "weak"
}
```

API 可返回 `uncertain`、`mixed_cluster`、`needs_split`、`unreadable`、`insufficient_evidence`；這些狀態不會 promotion 或傳播。Rime 欄位由程式在 IPA 驗證後自動派生。

`weak` 不能直接用來訓練或入庫。人工檢查 contact sheet 後，把可靠標籤改成：

- `reviewed`：可用於訓練。
- `gold`：高置信金標，優先放進驗證集/測試集。

也可以先用安全條件批量提升：只有 OpenAI 來源、合法聲韻配合、`needs_review=false` 且信心達標的項目會變成 `reviewed`。

```powershell
python tools\ocr\ruian_ipa_pipeline.py promote-labels `
  --labels output\ipa_closed_set\cluster_labels.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\cluster_labels_promoted.jsonl `
  --confidence-min 0.95
```

檢查標籤狀態統計：

```powershell
python tools\ocr\ruian_ipa_pipeline.py summarize-labels `
  --labels output\ipa_closed_set\cluster_labels_promoted.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv
```

## 4. 人工複核 weak 標籤

安全提升後仍然是 `weak` 的 cluster 需要人工處理。先生成複核包：

```powershell
python tools\ocr\ruian_ipa_pipeline.py review-labels `
  --labels output\ipa_closed_set\cluster_labels_promoted.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --cluster-manifest output\ipa_closed_set\clusters\cluster_manifest.jsonl `
  --out-dir output\ipa_closed_set\review
```

輸出：

- `review/review_queue.tsv`
- `review/review_report.md`
- `review/review_contact_sheets/*.png`

你只需要編輯 `review_queue.tsv` 這幾欄：

- `decision`
- `correct_romanization`
- `correct_ipa_initial`
- `correct_ipa_final`
- `correct_tone`
- `rime_initial_override` / `rime_final_override` / `rime_override_reason`（只用於真正例外）
- `review_note`

`decision` 只允許：

- `accept`：目前拼法正確，升為 `reviewed`。
- `incorrect`：修正 IPA initial/final/tone；Rime 自動派生。只填 `correct_romanization` 不能建立 IPA 真值。
- `gold`：非常確定，升為 `gold`。
- `reject`：不確定或看不清，保留非訓練狀態。
- `mixed`：同一 cluster 混了不同音節，不整簇傳播。

人工填完後套用決策：

```powershell
python tools\ocr\ruian_ipa_pipeline.py apply-review `
  --labels output\ipa_closed_set\cluster_labels_promoted.jsonl `
  --review output\ipa_closed_set\review\review_queue.tsv `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\cluster_labels_reviewed.jsonl
```

## 5. 傳播確認標籤

```powershell
python tools\ocr\ruian_ipa_pipeline.py build-labels `
  --manifest output\ipa_closed_set\ipa_cells_manifest.jsonl `
  --cell-clusters output\ipa_closed_set\clusters\cell_clusters.jsonl `
  --cluster-labels output\ipa_closed_set\cluster_labels_reviewed.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\cell_labels.jsonl
```

只有完整 IPA + tone 的 `reviewed` 和 `gold` 會傳播。衝突寫入 `label_conflicts.tsv/jsonl`，其他拒絕原因寫入 `label_rejected.jsonl`，不會靜默猜測。

## 6. 訓練 CNN

第一輪建議用 ResNet18：

```powershell
python tools\ocr\ruian_ipa_pipeline.py train `
  --labels output\ipa_closed_set\cell_labels.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\models `
  --arch resnet18 `
  --class-space ipa `
  --split-mode group_cluster `
  --epochs 12
```

第二輪再試 ConvNeXt-Tiny：

```powershell
python tools\ocr\ruian_ipa_pipeline.py train `
  --labels output\ipa_closed_set\cell_labels.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\models_convnext `
  --arch convnext_tiny `
  --epochs 12
```

訓練輸出：

- `*_best.pt`
- `train_history.jsonl`
- `train_summary.json`
- `training_rejected_labels.tsv/jsonl`
- `class_coverage.tsv/json`

`--class-space ipa` 是預設值，CNN 學 `ipa_initial` / `ipa_final` / `tone`。預設 `group_cluster` 確保同一 cluster 不跨 train/validation；另可明確選 `group_page` 或實驗用 `random_image`。少於正常門檻的 smoke run 必須顯式加 `--allow-small-dataset`。

## 7. 批量預測

```powershell
python tools\ocr\ruian_ipa_pipeline.py predict `
  --checkpoint output\ipa_closed_set\models\resnet18_best.pt `
  --manifest output\ipa_closed_set\ipa_cells_manifest.jsonl `
  --schema ruian_pinyin.schema.yaml `
  --legal-pairs ruian_legal_pairs.tsv `
  --out output\ipa_closed_set\ipa_predictions.jsonl
```

預測結果中：

- `needs_review=false`：高信心且合法。
- `needs_review=true`：低信心、非法拼寫或需人工/API 複核。
- `ipa_initial` / `ipa_final`：模型看到的 IPA 圖像層類別。
- `raw_ipa_initial` / `raw_ipa_final`：兩個 head 各自 argmax 的原始結果。
- `predicted_ipa_initial` / `predicted_ipa_final`：合法 IPA pair 約束後結果。
- `constraint_changed_prediction` 與 top-k 欄位：顯示 constraint 是否改寫模型答案。
- `mapped_rime_*` 以及相容 alias：規則映射後的 Rime 輸入碼。

checkpoint 保存 mapping/schema/legal-pairs/label manifest hash。predict 遇到不一致會預設報錯；只有明確加 `--allow-mapping-mismatch` 才會繼續，且每條輸出都標記 mismatch。

## Legal-pair 狀態

validator 不再只回傳 true/false，而是回傳：

- `invalid`：不能解析或沒有任何規則/字典證據。
- `allowed_unattested`：音系上允許，但尚未在字典中確認。
- `attested_unreviewed`：字典音節表有記錄，尚未人工複核。
- `reviewed`：已有獨立人工複核證據。
- `observed_training`：在目前訓練資料中觀測到；這不會反過來充當完整音系證據。

## 標定原則

標音節，不標字符。也就是標：

```json
{
  "ipa_initial": "tsʰ",
  "ipa_final": "əʉ",
  "tone": 6
}
```

正常派生結果為 `rime_initial=c`、`rime_final=ou`、`rime_syllable=cou6`，人工不必重複維護兩套答案。每條資料同時保存 `ipa_label_source`、`rime_label_source`、`tone_label_source`。

不要標成逐字符 OCR 結果。

每條標籤保留：

- 頁碼與行號；
- crop 路徑；
- cluster id；
- `weak` / `reviewed` / `gold`；
- 是否由 cluster 傳播。

聲調是重點錯誤來源，所以每個 cell 同時保存 `ipa_clean.png` 和 `tone_clean.png`。如果後面發現 tone head 表現不夠，可以用 `tone_clean.png` 再訓練獨立小 CNN。
