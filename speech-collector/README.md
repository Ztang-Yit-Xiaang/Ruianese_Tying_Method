# 溫州話/瑞安話 ASR 語音采集平台

這是一個第一版線上多人語音采集器，用於采集溫州市區溫州話與瑞安話 ASR 訓練資料。平台包含 React + Vite 前端、FastAPI 後端、SQLite 本地資料庫、邀請碼入口、研究授權、錄音上傳、審核狀態與 JSONL/CSV manifest 導出。

## 快速啟動

後端：

```powershell
cd D:\瑞安文化研究\Ruianese\speech-collector\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\import_tasks.py --limit-per-source 250
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

前端：

```powershell
cd D:\瑞安文化研究\Ruianese\speech-collector\frontend
npm install
npm run dev
```

打開 `http://127.0.0.1:5173`。內建示例邀請碼：`DEMO-RUIAN`、`DEMO-WENZHOU`。

## GitHub Pages

`frontend` 可以作為靜態頁部署到 GitHub Pages。若後端部署在其他服務，設定前端環境變數：

```text
VITE_API_BASE_URL=https://your-backend.example.org
```

未設定時，前端會使用相對路徑 `/api`，本地開發由 Vite proxy 轉到 `http://127.0.0.1:8010`。GitHub Pages 只承載靜態頁，FastAPI/SQLite/音頻上傳後端需另外部署。

## 資料來源

任務導入腳本會從現有工作區讀取：

- `Ruianese_upload/Ruianese_Tying_Method/ruianese.jie_yong_ki.dict.yaml`
- `rime-wenzhounese/wenzhounese.character_04.dict.yaml`
- `rime-wenzhounese/wenzhounese.phrases.dict.yaml`
- `backend/data/sample_sentences.tsv`

短句可用 TSV 補充，欄位：

```text
text	romanization	type	source	priority	status
```

## API

- `POST /api/invitations/verify`：驗證邀請碼。
- `GET /api/tasks`：按邀請碼、方言點、任務類型拉取待錄任務。
- `POST /api/submissions`：上傳錄音與說話人 metadata。
- `GET /api/submissions`：查看近期投稿。
- `PATCH /api/submissions/{id}/review`：更新審核狀態。
- `GET /api/export/manifest?format=jsonl|csv`：導出已審核 ASR manifest。

## 音頻策略

瀏覽器錄音以 WebM/Opus 原始檔保存到 `backend/storage/raw`。若環境有 `ffmpeg`，後端會轉為 16kHz mono WAV 到 `backend/storage/wav`；若沒有，投稿會保留原始檔並標記 `needs_review`。

## 測試

```powershell
cd D:\瑞安文化研究\Ruianese\speech-collector\backend
python -m unittest discover -s tests
```

前端可執行：

```powershell
cd D:\瑞安文化研究\Ruianese\speech-collector\frontend
npm run build
```
