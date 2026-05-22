import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .audio import convert_to_training_wav
from .config import CONSENT_VERSION, RAW_AUDIO_DIR, WAV_AUDIO_DIR
from .db import get_conn, init_db, row_to_dict
from .manifest import manifest_rows, render_csv, render_jsonl
from .schemas import InvitationVerifyRequest, ReviewUpdate

app = FastAPI(title="Wenzhounese/Ruianese ASR Collector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "consent_version": CONSENT_VERSION}


@app.post("/api/invitations/verify")
def verify_invitation(payload: InvitationVerifyRequest) -> dict:
    code = payload.code.strip().upper()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM invitations WHERE code = ?", (code,)).fetchone()
        if not row or not row["active"]:
            raise HTTPException(status_code=404, detail="邀請碼不存在或已停用")
        return {
            "code": row["code"],
            "label": row["label"],
            "dialect_hint": row["dialect_hint"],
            "active": bool(row["active"]),
        }


@app.get("/api/tasks")
def list_tasks(
    invite_code: str = Query(...),
    dialect: str | None = None,
    type: str | None = None,
    status: str = "ready",
    limit: int = Query(50, ge=1, le=200),
) -> list[dict]:
    with get_conn() as conn:
        invite = conn.execute("SELECT code FROM invitations WHERE code = ? AND active = 1", (invite_code.upper(),)).fetchone()
        if not invite:
            raise HTTPException(status_code=403, detail="邀請碼無效")

        where = ["status = ?"]
        params: list[object] = [status]
        if dialect:
            where.append("dialect = ?")
            params.append(dialect)
        if type:
            where.append("type = ?")
            params.append(type)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, dialect, type, text, romanization, source, priority, status
            FROM tasks
            WHERE {' AND '.join(where)}
            ORDER BY priority ASC, text ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]


@app.post("/api/submissions")
def create_submission(
    invite_code: str = Form(...),
    task_id: str = Form(...),
    region: str = Form(...),
    age_group: str = Form(...),
    dialect_point: str = Form(...),
    gender: str = Form(""),
    consent: bool = Form(...),
    duration_seconds: float = Form(0),
    browser_info: str = Form("{}"),
    audio: UploadFile = File(...),
) -> dict:
    if not consent:
        raise HTTPException(status_code=400, detail="提交前必須同意研究授權")

    invite_code = invite_code.strip().upper()
    submission_id = uuid.uuid4().hex
    speaker_seed = f"{invite_code}|{region}|{age_group}|{gender}|{dialect_point}"
    speaker_id = uuid.uuid5(uuid.NAMESPACE_URL, speaker_seed).hex[:16]

    with get_conn() as conn:
        invite = conn.execute("SELECT code FROM invitations WHERE code = ? AND active = 1", (invite_code,)).fetchone()
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not invite:
            raise HTTPException(status_code=403, detail="邀請碼無效")
        if not task:
            raise HTTPException(status_code=404, detail="任務不存在")

        raw_suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
        raw_path = RAW_AUDIO_DIR / f"{submission_id}{raw_suffix}"
        with raw_path.open("wb") as handle:
            shutil.copyfileobj(audio.file, handle)

        wav_path = WAV_AUDIO_DIR / f"{submission_id}.wav"
        converted = convert_to_training_wav(raw_path, wav_path)
        review_status = "pending" if converted else "needs_review"

        conn.execute(
            """
            INSERT OR REPLACE INTO speakers(
                id, invite_code, region, age_group, gender, dialect_point, consent_version, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (speaker_id, invite_code, region, age_group, gender, dialect_point, CONSENT_VERSION),
        )
        conn.execute(
            """
            INSERT INTO submissions(
                id, invite_code, speaker_id, task_id, dialect, raw_audio_path, wav_audio_path,
                duration_seconds, browser_info, consent_version, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                submission_id,
                invite_code,
                speaker_id,
                task_id,
                task["dialect"],
                str(raw_path),
                str(wav_path) if converted else "",
                duration_seconds,
                browser_info,
                CONSENT_VERSION,
                review_status,
            ),
        )

    return {
        "id": submission_id,
        "speaker_id": speaker_id,
        "review_status": review_status,
        "raw_audio_path": str(raw_path),
        "wav_audio_path": str(wav_path) if converted else "",
    }


@app.get("/api/submissions")
def list_submissions(review_status: str | None = None, limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    with get_conn() as conn:
        params: list[object] = []
        where = ""
        if review_status:
            where = "WHERE review_status = ?"
            params.append(review_status)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT s.*, t.text
            FROM submissions s
            JOIN tasks t ON t.id = s.task_id
            {where}
            ORDER BY s.created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]


@app.patch("/api/submissions/{submission_id}/review")
def update_review(submission_id: str, payload: ReviewUpdate) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="投稿不存在")
        conn.execute(
            """
            UPDATE submissions
            SET review_status = ?, reviewer_notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (payload.review_status, payload.reviewer_notes, submission_id),
        )
        updated = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        return row_to_dict(updated)


@app.get("/api/export/manifest")
def export_manifest(format: str = "jsonl", include_review: bool = False) -> Response:
    with get_conn() as conn:
        rows = manifest_rows(conn, include_review=include_review)
    if format == "csv":
        return Response(render_csv(rows), media_type="text/csv; charset=utf-8")
    if format == "jsonl":
        return Response(render_jsonl(rows), media_type="application/x-ndjson; charset=utf-8")
    raise HTTPException(status_code=400, detail="format 必須是 jsonl 或 csv")
