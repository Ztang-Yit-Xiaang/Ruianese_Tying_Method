import json
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .audio import convert_to_training_wav
from .admin import require_admin
from .config import CONSENT_VERSION, CORS_ORIGINS, RAW_AUDIO_DIR, WAV_AUDIO_DIR
from .db import get_conn, init_db, row_to_dict
from .invitations import generate_invitation_code, invitation_is_usable
from .manifest import manifest_rows, render_csv, render_jsonl
from .schemas import DictionaryEntryPatch, InvitationCreateRequest, InvitationVerifyRequest, ReviewUpdate
from .task_importer import stable_id

app = FastAPI(title="Wenzhounese/Ruianese ASR Collector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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
        if not invitation_is_usable(row):
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
        invite = conn.execute("SELECT * FROM invitations WHERE code = ?", (invite_code.upper(),)).fetchone()
        if not invitation_is_usable(invite):
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
        invite = conn.execute("SELECT * FROM invitations WHERE code = ?", (invite_code,)).fetchone()
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not invitation_is_usable(invite):
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
        conn.execute("UPDATE invitations SET used_count = used_count + 1 WHERE code = ?", (invite_code,))

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


@app.post("/api/admin/invitations", dependencies=[Depends(require_admin)])
def create_invitations(payload: InvitationCreateRequest) -> list[dict]:
    created: list[dict] = []
    with get_conn() as conn:
        for _ in range(payload.count):
            code = generate_invitation_code(payload.dialect_hint)
            while conn.execute("SELECT code FROM invitations WHERE code = ?", (code,)).fetchone():
                code = generate_invitation_code(payload.dialect_hint)
            conn.execute(
                """
                INSERT INTO invitations(code, label, dialect_hint, max_uses, expires_at, note)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (code, payload.label, payload.dialect_hint, payload.max_uses, payload.expires_at, payload.note),
            )
            created.append(
                {
                    "code": code,
                    "label": payload.label,
                    "dialect_hint": payload.dialect_hint,
                    "max_uses": payload.max_uses,
                    "used_count": 0,
                    "expires_at": payload.expires_at,
                    "note": payload.note,
                    "active": True,
                }
            )
    return created


@app.get("/api/admin/invitations", dependencies=[Depends(require_admin)])
def list_invitations(limit: int = Query(200, ge=1, le=1000)) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT code, label, dialect_hint, active, max_uses, used_count, expires_at, note, created_at
            FROM invitations
            ORDER BY created_at DESC, code ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


@app.get("/api/admin/dictionary-sources", dependencies=[Depends(require_admin)])
def list_dictionary_sources() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, title, author, pdf_path, dialect_scope, processing_status,
                   page_count, extractable_pages, note, updated_at
            FROM dictionary_sources
            ORDER BY title ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]


@app.get("/api/admin/dictionary-entries", dependencies=[Depends(require_admin)])
def list_dictionary_entries(
    review_status: str = "pending",
    source_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
) -> list[dict]:
    where = ["e.review_status = ?"]
    params: list[object] = [review_status]
    if source_id:
        where.append("e.source_id = ?")
        params.append(source_id)
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT e.*, s.title AS source_title
            FROM dictionary_entries e
            JOIN dictionary_sources s ON s.id = e.source_id
            WHERE {' AND '.join(where)}
            ORDER BY e.updated_at DESC, e.page ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]


@app.patch("/api/admin/dictionary-entries/{entry_id}", dependencies=[Depends(require_admin)])
def update_dictionary_entry(entry_id: str, payload: DictionaryEntryPatch) -> dict:
    allowed = ["text", "reading", "ipa", "gloss", "entry_type", "dialect", "review_status", "review_note"]
    values = payload.model_dump(exclude_unset=True)
    updates = [field for field in allowed if field in values]
    if not updates:
        raise HTTPException(status_code=400, detail="沒有可更新欄位")
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM dictionary_entries WHERE id = ?", (entry_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="字典條目不存在")
        assignments = ", ".join(f"{field} = ?" for field in updates)
        params = [values[field] or "" for field in updates] + [entry_id]
        conn.execute(
            f"UPDATE dictionary_entries SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            params,
        )
        updated = conn.execute("SELECT * FROM dictionary_entries WHERE id = ?", (entry_id,)).fetchone()
        return row_to_dict(updated)


@app.post("/api/admin/tasks/from-entry/{entry_id}", dependencies=[Depends(require_admin)])
def create_task_from_dictionary_entry(entry_id: str) -> dict:
    with get_conn() as conn:
        entry = conn.execute("SELECT * FROM dictionary_entries WHERE id = ?", (entry_id,)).fetchone()
        if not entry:
            raise HTTPException(status_code=404, detail="字典條目不存在")
        if entry["review_status"] != "approved":
            raise HTTPException(status_code=400, detail="只有 approved 條目可加入任務")
        romanization = entry["reading"] or entry["ipa"]
        task_id = stable_id(entry["dialect"], entry["entry_type"], entry["source"], entry["text"], romanization)
        conn.execute(
            """
            INSERT OR IGNORE INTO tasks(id, dialect, type, text, romanization, source, priority, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'ready')
            """,
            (task_id, entry["dialect"] or "wenzhou", entry["entry_type"], entry["text"], romanization, entry["source"], 500),
        )
        conn.execute(
            """
            UPDATE dictionary_entries
            SET task_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (task_id, entry_id),
        )
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return row_to_dict(task)
