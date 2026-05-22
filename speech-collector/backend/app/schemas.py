from pydantic import BaseModel, Field


class InvitationVerifyRequest(BaseModel):
    code: str = Field(min_length=3, max_length=80)


class InvitationVerifyResponse(BaseModel):
    code: str
    label: str
    dialect_hint: str
    active: bool


class Task(BaseModel):
    id: str
    dialect: str
    type: str
    text: str
    romanization: str
    source: str
    priority: int
    status: str


class ReviewUpdate(BaseModel):
    review_status: str = Field(pattern="^(pending|approved|rejected|needs_review)$")
    reviewer_notes: str = ""


class Submission(BaseModel):
    id: str
    invite_code: str
    speaker_id: str
    task_id: str
    dialect: str
    raw_audio_path: str
    wav_audio_path: str
    duration_seconds: float
    browser_info: str
    consent_version: str
    review_status: str
    reviewer_notes: str
    created_at: str
