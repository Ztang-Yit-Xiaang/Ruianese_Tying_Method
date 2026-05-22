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


class InvitationCreateRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=200)
    dialect_hint: str = ""
    label: str = ""
    max_uses: int = Field(default=0, ge=0, le=100000)
    expires_at: str = ""
    note: str = ""


class DictionaryEntryPatch(BaseModel):
    text: str | None = None
    reading: str | None = None
    ipa: str | None = None
    gloss: str | None = None
    entry_type: str | None = Field(default=None, pattern="^(word|sentence)$")
    dialect: str | None = None
    review_status: str | None = Field(default=None, pattern="^(pending|approved|rejected)$")
    review_note: str | None = None


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
