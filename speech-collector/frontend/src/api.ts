export type TaskType = "word" | "sentence";
export type Dialect = "ruian" | "wenzhou" | "";

export type Task = {
  id: string;
  dialect: string;
  type: TaskType;
  text: string;
  romanization: string;
  source: string;
  priority: number;
  status: string;
};

export type Submission = {
  id: string;
  task_id: string;
  dialect: string;
  text: string;
  review_status: string;
  duration_seconds: number;
  created_at: string;
  wav_audio_path: string;
  raw_audio_path: string;
};

export type Invitation = {
  code: string;
  label: string;
  dialect_hint: string;
  active: boolean;
  max_uses: number;
  used_count: number;
  expires_at: string;
  note: string;
  created_at?: string;
};

export type DictionarySource = {
  id: string;
  title: string;
  author: string;
  pdf_path: string;
  dialect_scope: string;
  processing_status: string;
  page_count: number;
  extractable_pages: number;
  note: string;
  updated_at: string;
};

export type DictionaryEntry = {
  id: string;
  source_id: string;
  source_title: string;
  text: string;
  reading: string;
  ipa: string;
  gloss: string;
  source: string;
  page: number;
  entry_type: TaskType;
  dialect: string;
  review_status: "pending" | "approved" | "rejected";
  review_note: string;
  task_id: string;
};

export type SpeakerMeta = {
  region: string;
  ageGroup: string;
  gender: string;
  dialectPoint: string;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function apiPath(path: string) {
  return API_BASE ? `${API_BASE}${path}` : path;
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiPath(url), options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail || "請求失敗");
  }
  return response.json() as Promise<T>;
}

export async function verifyInvitation(code: string) {
  return request<{ code: string; label: string; dialect_hint: string; active: boolean }>("/api/invitations/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code })
  });
}

export async function fetchTasks(inviteCode: string, dialect: Dialect, type: TaskType | "") {
  const params = new URLSearchParams({ invite_code: inviteCode, limit: "80" });
  if (dialect) params.set("dialect", dialect);
  if (type) params.set("type", type);
  return request<Task[]>(`/api/tasks?${params.toString()}`);
}

export function exportManifestUrl(format = "jsonl") {
  return apiPath(`/api/export/manifest?format=${encodeURIComponent(format)}`);
}

export async function fetchSubmissions() {
  return request<Submission[]>("/api/submissions?limit=20");
}

export async function submitRecording(args: {
  inviteCode: string;
  task: Task;
  meta: SpeakerMeta;
  consent: boolean;
  duration: number;
  browserInfo: Record<string, unknown>;
  blob: Blob;
}) {
  const form = new FormData();
  form.set("invite_code", args.inviteCode);
  form.set("task_id", args.task.id);
  form.set("region", args.meta.region);
  form.set("age_group", args.meta.ageGroup);
  form.set("gender", args.meta.gender);
  form.set("dialect_point", args.meta.dialectPoint);
  form.set("consent", String(args.consent));
  form.set("duration_seconds", args.duration.toFixed(3));
  form.set("browser_info", JSON.stringify(args.browserInfo));
  form.set("audio", args.blob, `recording-${args.task.id}.webm`);

  return request<{ id: string; review_status: string }>("/api/submissions", {
    method: "POST",
    body: form
  });
}

function adminHeaders(token: string) {
  return {
    "Content-Type": "application/json",
    "X-Admin-Token": token
  };
}

export async function createInvitations(token: string, payload: {
  count: number;
  dialect_hint: string;
  label: string;
  max_uses: number;
  expires_at: string;
  note: string;
}) {
  return request<Invitation[]>("/api/admin/invitations", {
    method: "POST",
    headers: adminHeaders(token),
    body: JSON.stringify(payload)
  });
}

export async function fetchInvitations(token: string) {
  return request<Invitation[]>("/api/admin/invitations", {
    headers: { "X-Admin-Token": token }
  });
}

export async function fetchDictionarySources(token: string) {
  return request<DictionarySource[]>("/api/admin/dictionary-sources", {
    headers: { "X-Admin-Token": token }
  });
}

export async function fetchDictionaryEntries(token: string, reviewStatus = "pending", sourceId = "") {
  const params = new URLSearchParams({ review_status: reviewStatus, limit: "100" });
  if (sourceId) params.set("source_id", sourceId);
  return request<DictionaryEntry[]>(`/api/admin/dictionary-entries?${params.toString()}`, {
    headers: { "X-Admin-Token": token }
  });
}

export async function updateDictionaryEntry(token: string, entryId: string, payload: Partial<DictionaryEntry>) {
  return request<DictionaryEntry>(`/api/admin/dictionary-entries/${entryId}`, {
    method: "PATCH",
    headers: adminHeaders(token),
    body: JSON.stringify(payload)
  });
}

export async function createTaskFromEntry(token: string, entryId: string) {
  return request<Task>(`/api/admin/tasks/from-entry/${entryId}`, {
    method: "POST",
    headers: { "X-Admin-Token": token }
  });
}
