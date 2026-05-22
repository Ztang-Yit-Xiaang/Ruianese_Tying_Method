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
