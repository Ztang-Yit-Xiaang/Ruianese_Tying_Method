import { useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  Circle,
  Database,
  Download,
  ListFilter,
  Loader2,
  Mic,
  Pause,
  Play,
  RefreshCcw,
  Send,
  Square,
  Waves
} from "lucide-react";
import {
  Dialect,
  SpeakerMeta,
  Submission,
  Task,
  TaskType,
  fetchSubmissions,
  fetchTasks,
  exportManifestUrl,
  submitRecording,
  verifyInvitation
} from "./api";

const dialectLabels: Record<string, string> = {
  ruian: "瑞安話",
  wenzhou: "溫州市區"
};

const typeLabels: Record<TaskType, string> = {
  word: "字詞",
  sentence: "短句"
};

function formatSeconds(value: number) {
  return `${value.toFixed(1)}s`;
}

export function App() {
  const [inviteCode, setInviteCode] = useState("DEMO-RUIAN");
  const [verifiedCode, setVerifiedCode] = useState("");
  const [syncState, setSyncState] = useState("尚未連線");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState("");
  const [dialect, setDialect] = useState<Dialect>("");
  const [taskType, setTaskType] = useState<TaskType | "">("");
  const [meta, setMeta] = useState<SpeakerMeta>({ region: "", ageGroup: "", gender: "", dialectPoint: "" });
  const [consent, setConsent] = useState(false);
  const [recordingState, setRecordingState] = useState<"idle" | "recording" | "ready" | "submitting">("idle");
  const [audioUrl, setAudioUrl] = useState("");
  const [duration, setDuration] = useState(0);
  const [message, setMessage] = useState("");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  const blobRef = useRef<Blob | null>(null);

  const selectedTask = useMemo(
    () => tasks.find((task) => task.id === selectedTaskId) ?? tasks[0],
    [selectedTaskId, tasks]
  );

  const progress = useMemo(() => {
    const total = tasks.length;
    const recorded = submissions.length;
    return { total, recorded, pending: Math.max(total - recorded, 0) };
  }, [tasks.length, submissions.length]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  async function verifyAndLoad() {
    setSyncState("驗證中");
    const invite = await verifyInvitation(inviteCode);
    setVerifiedCode(invite.code);
    if (invite.dialect_hint === "ruian" || invite.dialect_hint === "wenzhou") {
      setDialect(invite.dialect_hint);
    }
    setSyncState("已連線");
    setMessage("邀請碼已驗證，可以開始采集。");
    await loadData(invite.code, invite.dialect_hint === "ruian" || invite.dialect_hint === "wenzhou" ? invite.dialect_hint : dialect, taskType);
  }

  async function loadData(code = verifiedCode, nextDialect = dialect, nextType = taskType) {
    if (!code) return;
    setSyncState("同步中");
    const [nextTasks, nextSubmissions] = await Promise.all([
      fetchTasks(code, nextDialect, nextType),
      fetchSubmissions()
    ]);
    setTasks(nextTasks);
    setSubmissions(nextSubmissions);
    setSelectedTaskId(nextTasks[0]?.id ?? "");
    setSyncState("已同步");
  }

  async function startRecording() {
    if (!selectedTask) return;
    setMessage("");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    chunksRef.current = [];
    recorderRef.current = recorder;
    startedAtRef.current = performance.now();

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      blobRef.current = blob;
      const nextUrl = URL.createObjectURL(blob);
      setAudioUrl((oldUrl) => {
        if (oldUrl) URL.revokeObjectURL(oldUrl);
        return nextUrl;
      });
      setDuration((performance.now() - startedAtRef.current) / 1000);
      setRecordingState("ready");
    });

    recorder.start();
    setRecordingState("recording");
  }

  function stopRecording() {
    recorderRef.current?.stop();
  }

  function resetRecording() {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    blobRef.current = null;
    setAudioUrl("");
    setDuration(0);
    setRecordingState("idle");
  }

  async function submitCurrent() {
    if (!selectedTask || !blobRef.current) return;
    if (!meta.region || !meta.ageGroup || !meta.dialectPoint) {
      setMessage("請先補全地區、年齡段和方言點。");
      return;
    }
    if (!consent) {
      setMessage("提交前需要同意研究授權。");
      return;
    }
    setRecordingState("submitting");
    const result = await submitRecording({
      inviteCode: verifiedCode,
      task: selectedTask,
      meta,
      consent,
      duration,
      browserInfo: {
        userAgent: navigator.userAgent,
        sampleRateHint: "browser-default",
        recordedAt: new Date().toISOString()
      },
      blob: blobRef.current
    });
    setMessage(result.review_status === "needs_review" ? "已提交；後端未完成 WAV 轉碼，待審核。" : "已提交，等待審核。");
    resetRecording();
    await loadData();
  }

  function updateFilter(nextDialect: Dialect, nextType: TaskType | "") {
    setDialect(nextDialect);
    setTaskType(nextType);
    void loadData(verifiedCode, nextDialect, nextType);
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Waves size={22} />
          <div>
            <strong>溫州話語音采集</strong>
            <span>ASR collection console</span>
          </div>
        </div>
        <div className="invite-box">
          <input value={inviteCode} onChange={(event) => setInviteCode(event.target.value)} aria-label="邀請碼" />
          <button onClick={() => void verifyAndLoad()}>驗證邀請碼</button>
        </div>
        <div className="sync"><Circle size={10} fill="currentColor" />{syncState}</div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-title">
            <ListFilter size={16} />
            <span>任務隊列</span>
          </div>
          <div className="segmented">
            <button className={dialect === "" ? "active" : ""} onClick={() => updateFilter("", taskType)}>全部</button>
            <button className={dialect === "ruian" ? "active" : ""} onClick={() => updateFilter("ruian", taskType)}>瑞安</button>
            <button className={dialect === "wenzhou" ? "active" : ""} onClick={() => updateFilter("wenzhou", taskType)}>溫州</button>
          </div>
          <div className="segmented">
            <button className={taskType === "" ? "active" : ""} onClick={() => updateFilter(dialect, "")}>全部</button>
            <button className={taskType === "word" ? "active" : ""} onClick={() => updateFilter(dialect, "word")}>字詞</button>
            <button className={taskType === "sentence" ? "active" : ""} onClick={() => updateFilter(dialect, "sentence")}>短句</button>
          </div>
          <div className="stats">
            <span><strong>{progress.total}</strong>待選任務</span>
            <span><strong>{progress.recorded}</strong>近期提交</span>
            <span><strong>{progress.pending}</strong>估計未錄</span>
          </div>
          <div className="task-list">
            {tasks.map((task) => (
              <button
                key={task.id}
                className={`task-row ${task.id === selectedTask?.id ? "selected" : ""}`}
                onClick={() => {
                  setSelectedTaskId(task.id);
                  resetRecording();
                }}
              >
                <span className="task-text">{task.text}</span>
                <span>{dialectLabels[task.dialect] ?? task.dialect} · {typeLabels[task.type]}</span>
              </button>
            ))}
          </div>
        </aside>

        <section className="recording-stage">
          {selectedTask ? (
            <>
              <div className="prompt-meta">
                <span>{dialectLabels[selectedTask.dialect] ?? selectedTask.dialect}</span>
                <span>{typeLabels[selectedTask.type]}</span>
                <span>{selectedTask.source}</span>
              </div>
              <h1>{selectedTask.text}</h1>
              <p className="romanization">{selectedTask.romanization || "無拼音標註；請按本地方言自然朗讀"}</p>
              <div className={`waveform ${recordingState === "recording" ? "live" : ""}`} aria-hidden="true">
                {Array.from({ length: 46 }).map((_, index) => (
                  <i key={index} style={{ height: `${18 + ((index * 17) % 44)}px` }} />
                ))}
              </div>
              <div className="transport">
                {recordingState === "recording" ? (
                  <button className="primary danger" onClick={stopRecording}><Square size={20} />停止</button>
                ) : (
                  <button className="primary" onClick={() => void startRecording()} disabled={!verifiedCode || recordingState === "submitting"}>
                    <Mic size={20} />錄音
                  </button>
                )}
                <button disabled={!audioUrl} onClick={() => document.querySelector<HTMLAudioElement>("#preview-audio")?.play()}><Play size={18} />播放</button>
                <button disabled={!audioUrl} onClick={resetRecording}><RefreshCcw size={18} />重錄</button>
                <button disabled={!audioUrl || recordingState === "submitting"} onClick={() => void submitCurrent()}>
                  {recordingState === "submitting" ? <Loader2 size={18} className="spin" /> : <Send size={18} />}提交
                </button>
              </div>
              {audioUrl && <audio id="preview-audio" controls src={audioUrl} />}
              <div className="quality-strip">
                <span><CheckCircle2 size={15} />距離麥克風穩定</span>
                <span><CheckCircle2 size={15} />完整讀完提示</span>
                <span><CheckCircle2 size={15} />避免背景人聲</span>
                <span>{duration ? `時長 ${formatSeconds(duration)}` : "等待錄音"}</span>
              </div>
              {message && <p className="message">{message}</p>}
            </>
          ) : (
            <div className="empty-state">請先驗證邀請碼並載入任務。</div>
          )}
        </section>

        <aside className="inspector">
          <div className="section-title">
            <Database size={16} />
            <span>說話人與質檢</span>
          </div>
          <label>地區<input value={meta.region} onChange={(event) => setMeta({ ...meta, region: event.target.value })} placeholder="例：瑞安市塘下" /></label>
          <label>年齡段<select value={meta.ageGroup} onChange={(event) => setMeta({ ...meta, ageGroup: event.target.value })}>
            <option value="">選擇</option>
            <option>18-29</option>
            <option>30-39</option>
            <option>40-49</option>
            <option>50-64</option>
            <option>65+</option>
          </select></label>
          <label>性別可選<select value={meta.gender} onChange={(event) => setMeta({ ...meta, gender: event.target.value })}>
            <option value="">不填</option>
            <option>女</option>
            <option>男</option>
            <option>其他/不便透露</option>
          </select></label>
          <label>方言點<input value={meta.dialectPoint} onChange={(event) => setMeta({ ...meta, dialectPoint: event.target.value })} placeholder="例：瑞安 / 鹿城" /></label>
          <label className="consent">
            <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
            我同意錄音用於方言研究、ASR 訓練；公開資料集發布前另行審核。
          </label>
          <a className="export-link" href={exportManifestUrl("jsonl")} target="_blank" rel="noreferrer">
            <Download size={16} />下載已審核 JSONL
          </a>
          <div className="submission-table">
            <div className="table-head"><span>最近提交</span><span>狀態</span><span>時長</span></div>
            {submissions.slice(0, 8).map((submission) => (
              <div className="table-row" key={submission.id}>
                <span>{submission.text}</span>
                <span className={`status ${submission.review_status}`}>{submission.review_status}</span>
                <span>{formatSeconds(submission.duration_seconds)}</span>
              </div>
            ))}
          </div>
        </aside>
      </section>
    </main>
  );
}
