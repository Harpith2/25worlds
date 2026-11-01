"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, ArrowUpRight, CheckCircle2, CircleHelp, CloudUpload, Download, FileVideo2, Focus, LoaderCircle, LockKeyhole, Maximize2, MousePointer2, RotateCcw, Video, X } from "lucide-react";
import { PointCloudViewer, type PointCloudControls } from "@/components/PointCloudViewer";
import { type ReconstructionJob, type UIMode, statuses } from "@/lib/types";

const MAX_SIZE = 60 * 1024 * 1024;
const ALLOWED = /\.(mp4|mov|m4v)$/i;

function EmptyViewerArt() {
  return <svg aria-hidden="true" viewBox="0 0 410 270" fill="none" className="empty-drawing">
    <path d="m45 176 149 64 174-79-149-64-174 79Z" stroke="lab(98.26 0 0)" strokeOpacity=".35" strokeWidth="1.5"/>
    <path d="M45 176V90L219 27l149 65v69l-174 79" stroke="lab(98.26 0 0)" strokeOpacity=".37" strokeWidth="1.5"/>
    <path d="M45 90 194 158l174-66M194 158v82M219 27l-25 131" stroke="lab(98.26 0 0)" strokeOpacity=".3" strokeWidth="1.5"/>
    <path d="m104 131 62 27 71-29-63-26-70 28Z" stroke="#C6B5F6" strokeWidth="1.6"/>
    <path d="m104 131 62 27v57l-62-27v-57ZM166 158l71-29v59l-71 27v-57Z" stroke="#C6B5F6" strokeWidth="1.4"/>
    <g fill="#C6B5F6"><circle cx="104" cy="131" r="3.5"/><circle cx="166" cy="158" r="3.5"/><circle cx="237" cy="129" r="3.5"/><circle cx="104" cy="188" r="3.5"/><circle cx="166" cy="215" r="3.5"/></g>
    <g fill="#C6B5F6" opacity=".8"><circle cx="252" cy="90" r="2.5"/><circle cx="264" cy="98" r="2.5"/><circle cx="278" cy="104" r="2.5"/><circle cx="294" cy="112" r="2.5"/><circle cx="310" cy="118" r="2.5"/><circle cx="250" cy="110" r="2.5"/><circle cx="269" cy="117" r="2.5"/><circle cx="290" cy="125" r="2.5"/></g>
  </svg>;
}

function formatBytes(bytes: number) { return `${(bytes / (1024 * 1024)).toFixed(1)} MB`; }
function apiHeaders(code: string) { return { "x-worldpilot-key": code }; }
async function decodeError(response: Response): Promise<string> {
  try { const data: { detail?: string } = await response.json(); return data.detail || `Request failed (${response.status}).`; }
  catch { return `Request failed (${response.status}).`; }
}

export function Studio() {
  const [file, setFile] = useState<File | null>(null);
  const [fileURL, setFileURL] = useState<string | null>(null);
  const [accessCode, setAccessCode] = useState("");
  const [activeCode, setActiveCode] = useState("");
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<ReconstructionJob | null>(null);
  const [mode, setMode] = useState<UIMode>("ready");
  const [error, setError] = useState("");
  const [ply, setPly] = useState<ArrayBuffer | null>(null);
  const [pointSize, setPointSize] = useState(.022);
  const [dragged, setDragged] = useState(false);
  const [selectedDuration, setSelectedDuration] = useState<number | null>(null);
  const [downloading, setDownloading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const viewerRef = useRef<PointCloudControls>(null);
  const inProgress = ["uploading", "queued", "preparing", "reconstructing", "exporting", "loading"].includes(mode);
  const stage = statuses[mode];

  useEffect(() => {
    if (!file) { setFileURL(null); return; }
    const url = URL.createObjectURL(file);
    setFileURL(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const chooseFile = useCallback((candidate: File | undefined) => {
    if (!candidate) return;
    if (!ALLOWED.test(candidate.name)) { setError("Choose an MP4, MOV, or M4V video."); return; }
    if (candidate.size > MAX_SIZE) { setError("Your video must be 60 MB or smaller."); return; }
    if (candidate.size === 0) { setError("The selected video is empty."); return; }
    setError("");
    setSelectedDuration(null);
    setFile(candidate);
  }, []);

  useEffect(() => {
    if (!jobId || !activeCode) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${jobId}`, { headers: apiHeaders(activeCode), cache: "no-store" });
        if (!response.ok) throw new Error(await decodeError(response));
        const result = await response.json() as ReconstructionJob;
        if (cancelled) return;
        setJob(result);
        if (result.status === "failed") throw new Error(result.error || result.message || "Reconstruction failed.");
        if (result.status === "completed") {
          setMode("loading");
          const points = await fetch(`/api/jobs/${jobId}/preview`, { headers: apiHeaders(activeCode), cache: "no-store" });
          if (!points.ok) throw new Error(await decodeError(points));
          const data = await points.arrayBuffer();
          if (cancelled) return;
          setPly(data);
          setMode("completed");
          return;
        }
        setMode(result.status);
        timer = setTimeout(poll, 2500);
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : "Something went wrong during reconstruction.");
          setMode("failed");
        }
      }
    };
    void poll();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [jobId, activeCode]);

  async function submit() {
    if (!file || inProgress) return;
    if (accessCode.trim().length < 10) { setError("Enter the demo access code before creating a world."); return; }
    if (selectedDuration !== null && (selectedDuration < 4 || selectedDuration > 60)) {
      setError("Choose a walkthrough between 4 and 60 seconds."); return;
    }
    setError(""); setPly(null); setJob(null); setJobId(""); setActiveCode(""); setMode("uploading");
    try {
      const body = new FormData(); body.append("video", file, file.name);
      const code = accessCode.trim();
      const response = await fetch("/api/jobs", { method: "POST", headers: apiHeaders(code), body });
      if (!response.ok) throw new Error(await decodeError(response));
      const result = await response.json() as ReconstructionJob;
      if (!result.id) throw new Error("Backend did not provide a job ID.");
      setJob(result); setActiveCode(code); setJobId(result.id); setMode(result.status);
    } catch (cause) {
      setMode("failed"); setError(cause instanceof Error ? cause.message : "Upload failed.");
    }
  }

  async function download(kind: "preview" | "full") {
    if (!jobId || !activeCode || downloading) return;
    setDownloading(true); setError("");
    try {
      const response = await fetch(`/api/jobs/${jobId}/${kind}`, { headers: apiHeaders(activeCode) });
      if (!response.ok) throw new Error(await decodeError(response));
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url; link.download = `25worlds-${jobId.slice(0, 8)}-${kind}.ply`;
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Download failed.");
    } finally { setDownloading(false); }
  }

  function resetForNew() {
    if (inProgress) return;
    setPly(null); setJob(null); setJobId(""); setActiveCode(""); setMode("ready"); setFile(null); setSelectedDuration(null); setError("");
    if (fileInput.current) fileInput.current.value = "";
  }

  return <div className="app-shell" id="top">
    <header className="site-header">
      <div className="site-width topbar">
        <a className="wordmark" href="#top" aria-label="25worlds home"><span className="wordmark-number">25</span>worlds</a>
        <nav className="nav-links" aria-label="Main navigation">
          <a href="#workspace">Start</a>
          <a href="#how-it-works">How it works</a>
        </nav>
      </div>
    </header>

    <main className="site-width">
      <section className="hero" aria-labelledby="hero-tagline">
        <p className="hero-intro">An AI agent system that can help robots learn and navigate an unseen environment with a single walkthrough camera video.</p>
        <h1 className="hero-tagline" id="hero-tagline">To lower cost, time and increase the efficiency of learning.</h1>
      </section>
      <section className="workspace-section" id="workspace">
        <div className="section-title-row"><h2 className="section-label">Start</h2></div>
        <div className="workspace-grid">
          <div className="left-column">
            <div className="upload-card">
              <div className="card-top"><h3 className="area-label">Upload video</h3></div>
              <div className="upload-content">
                <input type="file" accept="video/mp4,video/quicktime,.mp4,.mov,.m4v" ref={fileInput} onChange={(e) => chooseFile(e.target.files?.[0])} className="sr-only" aria-label="Choose a walkthrough video" disabled={inProgress} />
                {!file ? <div className={`drop-target ${dragged ? "drop-target-active" : ""}`} role="button" tabIndex={0}
                  onClick={() => fileInput.current?.click()}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.current?.click(); } }}
                  onDragOver={(e) => { e.preventDefault(); setDragged(true); }} onDragLeave={(e) => { e.preventDefault(); setDragged(false); }} onDrop={(e) => { e.preventDefault(); setDragged(false); chooseFile(e.dataTransfer.files[0]); }}>
                  <div className="upload-illustration"><CloudUpload size={33} strokeWidth={1.6}/></div>
                  <strong>Drop your world here.</strong>
                  <span>Drag a video or <u>browse your files</u></span>
                  <small>MP4 / MOV &nbsp; · &nbsp; UP TO 60 MB &nbsp; · &nbsp; 4–60 SEC</small>
                </div> : <div className="selected-video">
                  {fileURL && <video src={fileURL} controls muted playsInline preload="metadata" onLoadedMetadata={(e) => setSelectedDuration(e.currentTarget.duration)} aria-label="Selected video preview" />}
                  <div className="selected-row"><div className="file-icon"><FileVideo2 size={18}/></div><div className="selected-file-name"><strong title={file.name}>{file.name}</strong><span>{formatBytes(file.size)} {selectedDuration !== null && Number.isFinite(selectedDuration) ? `· ${selectedDuration.toFixed(1)} sec` : ""}</span></div><button className="small-outline" onClick={() => fileInput.current?.click()} type="button" disabled={inProgress}>Change</button></div>
                </div>}
                <div className="tip"><span>Move slowly and capture overlapping views. Even lighting and visible surfaces work best.</span></div>
                <div className="auth-form"><label htmlFor="access">Demo access code <span>· cloud GPU</span></label><div className="input-wrap"><LockKeyhole size={17}/><input id="access" type="password" value={accessCode} onChange={(e) => setAccessCode(e.target.value)} placeholder="Enter your access code" autoComplete="off" disabled={inProgress}/></div>
                  <button className="primary-button" type="button" onClick={submit} disabled={!file || inProgress}>
                    {inProgress ? <><LoaderCircle className="spin" size={18}/> Reconstructing…</> : <>Reconstruct my world <ArrowUpRight size={19}/></>}
                  </button>
                </div>
                {error && <div role="alert" className="error-alert"><CircleHelp size={17}/><span>{error}</span><button aria-label="Dismiss error" type="button" onClick={() => setError("")}><X size={16}/></button></div>}
              </div>
            </div>
          </div>
          <div className="viewer-card">
            <div className="viewer-head"><h3 className="area-label">Viewer</h3>{(mode === "completed" || inProgress) && <span className="viewer-state">{mode === "completed" ? "3D SCENE READY" : "PROCESSING"}</span>}</div>
            <div className="viewer-stage">
              {ply && mode === "completed" ? <PointCloudViewer ref={viewerRef} data={ply} pointSize={pointSize}/> : inProgress ? <div className="viewer-loader"><div className="loader-rings"><span/><span/><span/></div><h3>{stage.title}</h3><p>{job?.message || stage.description}</p><span className="viewer-caption">This is real GPU processing, not a simulated progress bar.</span></div> : <div className="viewer-empty"><EmptyViewerArt/><p>Your 3D reconstruction appears here, ready to explore from every angle.</p><div className="viewer-corner-note">DRAG TO ROTATE WHEN READY ↗</div></div>}
            </div>
            <div className="viewer-toolbar">
              <div className="viewer-toolbar-text"><MousePointer2 size={15}/><span>{mode === "completed" ? `${(job?.preview_points || 0).toLocaleString()} points in preview` : "Your 3D canvas"}</span></div>
              <div className="viewer-actions">
                {mode === "completed" && <><label className="point-size-control">Point size <input type="range" min="0.008" max="0.055" step="0.002" value={pointSize} onChange={(e) => setPointSize(Number(e.target.value))}/></label><button className="toolbar-button" onClick={() => viewerRef.current?.reset()} type="button" title="Reset camera position and orientation"><RotateCcw size={15}/><span>Reset</span></button></>}
                <span className="toolbar-tip"><Maximize2 size={14}/> {mode === "completed" ? "Drag · Zoom" : "Interactive 3D"}</span>
              </div>
            </div>
            {mode === "completed" && job && <div className="result-strip"><div><div className="result-overline"><CheckCircle2 size={15}/> YOUR RECONSTRUCTION IS READY</div><strong>{(job.total_points || 0).toLocaleString()} points in your world</strong><span>{job.frame_count ?? "Multiple"} frames · Browser preview optimized for performance</span></div><div className="result-buttons"><button type="button" onClick={() => download("preview")} disabled={downloading}>Preview PLY <Download size={16}/></button><button type="button" className="result-primary" onClick={() => download("full")} disabled={downloading}>Full PLY <Download size={16}/></button></div></div>}
          </div>
        </div>
        {mode === "completed" && <div className="new-project-row"><span>Want to try a different environment?</span><button type="button" onClick={resetForNew}>Start a new reconstruction <ArrowRight size={16}/></button></div>}
      </section>
      <section id="how-it-works" className="how-section"><div className="how-heading"><h2 className="section-label">How it works</h2></div><div className="how-grid">
        <div className="how-card"><span className="how-number">01</span><div className="how-icon how-icon-accent"><Video size={30} strokeWidth={1.6}/></div><h3>Capture a space.</h3><p>Record a short walkthrough with overlapping frames. No special scanner or depth camera needed.</p></div>
        <div className="how-card"><span className="how-number">02</span><div className="how-icon how-icon-accent"><Focus size={30} strokeWidth={1.6}/></div><h3>Reconstruct your space.</h3><p>Our Modal GPU worker estimates 3D geometry from the video using a reconstruction algorithm.</p></div>
        <div className="how-card"><span className="how-number">03</span><div className="how-icon how-icon-accent"><MousePointer2 size={30} strokeWidth={1.6}/></div><h3>Explore your world.</h3><p>Orbit around an interactive colored point cloud and download the reconstruction.</p></div>
      </div></section>
    </main>
    <footer className="footer" aria-hidden="true"><div className="site-width footer-content" /></footer>
  </div>;
}
