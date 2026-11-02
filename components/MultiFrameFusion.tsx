"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { PointCloudViewer, type PointCloudControls } from "./PointCloudViewer";
import type { ReconstructionJob, SpatialFusionProgress, SpatialFusionResult, SpatialResult } from "@/lib/types";
import styles from "./MultiFrameFusion.module.css";
import { parseFusionResponse } from "@/lib/fusion-response";

type Props = {
  job: ReconstructionJob | null;
  ply: ArrayBuffer | null;
  accessCode: string;
  onResult?: (result: SpatialResult) => void;
  floorEstimate?: SpatialResult | null;
};


export function MultiFrameFusion({ job, ply, accessCode, onResult, floorEstimate }: Props) {
  const [concept, setConcept] = useState("chair");
  const [count, setCount] = useState<number | "all">("all");
  const [progress, setProgress] = useState<SpatialFusionProgress | null>(null);
  const [phase, setPhase] = useState<"idle" | "starting" | "queued" | "completed" | "failed">("idle");
  const [query, setQuery] = useState<{ concept: string; frames: number[]; deadline: number } | null>(null);
  const [result, setResult] = useState<SpatialFusionResult | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [align, setAlign] = useState(true);
  const [showFloor, setShowFloor] = useState(false);
  const displayOverlay = useMemo(() => result
    ? { ...result, floor: result.floor ?? floorEstimate?.floor ?? null }
    : floorEstimate ?? null, [result, floorEstimate]);
  const viewer = useRef<PointCloudControls>(null);
  const generation = useRef(0);
  const callback = useRef(onResult);
  useEffect(() => { callback.current = onResult; }, [onResult]);
  const framesAvailable = job?.spatial_frames || 0;
  const ready = Boolean(ply && accessCode && job?.status === "completed" && job.spatial_ready && framesAvailable >= 2);
  const busy = phase === "starting" || phase === "queued";

  useEffect(() => {
    generation.current++;
    setPhase("idle"); setQuery(null); setResult(null); setSelected(null); setError("");
    setAlign(true); setShowFloor(false); setProgress(null);
  }, [job?.id]);

  useEffect(() => {
    if (!query || !job?.id || phase !== "queued") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        if (Date.now() > query.deadline) throw new Error("Fusion is taking too long. Check Modal logs and retry.");
        const qs = new URLSearchParams({ concept: query.concept, frame_indices: query.frames.join(",") });
        const payload = await parseFusionResponse(await fetch(`/api/jobs/${job.id}/fusion?${qs}`, {
          headers: { "x-worldpilot-key": accessCode }, cache: "no-store",
        }));
        if (cancelled) return;
        if (payload.progress) setProgress(payload.progress);
        if (payload.status === "failed") throw new Error(payload.error || "Fusion failed. Check Modal logs and retry.");
        if (payload.status === "completed" && payload.result) {
          setResult(payload.result); setSelected(payload.result.objects[0]?.object_id?.toString() ?? null);
          setPhase("completed"); setQuery(null); callback.current?.(payload.result);
        } else { timer = setTimeout(poll, 2500); }
      } catch (cause) {
        if (!cancelled) { setError(cause instanceof Error ? cause.message : "Fusion failed."); setPhase("failed"); setQuery(null); }
      }
    };
    timer = setTimeout(poll, 1200);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [query, job?.id, phase, accessCode]);

  async function run() {
    if (!ready || busy || !job?.id) return;
    const label = concept.trim();
    if (!label || label.length > 120) { setError("Enter an object name, such as chair, table, plant, or door."); return; }
    const token = generation.current;
    const n = Math.min(count === "all" ? framesAvailable : count, framesAvailable);
    const frames = Array.from({ length: n }, (_, i) => Math.round(i * (framesAvailable - 1) / (n - 1)));
    setProgress({ completed_frames: 0, total_frames: n, stage: "preparing" });
    setError(""); setPhase("starting"); setQuery(null); setResult(null); setSelected(null);
    try {
      const payload = await parseFusionResponse(await fetch(`/api/jobs/${job.id}/fusion`, {
        method: "POST", headers: { "x-worldpilot-key": accessCode, "content-type": "application/json" },
        body: JSON.stringify({ concept: label, frame_indices: frames }),
      }));
      if (generation.current !== token) return;
      if (payload.status === "failed") throw new Error(payload.error || "Could not start fusion.");
      if (payload.status === "completed" && payload.result) {
        setResult(payload.result); setSelected(payload.result.objects[0]?.object_id?.toString() ?? null);
        setPhase("completed"); callback.current?.(payload.result);
      } else {
        setQuery({ concept: label, frames, deadline: Date.now() + 125 * 60 * 1000 }); setPhase("queued");
      }
    } catch (cause) {
      if (generation.current === token) { setError(cause instanceof Error ? cause.message : "Could not start fusion."); setPhase("failed"); }
    }
  }

  return <section className={styles.panel} aria-labelledby="fusion-heading">
    <div className="card-top"><h3 className="area-label" id="fusion-heading">Semantic grounding</h3>
      <span className="spatial-kicker">SAM 3.1 + Pi3X + MULTI-FRAME FUSION</span></div>
    <p className={styles.description}>Find an object across reconstructed video frames. SAM masks are grounded in the shared
      Pi3X world, then geometrically consistent observations are merged and deduplicated.</p>
    {!ready && <p className={styles.message} role="status">
      {job?.status === "completed"
        ? "This scene needs aligned spatial data from at least two frames. Reconstruct a new video if necessary."
        : "Complete a reconstruction to analyze objects across multiple views."}
    </p>}
    <div className={styles.controls}>
      <div className={styles.prompt}>
        <label htmlFor="fusion-concept">Object to find</label>
        <input id="fusion-concept" className="spatial-input" value={concept} maxLength={120}
          placeholder="e.g. table" disabled={busy} onChange={e => setConcept(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); void run(); } }} />
        <div className="spatial-suggestions">
          {["chair", "table", "plant", "door"].map(example => <button type="button" key={example}
            disabled={busy} className={concept === example ? "is-selected" : ""}
            onClick={() => setConcept(example)}>{example}</button>)}
        </div>
      </div>
      <label className={styles.frameControl} htmlFor="fusion-frame-count">Frames to analyze
        <select id="fusion-frame-count" value={count} onChange={e => setCount(e.target.value === "all" ? "all" : Number(e.target.value))}
          disabled={busy} className="spatial-input">
          <option value="all">All reconstructed frames ({framesAvailable || "—"})</option>
          <option value={3}>3 evenly spaced frames</option><option value={5}>5 evenly spaced frames</option>
          <option value={10}>10 evenly spaced frames</option>
        </select><small>{count === "all" ? "Every saved Pi3X frame, including views between the previous samples." : "Evenly spaced across the walkthrough."}</small>
      </label>
      <div className={styles.action}>
        <button className="primary-button" disabled={!ready || busy || !concept.trim()} onClick={() => void run()} type="button">
          {busy ? "Fusing observations…" : ready ? `Analyze ${Math.min(count === "all" ? framesAvailable : count, framesAvailable)} frames` : "Analyze multiple frames"}
        </button><small>Saved analyses are reused. Each uncached frame can make two SAM calls. More frames take longer.</small>
      </div>
    </div>
    {error && <p role="alert" className={styles.message}>{error}</p>}
    {busy && <div role="status" className={styles.message}>
      <p style={{ margin: "0 0 8px" }}>{progress?.stage === "associating"
        ? "All frames analyzed. Associating objects and deduplicating points…"
        : `Analyzed ${progress?.completed_frames ?? 0} / ${progress?.total_frames ?? query?.frames.length ?? 0} frames${progress?.current_frame != null ? ` · Processing frame ${progress.current_frame + 1}` : ""}…`}</p>
      <progress aria-label="Analyzed reconstruction frames" value={progress?.completed_frames ?? 0}
        max={Math.max(1, progress?.total_frames ?? query?.frames.length ?? 1)} style={{ width: "min(100%, 440px)" }} />
    </div>}
    {result && <>
      <div className={styles.summary}>
        <strong>{result.objects.length} object hypotheses · {result.object_concept} · {result.frame_indices.length} frames</strong>
        <span>Frames {result.frame_indices.map(f => f + 1).join(", ")}</span>
      </div>
      {result.objects.length ? <div className={styles.objects}>
        {result.objects.map(obj => <button key={obj.object_id} type="button"
          aria-pressed={String(obj.object_id) === selected}
          className={`spatial-object ${String(obj.object_id) === selected ? "is-active" : ""}`}
          onClick={() => setSelected(String(obj.object_id))}>
          <span><strong>{obj.label} · {obj.object_id}</strong>
            <small>{obj.observation_count} observations · {obj.unique_sample_points.toLocaleString()} unique sampled points</small>
            <small>Seen in frames {obj.frame_indices.map(f => f + 1).join(", ")}</small>
          </span>
        </button>)}
      </div> : <p className={styles.message}>No grounded matches in these frames. Try another visible object or analyze more frames.</p>}
      <div className={styles.stats}>
        <span>Floor estimate: <strong>{displayOverlay?.floor ? `${Math.round(displayOverlay.floor.inlier_ratio * 100)}% inliers · unverified` : "Not detected"}</strong></span>
        <span>Association: <strong>shared geometry + surface overlap</strong></span>
      </div>
    </>}
    <div className="viewer-head"><h3 className="area-label">Multi-frame object map</h3>
      <span className="viewer-state">{busy ? "ANALYZING" : result ? "FUSED OBSERVATIONS" : ply ? "READY TO ANALYZE" : "AWAITING RECONSTRUCTION"}</span>
    </div>
    <div className={styles.canvas}>
      {ply ? <PointCloudViewer ref={viewer} data={ply} pointSize={.023} overlay={displayOverlay}
        selectedObjectId={selected} showFloor={showFloor} alignToFloor={align} />
        : <div className={styles.placeholder}>Your reconstructed world and fused object highlights will appear here.</div>}
    </div>
    <div className={`viewer-toolbar ${styles.toolbar}`}>
      <label className="spatial-floor-toggle"><input type="checkbox" checked={align} disabled={!displayOverlay?.floor}
        onChange={e => setAlign(e.target.checked)} />Align to estimated floor</label>
      <label className="spatial-floor-toggle"><input type="checkbox" checked={showFloor} disabled={!displayOverlay?.floor}
        onChange={e => setShowFloor(e.target.checked)} />Show floor evidence</label>
      <button className="toolbar-button" disabled={!ply || !selected} onClick={() => viewer.current?.focusSelected()} type="button">Focus on object</button>
      <button className="toolbar-button" disabled={!ply} onClick={() => viewer.current?.reset()} type="button">Reset view</button>
    </div>
    {result?.warnings.map((warning, index) => <p className={styles.message} key={index}>{warning}</p>)}
    <p className={styles.message}>Coral highlights the selected object's merged observations. Ambiguous matches remain separate.
      IDs belong to this saved object map; hidden surfaces and moving-object tracking are not inferred.</p>
  </section>;
}
