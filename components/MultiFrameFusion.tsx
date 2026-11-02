"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { PointCloudViewer, type PointCloudControls } from "./PointCloudViewer";
import type { ReconstructionJob, SpatialFusionProgress, SpatialFusionResult, SpatialResult } from "@/lib/types";
import styles from "./MultiFrameFusion.module.css";
import { fetchFusionStatus, parseFusionResponse } from "@/lib/fusion-response";

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
  const [pollNotice, setPollNotice] = useState("");
  const [align, setAlign] = useState(true);
  const [showFloor, setShowFloor] = useState(false);
  const [showAll, setShowAll] = useState(true);
  const [downloading, setDownloading] = useState(false);
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
    setAlign(true); setShowFloor(false); setProgress(null); setPollNotice(""); setShowAll(true);
  }, [job?.id]);

  useEffect(() => {
    if (!query || !job?.id || phase !== "queued") return;
    let cancelled = false;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        if (Date.now() > query.deadline) throw new Error("Fusion is taking too long. Check Modal logs and retry.");
        const qs = new URLSearchParams({ concept: query.concept, frame_indices: query.frames.join(",") });
        const payload = await fetchFusionStatus(`/api/jobs/${job.id}/fusion?${qs}`, {
          headers: { "x-worldpilot-key": accessCode }, cache: "no-store", signal: controller.signal,
        }, (attempt, limit) => {
          if (!cancelled) setPollNotice(`Connection interrupted. Reconnecting (${attempt}/${limit})…`);
        });
        if (cancelled) return;
        setPollNotice("");
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
    return () => { cancelled = true; controller.abort(); clearTimeout(timer); };
  }, [query, job?.id, phase, accessCode]);

  async function run() {
    if (!ready || busy || !job?.id) return;
    const label = concept.trim();
    if (!label || label.length > 120) { setError("Enter an object name, such as chair, table, plant, or door."); return; }
    const token = generation.current;
    const n = Math.min(count === "all" ? framesAvailable : count, framesAvailable);
    const frames = Array.from({ length: n }, (_, i) => Math.round(i * (framesAvailable - 1) / (n - 1)));
    setProgress({ completed_frames: 0, total_frames: n, stage: "preparing" });
    setError(""); setPollNotice(""); setShowAll(true); setPhase("starting"); setQuery(null); setResult(null); setSelected(null);
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

  async function downloadGeometry() {
    if (!result?.geometry_file || !job?.id || downloading) return;
    setDownloading(true);
    try {
      const qs = new URLSearchParams({ concept: result.object_concept, frame_indices: result.frame_indices.join(",") });
      const response = await fetch(`/api/jobs/${job.id}/fusion/geometry?${qs}`, {
        headers: { "x-worldpilot-key": accessCode }, cache: "no-store",
      });
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail || `Object geometry download failed (HTTP ${response.status}).`);
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url; link.download = `25worlds-${job.id.slice(0, 8)}-objects.npz`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Could not download object geometry."); }
    finally { setDownloading(false); }
  }

  const tracking = progress?.stage === "tracking_object" || progress?.stage === "tracking_floor";
  const progressValue = tracking ? progress?.tracking_frames ?? 0 : progress?.completed_frames ?? 0;
  const progressMax = tracking ? progress?.tracking_total ?? framesAvailable : progress?.total_frames ?? query?.frames.length ?? 1;
  const stageMessage = progress?.stage === "tracking_object" ? `Tracking object instances across ${progressMax} aligned views…`
    : progress?.stage === "tracking_floor" ? `Tracking floor evidence across ${progressMax} aligned views…`
    : progress?.stage === "associating" ? "Checking visibility and geometry, then reconciling fragmented tracks…"
    : progress?.stage === "exporting" ? "Saving full object geometry and preparing viewer samples…"
    : `Grounded ${progressValue} / ${progressMax} views${progress?.current_frame != null ? ` · Processing view ${progress.current_frame+1}` : ""}…`;

  return <section className={styles.panel} aria-labelledby="fusion-heading">
    <div className="card-top"><h3 className="area-label" id="fusion-heading">Semantic grounding</h3>
      <span className="spatial-kicker">SAM 3.1 VIDEO TRACKING + COLMAP</span></div>
    <p className={styles.description}>Track object instances across the walkthrough and combine their observed 3D surfaces.
      Visibility and geometry checks help keep nearby objects separate; uncertain fragments remain visible.</p>
    {!ready && <p className={styles.message} role="status">
      {job?.status === "completed"
        ? "This scene needs aligned spatial data from at least two frames. Reconstruct a new video if necessary."
        : "Complete a reconstruction to analyze objects across multiple views."}
    </p>}
    <div className={styles.controls}>
      <div className={styles.prompt}>
        <label htmlFor="fusion-concept">Object to find</label>
        <input id="fusion-concept" className="spatial-input" value={concept} maxLength={120}
          placeholder="e.g. sofa or water bottle" disabled={busy} onChange={e => setConcept(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); void run(); } }} />
        <div className="spatial-suggestions">
          {["sofa", "chair", "table", "plant", "door"].map(example => <button type="button" key={example}
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
        </select><small>{count === "all" ? "Every saved COLMAP frame, including views between the previous samples." : "Evenly spaced across the walkthrough."}</small>
        <small>Tracking uses all {framesAvailable || "saved"} aligned views to retain continuity; this selection controls which views contribute 3D points.</small>
      </label>
      <div className={styles.action}>
        <button className="primary-button" disabled={!ready || busy || !concept.trim()} onClick={() => void run()} type="button">
          {busy ? "Fusing observations…" : ready ? `Analyze ${Math.min(count === "all" ? framesAvailable : count, framesAvailable)} frames` : "Analyze multiple frames"}
        </button><small>Saved video masks are reused. A new object prompt can require an object-tracking video request and a floor request. Video billing applies.</small>
      </div>
    </div>
    {error && <p role="alert" className={styles.message}>{error}</p>}
    {busy && <div role="status" className={styles.message}>
      {pollNotice && <p style={{ margin: "0 0 8px" }}>{pollNotice}</p>}
      <p style={{ margin: "0 0 8px" }}>{stageMessage}</p>
      <progress aria-label="Processed reconstruction views" value={progressValue}
        max={Math.max(1, progressMax)} style={{ width: "min(100%, 440px)" }} />
    </div>}
    {result && <>
      <div className={styles.summary}>
        <strong>{result.objects.length} object hypotheses · {result.object_concept} · {result.frame_indices.length} frames</strong>
        <span>Frames {result.frame_indices.map(f => f + 1).join(", ")}</span>
      </div>
      {result.objects.length ? <div className={styles.objects}>
        {result.objects.map(obj => <button key={obj.object_id} type="button"
          aria-pressed={!showAll && String(obj.object_id) === selected}
          className={`spatial-object ${!showAll && String(obj.object_id) === selected ? "is-active" : ""}`}
          onClick={() => { setSelected(String(obj.object_id)); setShowAll(false); }}>
          <span><strong>{obj.label} · {obj.object_id}</strong>
            <small>{obj.observation_count} observations · {obj.unique_sample_points.toLocaleString()} deduplicated 3D points</small>
            <small>{obj.association_status === "geometry_supported" ? "Tracking / geometry agreement"
              : obj.association_status === "tracking_supported" ? "Tracking supported · limited geometric confirmation"
              : obj.association_status === "ambiguous" ? "Uncertain association · kept separate"
              : "Single observation · identity unconfirmed"}</small>
            <small>Seen in frames {obj.frame_indices.map(f => f + 1).join(", ")}</small>
          </span>
        </button>)}
      </div> : <p className={styles.message}>No grounded matches. Check the detected mask and insufficient-depth counts below, or try a simpler visible object name.</p>}
      <div className={styles.stats}>
        <span>Floor estimate: <strong>{displayOverlay?.floor ? `${Math.round(displayOverlay.floor.inlier_ratio * 100)}% inliers · unverified` : "Not detected"}</strong></span>
        <span>Association: <strong>video identity + visibility + shared geometry</strong></span>
        {result.diagnostics && <span>{result.diagnostics.sam_masks} masks · {result.diagnostics.grounded_observations} grounded observations · {result.diagnostics.insufficient_depth_masks} masks without sufficient depth · {result.diagnostics.reconciled_fragments} fragments reconciled</span>}
      </div>
    </>}
    <div className="viewer-head"><h3 className="area-label">Multi-frame object map</h3>
      <span className="viewer-state">{busy ? "ANALYZING" : result ? "FUSED OBSERVATIONS" : ply ? "READY TO ANALYZE" : "AWAITING RECONSTRUCTION"}</span>
    </div>
    <div className={styles.canvas}>
      {ply ? <PointCloudViewer ref={viewer} data={ply} pointSize={.023} overlay={displayOverlay}
        selectedObjectId={showAll ? null : selected} showAllObjects={Boolean(result && showAll)} showFloor={showFloor} alignToFloor={align} />
        : <div className={styles.placeholder}>Your reconstructed world and fused object highlights will appear here.</div>}
    </div>
    <div className={`viewer-toolbar ${styles.toolbar}`}>
      <label className="spatial-floor-toggle"><input type="checkbox" checked={showAll} disabled={!result?.objects.length}
        onChange={e => { setShowAll(e.target.checked); if (!e.target.checked && !selected) setSelected(result?.objects[0]?.object_id.toString() ?? null); }} />Show all matches</label>
      <label className="spatial-floor-toggle"><input type="checkbox" checked={align} disabled={!displayOverlay?.floor}
        onChange={e => setAlign(e.target.checked)} />Align to estimated floor</label>
      <label className="spatial-floor-toggle"><input type="checkbox" checked={showFloor} disabled={!displayOverlay?.floor}
        onChange={e => setShowFloor(e.target.checked)} />Show floor evidence</label>
      <button className="toolbar-button" disabled={!ply || !result?.objects.length || (!showAll && !selected)} onClick={() => viewer.current?.focusSelected()} type="button">{showAll ? "Focus on matches" : "Focus on object"}</button>
      <button className="toolbar-button" disabled={!ply} onClick={() => viewer.current?.reset()} type="button">Reset view</button>
      {result?.geometry_file && <button className="toolbar-button" disabled={downloading} onClick={() => void downloadGeometry()} type="button">{downloading ? "Downloading…" : "Full object geometry"}</button>}
    </div>
    {result?.warnings.map((warning, index) => <p className={styles.message} key={index}>{warning}</p>)}
    <p className={styles.message}>Coral shows {showAll ? "all matched objects" : "the selected object"}. Each object displays up to 5,000 preview points;
      the download preserves all deduplicated grounded points. Object IDs belong to this saved map. Moving objects and hidden surfaces are not reconstructed.</p>
  </section>;
}
