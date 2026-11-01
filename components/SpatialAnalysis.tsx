"use client";

import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  ArrowUpRight,
  CheckCircle2,
  CircleHelp,
  Layers3,
  LoaderCircle,
  ScanSearch,
  X,
} from "lucide-react";

import {
  PointCloudViewer,
  type PointCloudControls,
} from "@/components/PointCloudViewer";

import styles from "./SpatialAnalysis.module.css";

import type {
  ReconstructionJob,
  SpatialObject,
  SpatialResponse,
  SpatialResult,
} from "@/lib/types";

type Props = {
  job: ReconstructionJob | null;
  ply: ArrayBuffer | null;
  accessCode: string;
};

type GroundedWithMask = SpatialObject & {
  mask_png?: string;
};

type Query = {
  concept: string;
  frameIndex: number;
};

type Phase =
  | "idle"
  | "submitting"
  | "queued"
  | "completed"
  | "failed";

const examples = [
  "chair",
  "table",
  "plant",
  "door",
];

async function errorMessage(
  response: Response
): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?: string;
    };

    return (
      body.detail ||
      `Request failed (${response.status}).`
    );
  } catch {
    return `Request failed (${response.status}).`;
  }
}

function coordsText(value: number[]) {
  return value
    .map((v) => v.toFixed(2))
    .join(", ");
}

export function SpatialAnalysis({
  job,
  ply,
  accessCode,
}: Props) {
  const [concept, setConcept] = useState("chair");
  const [frameIndex, setFrameIndex] = useState(0);

  const [query, setQuery] =
    useState<Query | null>(null);

  const [phase, setPhase] =
    useState<Phase>("idle");

  const [result, setResult] =
    useState<SpatialResult | null>(null);

  const [frameImage, setFrameImage] =
    useState<string | null>(null);

  const [selectedId, setSelectedId] =
    useState<string | null>(null);

  const [showFloor, setShowFloor] =
    useState(false);

  const [alignToFloor, setAlignToFloor] = useState(false);

  const [error, setError] = useState("");

  const viewer = useRef<PointCloudControls>(null);

  const ready = Boolean(
    job?.status === "completed" &&
    job.spatial_ready &&
    ply &&
    accessCode
  );

  const busy =
    phase === "submitting" ||
    phase === "queued";

  const objects = (
    result?.objects ?? []
  ) as GroundedWithMask[];

  const selected =
    objects.find(
      (object) =>
        String(object.object_id) === selectedId
    ) ?? null;

  // --------------------------------------------------
  // RESET WHEN THE SCENE CHANGES
  // --------------------------------------------------

  useEffect(() => {
    setPhase("idle");
    setQuery(null);
    setResult(null);
    setFrameImage(null);
    setSelectedId(null);
    setFrameIndex(0);
    setShowFloor(false);
    setAlignToFloor(false);
    setError("");
  }, [job?.id]);

  function acceptResult(payload: SpatialResponse) {
    if (!payload.result) {
      throw new Error(
        "SAM returned no analysis result."
      );
    }

    setResult(payload.result);

    setFrameImage(
      payload.frame_image_url ?? null
    );

    setSelectedId(
      payload.result.objects.length
        ? String(
            payload.result.objects[0].object_id
          )
        : null
    );

    setPhase("completed");
    setQuery(null);
  }

  // --------------------------------------------------
  // POLL THE MODAL ANALYSIS WORKER
  // --------------------------------------------------

  useEffect(() => {
    if (
      !query ||
      !job?.id ||
      !accessCode ||
      phase !== "queued"
    ) {
      return;
    }

    let cancelled = false;

    let timer:
      ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const params = new URLSearchParams({
          concept: query.concept,
          frame_index: String(
            query.frameIndex
          ),
        });

        const response = await fetch(
          `/api/jobs/${job.id}/analysis?${params}`,
          {
            headers: {
              "x-worldpilot-key": accessCode,
            },
            cache: "no-store",
          }
        );

        if (!response.ok) {
          throw new Error(
            await errorMessage(response)
          );
        }

        const payload =
          (await response.json()) as SpatialResponse;

        if (cancelled) return;

        if (payload.status === "completed") {
          acceptResult(payload);

        } else if (payload.status === "failed") {
          throw new Error(
            payload.error ||
            "SAM worker failed. You can retry."
          );

        } else {
          timer = setTimeout(
            poll,
            2500
          );
        }

      } catch (cause) {
        if (cancelled) return;

        setError(
          cause instanceof Error
            ? cause.message
            : "Could not retrieve spatial analysis."
        );

        setPhase("failed");
        setQuery(null);
      }
    };

    timer = setTimeout(poll, 1000);

    return () => {
      cancelled = true;

      if (timer) {
        clearTimeout(timer);
      }
    };
  }, [
    query,
    job?.id,
    accessCode,
    phase,
  ]);

  // --------------------------------------------------
  // START SPATIAL ANALYSIS
  // --------------------------------------------------

  async function analyze() {
    if (!ready || busy || !job?.id) {
      return;
    }

    const label = concept.trim();

    if (!label || label.length > 120) {
      setError(
        "Enter a short object name, like chair or plant."
      );
      return;
    }

    setError("");
    setPhase("submitting");
    setQuery(null);
    setResult(null);
    setFrameImage(null);
    setSelectedId(null);

    try {
      const response = await fetch(
        `/api/jobs/${job.id}/analysis`,
        {
          method: "POST",
          headers: {
            "x-worldpilot-key": accessCode,
            "content-type": "application/json",
          },
          body: JSON.stringify({
            concept: label,
            frame_index: frameIndex,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          await errorMessage(response)
        );
      }

      const payload =
        (await response.json()) as SpatialResponse;

      if (payload.status === "completed") {
        acceptResult(payload);

      } else if (payload.status === "queued") {
        setQuery({
          concept: label,
          frameIndex,
        });

        setPhase("queued");

      } else {
        throw new Error(
          payload.error ||
          "Could not start SAM analysis."
        );
      }

    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not start analysis."
      );

      setPhase("failed");
    }
  }

  // --------------------------------------------------
  // FRONTEND
  // --------------------------------------------------

  return (
    <section
      className={`${styles.workspace} spatial-section`}
      id="spatial"
      aria-labelledby="spatial-heading"
    >
      <div className="section-title-row spatial-title-row">
        <h2
          className="section-label"
          id="spatial-heading"
        >
          02 / Spatial intelligence
        </h2>

        <p>
          From segmented video pixels to
          grounded 3D objects.
        </p>
      </div>

      <div className="workspace-grid">
        <div className="upload-card spatial-control-card">
          <div className="card-top">
            <h3 className="area-label">
              Semantic grounding
            </h3>

            <span className="spatial-kicker">
              SAM 3.1 + RANSAC
            </span>
          </div>

          <div className="upload-content spatial-content">
            <p className="spatial-description">
              SAM 3.1 segments objects and floor
              pixels in a camera frame. Pi3X maps
              each pixel into 3D; RANSAC estimates
              a candidate floor plane.
            </p>

            {!ready && (
              <div className="spatial-not-ready">
                <Layers3 size={24} />

                <strong>
                  {job?.status === "completed" &&
                  !job.spatial_ready
                    ? "This scene needs aligned spatial data"
                    : "Reconstruct a world first"}
                </strong>

                <span>
                  {job?.status === "completed" &&
                  !job.spatial_ready
                    ? "Reconstruct a new video with the updated Modal worker. Older PLY-only jobs cannot be grounded."
                    : "Finish the video reconstruction above to unlock SAM 3.1 analysis."}
                </span>
              </div>
            )}

            <fieldset
              disabled={!ready || busy}
              className="spatial-fields"
            >
              <div className="spatial-query-field">
              <label htmlFor="sam-concept">
                Object to find
              </label>

              <input
                id="sam-concept"
                className="spatial-input"
                value={concept}
                maxLength={120}
                placeholder="e.g. chair"
                onChange={(event) =>
                  setConcept(
                    event.target.value
                  )
                }
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void analyze();
                  }
                }}
              />

              <div className="spatial-suggestions">
                {examples.map((example) => (
                  <button
                    key={example}
                    type="button"
                    className={
                      concept === example
                        ? "is-selected"
                        : ""
                    }
                    onClick={() =>
                      setConcept(example)
                    }
                  >
                    {example}
                  </button>
                ))}
              </div>

              </div>

              <div className="spatial-frame-field">
              <label htmlFor="sam-frame">
                Video frame
              </label>

              <select
                id="sam-frame"
                className="spatial-input"
                value={frameIndex}
                onChange={(event) =>
                  setFrameIndex(
                    Number(event.target.value)
                  )
                }
              >
                {Array.from(
                  {
                    length: Math.max(
                      job?.spatial_frames || 0,
                      1
                    ),
                  },
                  (_, index) => (
                    <option
                      value={index}
                      key={index}
                    >
                      Frame{" "}
                      {String(
                        index + 1
                      ).padStart(2, "0")}
                    </option>
                  )
                )}
              </select>
              </div>

              <button
                className="primary-button spatial-action"
                type="button"
                onClick={() => void analyze()}
                disabled={!ready || busy}
              >
                {busy ? (
                  <>
                    <LoaderCircle
                      className="spin"
                      size={18}
                    />

                    {phase === "queued"
                      ? "Analyzing with SAM…"
                      : "Starting analysis…"}
                  </>
                ) : (
                  <>
                    <ScanSearch size={18} />
                    Ground object in 3D
                    <ArrowUpRight size={17} />
                  </>
                )}
              </button>
            </fieldset>

            {error && (
              <div
                role="alert"
                className="error-alert"
              >
                <CircleHelp size={17} />
                <span>{error}</span>

                <button
                  type="button"
                  aria-label="Dismiss error"
                  onClick={() =>
                    setError("")
                  }
                >
                  <X size={16} />
                </button>
              </div>
            )}

            {phase === "queued" && (
              <p
                className="spatial-caption"
                role="status"
              >
                Meta is segmenting the selected
                frame; the Modal worker is grounding
                its masks to Pi3X geometry.
                Failed runs can be retried.
              </p>
            )}

            {result && (
              <div className="spatial-results">
                <div className="spatial-result-heading">
                  <CheckCircle2 size={16} />

                  <strong>
                    {objects.length} grounded{" "}
                    {objects.length === 1
                      ? "instance"
                      : "instances"}
                  </strong>
                </div>

                <p className="spatial-caption">
                  Select an instance to highlight
                  its exact SAM mask and
                  corresponding 3D points.
                </p>

                {objects.length ? (
                  <div className="spatial-object-list">
                    {objects.map((object) => (
                      <button
                        key={String(
                          object.object_id
                        )}
                        type="button"
                        className={`spatial-object ${
                          String(object.object_id) ===
                          selectedId
                            ? "is-active"
                            : ""
                        }`}
                        onClick={() =>
                          setSelectedId(
                            String(
                              object.object_id
                            )
                          )
                        }
                      >
                        <span>
                          <strong>
                            {object.label}
                          </strong>

                          <small>
                            Instance{" "}
                            {String(
                              object.object_id
                            )}{" "}
                            ·{" "}
                            {object.grounded_points.toLocaleString()}{" "}
                            3D points
                          </small>
                        </span>

                        <span className="spatial-object-coords">
                          {coordsText(
                            object.center_xyz
                          )}
                        </span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="spatial-caption">
                    No reliable 3D instances found.
                    Try another frame or prompt.
                  </p>
                )}

                <div className="spatial-stat-row">
                  <span>
                    SAM-guided floor candidate
                  </span>

                  <strong>
                    {result.floor
                      ? `${Math.round(
                          result.floor.inlier_ratio *
                            100
                        )}% inliers · unverified`
                      : "Not detected"}
                  </strong>
                </div>

                {selected && (
                  <div className="spatial-stat-row">
                    <span>
                      Selected center (Pi3X units)
                    </span>

                    <strong>
                      {coordsText(
                        selected.center_xyz
                      )}
                    </strong>
                  </div>
                )}

                {selected && (
                  <div className="spatial-stat-row">
                    <span>Displayed object samples</span>
                    <strong>{selected.sample_xyz.filter((xyz) =>
                      xyz.length === 3 && xyz.every(Number.isFinite)).length.toLocaleString()}</strong>
                  </div>
                )}

                {result.warnings.map(
                  (warning, index) => (
                    <p
                      className="spatial-caption"
                      key={index}
                    >
                      {warning}
                    </p>
                  )
                )}
              </div>
            )}
          </div>
        </div>

        <div className="viewer-card spatial-viewer-card">
          <div className="viewer-head">
            <h3 className="area-label">
              2D ↔ 3D grounding
            </h3>

            <span className="viewer-state">
              {result
                ? "SPATIAL DATA"
                : "AWAITING SAM"}
            </span>
          </div>

          <div className="spatial-visual-grid">
            {/* Original Pi3X-aligned frame */}
            <div className="spatial-frame-panel">
              <div className="spatial-visual-label">
                01 / SAM source frame
              </div>

              <div className="spatial-frame-stage">
              {frameImage && result ? (
                <div className="spatial-image-wrap" style={{
                  aspectRatio: `${result.image_width} / ${result.image_height}`,
                  maxWidth: `calc(var(--spatial-stage-height) * ${result.image_width / result.image_height})`,
                }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={frameImage}
                    alt={`Frame ${
                      result.frame_index + 1
                    } analyzed by SAM`}
                  />

                  {/* Exact SAM mask, not just a box. */}
                  {selected?.mask_png ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={selected.mask_png}
                      alt=""
                      aria-hidden="true"
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "fill",
                        pointerEvents: "none",
                        zIndex: 1,
                      }}
                    />
                  ) : selected ? (
                    <span
                      className="spatial-caption"
                      style={{
                        position: "absolute",
                        bottom: 8,
                        left: 8,
                        background: "#171717",
                        color: "white",
                        padding: 6,
                        zIndex: 2,
                      }}
                    >
                      Legacy result: mask overlay
                      unavailable. Run a fresh
                      analysis after redeployment.
                    </span>
                  ) : null}

                  {/* Clickable object instance boxes */}
                  {objects.map((object) => {
                    const [
                      left,
                      top,
                      right,
                      bottom,
                    ] = object.bounds_xyxy;

                    const active =
                      String(object.object_id) ===
                      selectedId;

                    return (
                      <button
                        key={String(
                          object.object_id
                        )}
                        type="button"
                        className={`spatial-mask-box ${
                          active
                            ? "is-active"
                            : ""
                        }`}
                        aria-label={`Select ${
                          object.label
                        } instance ${String(
                          object.object_id
                        )}`}
                        onClick={() =>
                          setSelectedId(
                            String(
                              object.object_id
                            )
                          )
                        }
                        style={{
                          left: `${
                            (100 * left) /
                            result.image_width
                          }%`,
                          top: `${
                            (100 * top) /
                            result.image_height
                          }%`,
                          width: `${
                            (100 *
                              (right - left)) /
                            result.image_width
                          }%`,
                          height: `${
                            (100 *
                              (bottom - top)) /
                            result.image_height
                          }%`,
                          background: "transparent",
                          zIndex: 2,
                        }}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="spatial-placeholder">
                  <ScanSearch size={28} />

                  <span>
                    {busy
                      ? "SAM is processing your frame…"
                      : "The source frame will appear after analysis"}
                  </span>
                </div>
              )}

              </div>

              <p className="spatial-panel-foot">
                Coral = exact SAM mask of selected
                object; outlines = instance selection
              </p>
            </div>

            {/* Full point cloud with selected object */}
            <div className="spatial-cloud-panel">
              <div className="spatial-visual-label">
                02 / Pi3X spatial world
              </div>

              <div className="spatial-cloud-canvas">
                {ply ? (
                  <PointCloudViewer
                    ref={viewer}
                    data={ply}
                    pointSize={0.023}
                    overlay={result}
                    selectedObjectId={selectedId}
                    showFloor={showFloor}
                    alignToFloor={alignToFloor}
                  />
                ) : (
                  <div className="spatial-placeholder">
                    <Layers3 size={28} />

                    <span>
                      Point cloud appears after
                      reconstruction
                    </span>
                  </div>
                )}
              </div>

              <p className="spatial-panel-foot">
                Full background cloud remains
                visible · Coral = selected object ·
                Mint = candidate floor
              </p>
            </div>
          </div>

          <div className="viewer-toolbar spatial-toolbar">
            <label className="spatial-floor-toggle">
              <input
                type="checkbox"
                checked={showFloor}
                onChange={(event) =>
                  setShowFloor(
                    event.target.checked
                  )
                }
              />

              Show candidate floor plane
            </label>

            <label className="spatial-floor-toggle">
              <input
                type="checkbox"
                checked={alignToFloor}
                disabled={!ply || !result?.floor ||
                  !result.floor.normal.every(Number.isFinite) ||
                  Math.hypot(...result.floor.normal) <= 1e-8 ||
                  !Number.isFinite(result.floor.offset)}
                onChange={(event) => setAlignToFloor(event.target.checked)}
              />
              Align to estimated floor
            </label>

            <button
              className="toolbar-button"
              type="button"
              disabled={!ply || !selected || !(selected.grounded_points > 0)}
              onClick={() => viewer.current?.focusSelected()}
            >
              Focus on object
            </button>

            <button
              className="toolbar-button"
              type="button"
              disabled={!ply}
              onClick={() => viewer.current?.reset()}
            >
              Reset view
            </button>
          </div>
        </div>
      </div>

      <p className="spatial-disclaimer">
        Alignment changes only the display; the upward direction is heuristic.
        The floor is a SAM-guided RANSAC plane
        estimate, not a verified walkable surface.
        Pi3X coordinates are approximate and not a
        collision-safe navigation map.
      </p>
    </section>
  );
}
