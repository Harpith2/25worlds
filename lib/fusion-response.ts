import type { SpatialFusionResponse } from "./types";

const temporaryStatuses = new Set([408, 429, 500, 502, 503, 504]);

export class FusionResponseError extends Error {
  constructor(message: string, public readonly status: number, public readonly retryable: boolean) {
    super(message);
    this.name = "FusionResponseError";
  }
}

function responseHint(status: number): string {
  if (status === 404 || status === 405) {
    return " Check that app/api/jobs/[id]/fusion/route.ts exists, restart Next.js, and redeploy cloud/modal_app.py with the fusion backend.";
  }
  if ([502, 503, 504].includes(status)) {
    return " Check the Next.js terminal and Modal web/worker logs, and confirm the fusion backend is deployed.";
  }
  return " Check the Next.js terminal and Modal logs for this request.";
}

function detailMessage(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map(item => typeof item?.msg === "string" ? item.msg : "Invalid request field").join("; ");
  }
  return "";
}

export async function parseFusionResponse(response: Response): Promise<SpatialFusionResponse> {
  const status = response.status;
  let text: string;
  try { text = await response.text(); }
  catch {
    throw new FusionResponseError(`Fusion response was interrupted (HTTP ${status}).${responseHint(status)}`, status, true);
  }
  if (!text.trim()) {
    throw new FusionResponseError(`Fusion returned an empty response (HTTP ${status}).${responseHint(status)}`, status, response.ok || temporaryStatuses.has(status));
  }
  let payload;
  try { payload = JSON.parse(text); }
  catch {
    throw new FusionResponseError(`Fusion returned a non-JSON response (HTTP ${status}).${responseHint(status)}`, status, temporaryStatuses.has(status));
  }
  if (!response.ok) {
    const detail = detailMessage(payload?.detail) || detailMessage(payload?.error);
    throw new FusionResponseError(detail || `Fusion request failed (HTTP ${status}).${responseHint(status)}`,
      status, payload?.retryable !== false && temporaryStatuses.has(status));
  }
  if (!payload || typeof payload !== "object" ||
      !["queued", "failed", "completed"].includes(payload.status)) {
    throw new FusionResponseError(`Fusion returned an unexpected response (HTTP ${status}). Confirm that both frontend and Modal use the fusion update.`, status, false);
  }
  if (payload.status === "completed") {
    const result = payload.result;
    if (!result || result.version !== "video-world-v1" || typeof result.object_concept !== "string" ||
        !Array.isArray(result.objects) || !Array.isArray(result.frame_indices) ||
        !result.frame_indices.every(Number.isInteger) || !Array.isArray(result.warnings) ||
        !result.warnings.every((v: unknown) => typeof v === "string") ||
        !result.objects.every((obj: Record<string, unknown>) => obj &&
          ["string", "number"].includes(typeof obj.object_id) && typeof obj.label === "string" &&
          Array.isArray(obj.frame_indices) && obj.frame_indices.every(Number.isInteger) &&
          Array.isArray(obj.sample_xyz) && Array.isArray(obj.center_xyz) &&
          typeof obj.observation_count === "number" && typeof obj.unique_sample_points === "number")) {
      throw new FusionResponseError("Fusion completed without a valid object-map result. Confirm that the latest fusion backend is deployed.", status, false);
    }
  }
  return payload as SpatialFusionResponse;
}

/** Retry only read-only polling; never resubmit a paid SAM analysis automatically. */
export async function fetchFusionStatus(
  url: string, init: RequestInit,
  onRetry?: (attempt: number, limit: number) => void,
): Promise<SpatialFusionResponse> {
  const limit = 5;
  for (let attempt = 0; ; attempt++) {
    try {
      const timeout = AbortSignal.timeout(45000);
      const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
      return await parseFusionResponse(await fetch(url, { ...init, method: "GET", signal }));
    } catch (cause) {
      if (init.signal?.aborted) throw cause;
      const retryable = cause instanceof FusionResponseError ? cause.retryable
        : cause instanceof TypeError || (cause instanceof Error && ["AbortError", "TimeoutError"].includes(cause.name));
      if (!retryable) throw cause;
      if (attempt >= limit) {
        throw new FusionResponseError(
          "Could not reconnect to fusion status. The analysis may still be running. " +
          "Click Analyze again with the same object and frames to reconnect. " +
          (cause instanceof Error ? cause.message : "Status request failed."),
          cause instanceof FusionResponseError ? cause.status : 0, true,
        );
      }
      onRetry?.(attempt + 1, limit);
      await new Promise<void>((resolve, reject) => {
        const signal = init.signal;
        const done = () => { signal?.removeEventListener("abort", aborted); resolve(); };
        const timer = setTimeout(done, Math.min(3000 * 2 ** attempt, 15000));
        const aborted = () => { clearTimeout(timer); signal?.removeEventListener("abort", aborted); reject(signal?.reason); };
        signal?.addEventListener("abort", aborted, { once: true });
        if (signal?.aborted) aborted();
      });
    }
  }
}
