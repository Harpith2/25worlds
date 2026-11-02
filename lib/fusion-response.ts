import type { SpatialFusionResponse } from "./types";

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
    throw new Error(`Fusion response was interrupted (HTTP ${status}).${responseHint(status)}`);
  }
  if (!text.trim()) {
    throw new Error(`Fusion returned an empty response (HTTP ${status}).${responseHint(status)}`);
  }
  let payload;
  try { payload = JSON.parse(text); }
  catch {
    throw new Error(`Fusion returned a non-JSON response (HTTP ${status}).${responseHint(status)}`);
  }
  if (!response.ok) {
    const detail = detailMessage(payload?.detail) || detailMessage(payload?.error) || "Fusion request failed.";
    throw new Error(`${detail} (HTTP ${status}).${responseHint(status)}`);
  }
  if (!payload || typeof payload !== "object" ||
      !["queued", "failed", "completed"].includes(payload.status)) {
    throw new Error(`Fusion returned an unexpected response (HTTP ${status}). Confirm that both frontend and Modal use the fusion update.`);
  }
  if (payload.status === "completed") {
    const result = payload.result;
    if (!result || typeof result.object_concept !== "string" ||
        !Array.isArray(result.objects) || !Array.isArray(result.frame_indices) ||
        !result.frame_indices.every(Number.isInteger) || !Array.isArray(result.warnings) ||
        !result.warnings.every((v: unknown) => typeof v === "string") ||
        !result.objects.every((obj: Record<string, unknown>) => obj &&
          ["string", "number"].includes(typeof obj.object_id) && typeof obj.label === "string" &&
          Array.isArray(obj.frame_indices) && obj.frame_indices.every(Number.isInteger) &&
          Array.isArray(obj.sample_xyz) && Array.isArray(obj.center_xyz) &&
          typeof obj.observation_count === "number" && typeof obj.unique_sample_points === "number")) {
      throw new Error("Fusion completed without a valid object-map result. Confirm that the latest fusion backend is deployed.");
    }
  }
  return payload as SpatialFusionResponse;
}
