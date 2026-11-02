import { forwardApi, getConnection, requestError } from "@/lib/modal";
import { parseFusionResponse } from "@/lib/fusion-response";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;
type Context = { params: Promise<{ id: string }> };

async function forwardFusion(upstream: string, init: RequestInit): Promise<Response> {
  // Consume the upstream body inside the handler's try/catch. Re-encode JSON
  // instead of forwarding stale compressed content-length headers or HTML.
  const response = await forwardApi(upstream, init);
  try {
    const payload = await parseFusionResponse(response);
    return Response.json(payload, { status: response.status, headers: { "cache-control": "no-store" } });
  } catch (cause) {
    return Response.json({ detail: cause instanceof Error ? cause.message : "Invalid fusion response." }, {
      status: response.status >= 400 && response.status <= 599 ? response.status : 502,
      headers: { "cache-control": "no-store" },
    });
  }
}

function validFrames(value: unknown): value is number[] {
  return Array.isArray(value) && value.length >= 2 &&
    value.every((v) => Number.isInteger(v) && v >= 0) && new Set(value).size === value.length;
}

export async function POST(request: Request, { params }: Context) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;
    const { id } = await params;
    if (!/^[a-f0-9]{32}$/.test(id)) return Response.json({ detail: "Invalid job ID." }, { status: 404 });
    const body = await request.json();
    if (!body || typeof body.concept !== "string" || !body.concept.trim() ||
      body.concept.length > 120 || !validFrames(body.frame_indices)) {
      return Response.json({ detail: "Enter an object prompt and at least 2 distinct reconstructed frames." }, { status: 400 });
    }
    return await forwardFusion(`${auth.cfg!.url}/api/jobs/${id}/fusion`, {
      method: "POST", headers: { "x-worldpilot-key": auth.cfg!.code, "content-type": "application/json" },
      body: JSON.stringify({ concept: body.concept.trim(), frame_indices: [...body.frame_indices].sort((a,b) => a-b) }),
    });
  } catch (error) { return requestError(error); }
}

export async function GET(request: Request, { params }: Context) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;
    const { id } = await params;
    if (!/^[a-f0-9]{32}$/.test(id)) return Response.json({ detail: "Invalid job ID." }, { status: 404 });
    const query = new URL(request.url).searchParams;
    const concept = query.get("concept") || "";
    const raw = query.get("frame_indices") || "";
    const frames = /^\d+(,\d+)+$/.test(raw) ? raw.split(",").map(Number) : [];
    if (!concept.trim() || concept.length > 120 || !validFrames(frames)) {
      return Response.json({ detail: "Invalid fusion query." }, { status: 400 });
    }
    const qs = new URLSearchParams({ concept: concept.trim(), frame_indices: frames.sort((a,b)=>a-b).join(",") });
    return await forwardFusion(`${auth.cfg!.url}/api/jobs/${id}/fusion?${qs}`, {
      headers: { "x-worldpilot-key": auth.cfg!.code },
    });
  } catch (error) { return requestError(error); }
}
