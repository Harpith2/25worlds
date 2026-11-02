import { getConnection, requestError } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

export async function GET(request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;
    const { id } = await params;
    const query = new URL(request.url).searchParams;
    const concept = query.get("concept")?.trim() ?? "";
    const indices = query.get("frame_indices") ?? "";
    const frames = indices.split(",").map(Number);
    if (!/^[a-f0-9]{32}$/.test(id) || !concept || concept.length > 120 ||
        !/^\d+(,\d+)+$/.test(indices) || !frames.every(Number.isSafeInteger) || new Set(frames).size !== frames.length) {
      return Response.json({ detail: "Invalid geometry download request." }, { status: 400 });
    }
    const qs = new URLSearchParams({ concept, frame_indices: frames.sort((a,b)=>a-b).join(",") });
    const response = await fetch(`${auth.cfg!.url}/api/jobs/${id}/fusion/geometry?${qs}`, {
      headers: { "x-worldpilot-key": auth.cfg!.code }, cache: "no-store", signal: AbortSignal.timeout(300000),
    });
    const headers = new Headers({ "cache-control": "no-store",
      "content-type": response.headers.get("content-type") || "application/octet-stream" });
    const disposition = response.headers.get("content-disposition");
    if (disposition) headers.set("content-disposition", disposition);
    // Fetch can decompress upstream data. Do not forward stale encoded lengths.
    return new Response(response.body, { status: response.status, headers });
  } catch (cause) { return requestError(cause); }
}
