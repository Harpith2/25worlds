import { forwardApi, getConnection, requestError } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: Promise<{ id: string; resolution: string }> }) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;
    const { id, resolution } = await params;
    if (!/^[0-9a-f]{32}$/.test(id) || !["preview", "full"].includes(resolution)) {
      return Response.json({ detail: "PLY not found." }, { status: 404 });
    }
    const { cfg } = auth;
    return forwardApi(`${cfg!.url}/api/jobs/${id}/${resolution}.ply`, {
      headers: { "x-worldpilot-key": cfg!.code },
    });
  } catch (error) {
    return requestError(error);
  }
}
