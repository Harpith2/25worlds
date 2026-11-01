import { forwardApi, getConnection, requestError } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;
    const size = Number(request.headers.get("content-length") || 0);
    if (size > 62 * 1024 * 1024) return Response.json({ detail: "Video is too large (60 MB maximum)." }, { status: 413 });
    const contentType = request.headers.get("content-type") || "";
    if (!contentType.startsWith("multipart/form-data;")) {
      return Response.json({ detail: "Expected a video upload." }, { status: 415 });
    }
    if (!request.body) return Response.json({ detail: "Empty upload." }, { status: 400 });
    const { cfg } = auth;
    // Stream the multipart body to Modal; no video is stored in the Next.js project.
    return await forwardApi(`${cfg!.url}/api/jobs`, {
      method: "POST",
      headers: { "content-type": contentType, "x-worldpilot-key": cfg!.code },
      body: request.body,
      duplex: "half",
    } as RequestInit & { duplex: "half" });
  } catch (error) {
    return requestError(error);
  }
}
