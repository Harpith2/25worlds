import { timingSafeEqual } from "node:crypto";

export const runtime = "nodejs";

function configured() {
  const url = process.env.MODAL_API_URL;
  const code = process.env.WORLD_ACCESS_CODE;
  if (!url || !/^https:\/\//.test(url) || !code || code.length < 10) {
    throw new Error("Backend is not configured. Set MODAL_API_URL and WORLD_ACCESS_CODE on the Next.js server.");
  }
  return { url: url.replace(/\/$/, ""), code };
}

function sameSecret(input: string, expected: string): boolean {
  const a = Buffer.from(input);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

export function getConnection(request: Request) {
  const cfg = configured();
  const input = request.headers.get("x-worldpilot-key") || "";
  if (!sameSecret(input, cfg.code)) {
    return { error: Response.json({ detail: "Incorrect access code." }, { status: 401 }) };
  }
  return { cfg };
}

export function requestError(error: unknown): Response {
  return Response.json({ detail: error instanceof Error ? error.message : "Could not contact reconstruction backend." }, { status: 503 });
}

export async function forwardApi(upstream: string, init: RequestInit): Promise<Response> {
  const response = await fetch(upstream, { ...init, cache: "no-store", signal: AbortSignal.timeout(300000) });
  const contentType = response.headers.get("content-type") || "application/json";
  const headers = new Headers({ "content-type": contentType, "cache-control": "no-store" });
  const disposition = response.headers.get("content-disposition");
  if (disposition) headers.set("content-disposition", disposition);
  if (response.headers.get("content-length")) headers.set("content-length", response.headers.get("content-length")!);
  return new Response(response.body, { status: response.status, headers });
}
