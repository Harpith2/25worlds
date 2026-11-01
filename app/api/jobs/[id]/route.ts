import { forwardApi, getConnection, requestError } from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Params = { params: Promise<{ id: string }> };

async function dispatch(request: Request, context: Params, method: "GET" | "DELETE") {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;
    const { id } = await context.params;
    if (!/^[0-9a-f]{32}$/.test(id)) return Response.json({ detail: "Invalid job ID." }, { status: 404 });
    const { cfg } = auth;
    return forwardApi(`${cfg!.url}/api/jobs/${id}`, {
      method,
      headers: { "x-worldpilot-key": cfg!.code },
    });
  } catch (error) {
    return requestError(error);
  }
}

export async function GET(request: Request, context: Params) { return dispatch(request, context, "GET"); }
export async function DELETE(request: Request, context: Params) { return dispatch(request, context, "DELETE"); }
