
import {
  forwardApi,
  getConnection,
  requestError,
} from "@/lib/modal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Context = {
  params: Promise<{ id: string }>;
};

export async function POST(
  request: Request,
  { params }: Context
) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;

    const { id } = await params;

    if (!/^[a-f0-9]{32}$/.test(id)) {
      return Response.json(
        { detail: "Invalid job ID." },
        { status: 404 }
      );
    }

    const body = await request.json();

    if (
      typeof body.concept !== "string" ||
      body.concept.trim().length > 120 ||
      body.concept.trim().length === 0 ||
      !Number.isInteger(body.frame_index)
    ) {
      return Response.json(
        {
          detail: "Enter a valid object concept and frame number.",
        },
        { status: 400 }
      );
    }

    const { cfg } = auth;

    return forwardApi(
      `${cfg!.url}/api/jobs/${id}/analysis`,
      {
        method: "POST",
        headers: {
          "x-worldpilot-key": cfg!.code,
          "content-type": "application/json",
        },
        body: JSON.stringify({
          concept: body.concept.trim(),
          frame_index: body.frame_index,
        }),
      }
    );
  } catch (error) {
    return requestError(error);
  }
}

export async function GET(
  request: Request,
  { params }: Context
) {
  try {
    const auth = getConnection(request);
    if ("error" in auth) return auth.error;

    const { id } = await params;

    if (!/^[a-f0-9]{32}$/.test(id)) {
      return Response.json(
        { detail: "Invalid job ID." },
        { status: 404 }
      );
    }

    const query = new URL(request.url).searchParams;
    const concept = query.get("concept") || "";
    const frameIndex = Number(
      query.get("frame_index")
    );

    if (
      !concept.trim() ||
      concept.length > 120 ||
      !Number.isInteger(frameIndex) ||
      frameIndex < 0
    ) {
      return Response.json(
        { detail: "Invalid analysis query." },
        { status: 400 }
      );
    }

    const { cfg } = auth;

    const qs = new URLSearchParams({
      concept: concept.trim(),
      frame_index: String(frameIndex),
    });

    return forwardApi(
      `${cfg!.url}/api/jobs/${id}/analysis?${qs}`,
      {
        headers: {
          "x-worldpilot-key": cfg!.code,
        },
      }
    );
  } catch (error) {
    return requestError(error);
  }
}
