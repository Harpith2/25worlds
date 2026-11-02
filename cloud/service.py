
"""HTTP API for 25worlds: reconstruction and spatial analysis."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from fusion_cache import FUSION_VERSION, fusion_paths as tracked_fusion_paths

SERVICE_REVISION = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()[:12]

logger = logging.getLogger("uvicorn.error")

MAX_BYTES = 60 * 1024 * 1024
MAX_SECONDS = 60.0
MIN_SECONDS = 4.0

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def save_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def get_job_dir(root: Path, job_id: str) -> Path:
    if not ID_PATTERN.fullmatch(job_id):
        raise HTTPException(404, "Reconstruction not found")

    path = root / job_id
    if not path.is_dir():
        raise HTTPException(404, "Reconstruction not found")

    return path


def inspect_video(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "format=duration:stream=width,height,codec_name",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=25,
            check=True,
        )

        data = json.loads(proc.stdout)
        duration = float(
            data.get("format", {}).get("duration", 0)
        )
        stream = data.get("streams", [{}])[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

    except (
        ValueError,
        IndexError,
        KeyError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ):
        raise HTTPException(
            400,
            "Could not decode this video. Export as H.264 MP4 or MOV.",
        )

    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise HTTPException(
            400,
            f"Walkthrough must be between "
            f"{MIN_SECONDS:g} and {MAX_SECONDS:g} seconds.",
        )

    if (
        width < 160
        or height < 160
        or width > 7680
        or height > 7680
    ):
        raise HTTPException(
            400,
            "Unsupported video resolution.",
        )

    return {
        "duration_seconds": round(duration, 1),
        "width": width,
        "height": height,
        "codec": stream.get("codec_name", "unknown"),
    }


def extract_frames(video: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", str(video),
        "-vf", "fps=1,scale=960:-2",
        "-frames:v", "60",
        "-q:v", "3",
        str(output / "%04d.jpg"),
    ]

    try:
        subprocess.run(
            args,
            timeout=60,
            capture_output=True,
            check=True,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        raise HTTPException(
            400,
            "Frame extraction failed. Try a standard H.264 MP4.",
        )

    count = len(list(output.glob("*.jpg")))

    if count < 2:
        raise HTTPException(
            400,
            "At least two usable video frames are required.",
        )

    return count


def create_api(
    *,
    volume,
    enqueue: Callable[[str], Awaitable[object]],
    root: Path = Path("/data"),
    enqueue_analysis: Callable[
        [str, str, int], Awaitable[object]
    ] | None = None,
    enqueue_fusion: Callable[[str, str, list[int]], Awaitable[object]] | None = None,
) -> FastAPI:

    api = FastAPI(
        title="25worlds · Geometry Studio",
        version="0.2.0",
    )

    # Prevent duplicate submissions within one API process.
    analysis_locks: dict[str, asyncio.Lock] = {}
    refresh_lock = asyncio.Lock()

    @api.exception_handler(Exception)
    async def explain_unexpected_error(request: Request, exc: Exception):
        logger.exception("API revision %s: %s %s failed",
                         SERVICE_REVISION, request.method, request.url.path)
        return JSONResponse(status_code=500, content={
            "detail": "Spatial API request failed. Check the Modal web logs for the traceback.",
            "retryable": request.method == "GET",
            "revision": SERVICE_REVISION,
            "fusion_version": FUSION_VERSION,
        }, headers={"Cache-Control": "no-store"})

    @api.exception_handler(HTTPException)
    async def explain_http_error(
        request: Request,
        exc: HTTPException,
    ):
        if exc.status_code == 400:
            logger.warning(
                "API revision %s: %s %s returned 400: %s",
                SERVICE_REVISION,
                request.method,
                request.url.path,
                exc.detail,
            )

        return await http_exception_handler(
            request,
            exc,
        )

    def require_access(
        x_worldpilot_key: str | None = Header(default=None),
    ) -> None:
        import hmac

        expected = os.getenv("WORLD_ACCESS_CODE", "")

        if len(expected) < 10:
            raise HTTPException(
                503,
                "Demo access is not configured. "
                "Set WORLD_ACCESS_CODE in your Modal secret.",
            )

        if (
            not x_worldpilot_key
            or not hmac.compare_digest(
                x_worldpilot_key,
                expected,
            )
        ):
            raise HTTPException(
                401,
                "Invalid access code.",
            )

    async def refresh() -> None:
        try:
            # Bound status requests and avoid overlapping reloads in this process.
            async with asyncio.timeout(20):
                async with refresh_lock:
                    await volume.reload.aio()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.exception("Could not refresh job storage (API revision %s)", SERVICE_REVISION)
            raise HTTPException(
                503, "Job storage is temporarily unavailable. Retry this status request.",
                headers={"Retry-After": "3", "Cache-Control": "no-store"},
            ) from exc

    @api.get("/api/health")
    def health():
        return {
            "ok": True,
            "service": "25worlds",
            "phase": "spatial-grounding",
            "revision": SERVICE_REVISION,
            "fusion_version": FUSION_VERSION,
        }

    @api.post(
        "/api/jobs",
        dependencies=[Depends(require_access)],
        status_code=202,
    )
    async def submit(video: UploadFile = File(...)):
        filename = Path(video.filename or "video.mp4").name
        ext = Path(filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                "Use an MP4, MOV, or M4V walkthrough.",
            )

        await refresh()

        job_id = uuid.uuid4().hex
        folder = root / job_id
        folder.mkdir(parents=True, exist_ok=False)
        source = folder / f"input{ext}"

        try:
            total = 0

            with source.open("wb") as out:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)

                    if total > MAX_BYTES:
                        raise HTTPException(
                            413,
                            "Video must be 60 MB or smaller.",
                        )

                    out.write(chunk)

            if total == 0:
                raise HTTPException(
                    400,
                    "The video file is empty.",
                )

            details = await run_in_threadpool(
                inspect_video,
                source,
            )

            frame_count = await run_in_threadpool(
                extract_frames,
                source,
                folder / "frames",
            )

            manifest = {
                "id": job_id,
                "name": filename[:128],
                "status": "queued",
                "message": "Waiting for a cloud GPU worker.",
                "created_at": utc_now(),
                "frame_count": frame_count,
                **details,
            }

            save_json(folder / "manifest.json", manifest)
            await volume.commit.aio()
            await enqueue(job_id)

            return manifest

        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            await volume.commit.aio()
            raise

        finally:
            await video.close()

    async def load_manifest(
        job_id: str,
    ) -> tuple[Path, dict]:
        await refresh()
        folder = get_job_dir(root, job_id)
        path = folder / "manifest.json"

        if not path.is_file():
            raise HTTPException(
                404,
                "Reconstruction not found",
            )

        try:
            return folder, json.loads(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            raise HTTPException(
                500,
                "Could not read reconstruction state",
            )

    @api.get(
        "/api/jobs/{job_id}",
        dependencies=[Depends(require_access)],
    )
    async def status(job_id: str):
        _, manifest = await load_manifest(job_id)
        return manifest

    @api.get(
        "/api/jobs/{job_id}/{resolution}.ply",
        dependencies=[Depends(require_access)],
    )
    async def ply(
        job_id: str,
        resolution: str,
    ):
        if resolution not in {"preview", "full"}:
            raise HTTPException(404, "PLY not found")

        folder, manifest = await load_manifest(job_id)

        if manifest.get("status") != "completed":
            raise HTTPException(
                409,
                "Reconstruction is still processing",
            )

        filename = (
            "preview.ply"
            if resolution == "preview"
            else "full.ply"
        )

        path = folder / filename

        if not path.is_file():
            raise HTTPException(
                404,
                "PLY not available",
            )

        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=(
                f"worldpilot-{job_id[:8]}-{resolution}.ply"
            ),
            content_disposition_type="attachment",
        )

    # --------------------------------------------------
    # SAM ANALYSIS HELPERS
    # --------------------------------------------------

    def analysis_paths(
        folder: Path,
        concept: str,
        frame_index: int,
    ):
        digest = hashlib.sha256(
            concept.lower().encode("utf-8")
        ).hexdigest()[:12]

        stem = f"frame-{frame_index:03d}-{digest}"
        base = folder / "analysis"

        return (
            base / f"{stem}.json",
            base / f"{stem}.request.json",
        )

    def validate_analysis(
        manifest: dict,
        concept: str,
        frame_index: int,
    ) -> str:
        concept = concept.strip().lower()

        if (
            not concept
            or len(concept) > 120
            or any(ord(ch) < 32 for ch in concept)
        ):
            raise HTTPException(
                400,
                "Enter a short object noun phrase.",
            )

        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
        ):
            raise HTTPException(
                400,
                "Invalid frame index.",
            )

        if manifest.get("status") != "completed":
            raise HTTPException(
                409,
                "Finish reconstruction before spatial analysis.",
            )

        if not manifest.get("spatial_ready"):
            raise HTTPException(
                409,
                "This job has no spatial data. "
                "Reconstruct a new video.",
            )

        count = int(
            manifest.get("spatial_frames") or 0
        )

        if not 0 <= frame_index < count:
            raise HTTPException(
                400,
                f"Choose frame 0 through {max(0, count - 1)}.",
            )

        return concept

    def completed_response(
        folder: Path,
        result_file: Path,
        frame_index: int,
    ) -> dict:
        try:
            result = json.loads(
                result_file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                500,
                "Could not read spatial analysis result",
            ) from exc

        image_path = (
            folder
            / "aligned_frames"
            / f"{frame_index:03d}.jpg"
        )

        image_url = None

        if not result_file.name.startswith("fusion-") and image_path.is_file():
            image_url = (
                "data:image/jpeg;base64,"
                + base64.b64encode(
                    image_path.read_bytes()
                ).decode("ascii")
            )

        return {
            "status": "completed",
            "result": result,
            "frame_image_url": image_url,
        }

    def load_analysis_marker(
        marker: Path,
    ) -> dict | None:
        if not marker.is_file():
            return None

        try:
            return json.loads(
                marker.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            return {
                "status": "failed",
                "error": "Invalid job state. Retry the analysis.",
            }

    async def analysis_state(
        folder: Path,
        result_file: Path,
        marker: Path,
        frame: int,
    ) -> dict:

        def pending_response() -> dict:
            response = {"status": "queued"}
            if result_file.name.startswith("fusion-"):
                progress_file = result_file.with_suffix(".progress.json")
                try:
                    progress = json.loads(progress_file.read_text(encoding="utf-8"))
                    if isinstance(progress, dict):
                        response["progress"] = progress
                except (OSError, ValueError):
                    pass
            return response

        if result_file.is_file():
            return completed_response(
                folder,
                result_file,
                frame,
            )

        state = load_analysis_marker(marker)

        if state is None:
            raise HTTPException(
                404,
                "No analysis has been requested for "
                "this frame and object.",
            )

        if state.get("status") == "failed":
            return {
                "status": "failed",
                "error": (
                    state.get("error")
                    or "Analysis failed."
                ),
            }

        if state.get("status") != "queued":
            return {
                "status": "failed",
                "error": (
                    "Unknown analysis state. Retry the analysis."
                ),
            }

        call_id = state.get("call_id")

        if call_id:
            try:
                import modal

                call = modal.FunctionCall.from_id(
                    call_id
                )

                # Nonblocking status check.
                await call.get.aio(timeout=0)

            except TimeoutError:
                return pending_response()

            except Exception:
                logger.exception(
                    "SAM worker %s failed for job %s",
                    call_id,
                    folder.name,
                )

                failure = {
                    "status": "failed",
                    "error": (
                        "SAM analysis failed. "
                        "Check Modal logs, then retry."
                    ),
                }

                save_json(
                    marker,
                    {**state, **failure},
                )

                await volume.commit.aio()
                return failure

            # The Modal worker finished successfully.
            # Its result should now be on the Volume.
            await refresh()

            if result_file.is_file():
                return completed_response(
                    folder,
                    result_file,
                    frame,
                )

            failure = {
                "status": "failed",
                "error": (
                    "SAM worker returned without a result. Retry."
                ),
            }

            save_json(marker, {**state, **failure})
            await volume.commit.aio()

            return failure

        # A request marker was committed, but the call ID
        # may not have been saved yet.
        try:
            created = datetime.fromisoformat(
                state["created_at"]
            )
        except (KeyError, ValueError, TypeError):
            created = (
                datetime.now(timezone.utc)
                - timedelta(minutes=11)
            )

        if (
            datetime.now(timezone.utc) - created
            >= timedelta(minutes=2)
        ):
            failure = {
                "status": "failed",
                "error": (
                    "Worker did not start. Retry the analysis."
                ),
            }

            save_json(marker, {**state, **failure})
            await volume.commit.aio()

            return failure

        return pending_response()

    # --------------------------------------------------
    # START SAM ANALYSIS
    # --------------------------------------------------

    @api.post(
        "/api/jobs/{job_id}/analysis",
        dependencies=[Depends(require_access)],
    )
    async def start_analysis(
        job_id: str,
        request: Request,
    ):
        if enqueue_analysis is None:
            raise HTTPException(
                503,
                "SAM analysis is not configured.",
            )

        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(
                400,
                "Expected JSON data.",
            )

        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("concept"), str)
        ):
            raise HTTPException(
                400,
                "Enter an object concept.",
            )

        folder, manifest = await load_manifest(job_id)

        frame = payload.get("frame_index", 0)

        concept = validate_analysis(
            manifest,
            payload["concept"],
            frame,
        )

        result_file, marker = analysis_paths(
            folder,
            concept,
            frame,
        )

        lock_key = f"{job_id}:{frame}:{concept}"

        lock = analysis_locks.setdefault(
            lock_key,
            asyncio.Lock(),
        )

        async with lock:
            await refresh()

            if result_file.is_file():
                return completed_response(
                    folder,
                    result_file,
                    frame,
                )

            old = load_analysis_marker(marker)

            if old and old.get("status") == "queued":
                status = await analysis_state(
                    folder,
                    result_file,
                    marker,
                    frame,
                )

                if status["status"] != "failed":
                    return status

            marker.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            attempt = uuid.uuid4().hex

            request_state = {
                "status": "queued",
                "attempt": attempt,
                "created_at": utc_now(),
                "concept": concept,
                "frame_index": frame,
            }

            save_json(marker, request_state)
            await volume.commit.aio()

            try:
                call = await enqueue_analysis(
                    job_id,
                    concept,
                    frame,
                )

                call_id = getattr(
                    call,
                    "object_id",
                    None,
                )

                if call_id:
                    request_state["call_id"] = call_id

                    save_json(
                        marker,
                        request_state,
                    )

                    await volume.commit.aio()

            except Exception:
                logger.exception(
                    "Failed to enqueue SAM analysis "
                    "for job %s",
                    job_id,
                )

                save_json(
                    marker,
                    {
                        **request_state,
                        "status": "failed",
                        "error": (
                            "Unable to start worker. Retry."
                        ),
                    },
                )

                await volume.commit.aio()

                raise HTTPException(
                    503,
                    "Could not start SAM analysis. "
                    "Please retry.",
                )

            return {
                "status": "queued",
                "concept": concept,
                "frame_index": frame,
            }

    # --------------------------------------------------
    # POLL SAM ANALYSIS
    # --------------------------------------------------

    @api.get(
        "/api/jobs/{job_id}/analysis",
        dependencies=[Depends(require_access)],
    )
    async def get_analysis(
        job_id: str,
        concept: str,
        frame_index: int = 0,
    ):
        folder, manifest = await load_manifest(job_id)

        concept = validate_analysis(
            manifest,
            concept,
            frame_index,
        )

        result_file, marker = analysis_paths(
            folder,
            concept,
            frame_index,
        )

        return await analysis_state(
            folder,
            result_file,
            marker,
            frame_index,
        )

    def fusion_request(manifest: dict, concept: str, frames: object) -> list[int]:
        available = int(manifest.get("spatial_frames") or 0)
        if not isinstance(frames, list) or not 2 <= len(frames) <= available:
            raise HTTPException(400, f"Choose 2 to {available} distinct reconstructed frames.")
        for frame in frames:
            validate_analysis(manifest, concept, frame)
        if len(set(frames)) != len(frames):
            raise HTTPException(400, "Fusion frame indices must be distinct.")
        return sorted(frames)

    def fusion_paths(folder: Path, concept: str, frames: list[int]):
        return tracked_fusion_paths(folder, concept, frames)

    @api.post("/api/jobs/{job_id}/fusion", dependencies=[Depends(require_access)])
    async def start_fusion(job_id: str, request: Request):
        if enqueue_fusion is None:
            raise HTTPException(503, "Multi-frame fusion is not configured. Redeploy Modal.")
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "Expected JSON data.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("concept"), str):
            raise HTTPException(400, "Enter an object concept.")
        folder, manifest = await load_manifest(job_id)
        frames = fusion_request(manifest, payload["concept"], payload.get("frame_indices"))
        concept = validate_analysis(manifest, payload["concept"], frames[0])
        target, marker = fusion_paths(folder, concept, frames)
        lock = analysis_locks.setdefault(f"fusion:{job_id}:{target.name}", asyncio.Lock())
        async with lock:
            await refresh()
            if target.is_file():
                return completed_response(folder, target, frames[0])
            old = load_analysis_marker(marker)
            if old and old.get("status") == "queued":
                status = await analysis_state(folder, target, marker, frames[0])
                if status["status"] != "failed":
                    return status
            target.with_suffix(".progress.json").unlink(missing_ok=True)
            marker.parent.mkdir(parents=True, exist_ok=True)
            state = {"status": "queued", "created_at": utc_now(),
                     "concept": concept, "frame_indices": frames}
            save_json(marker, state)
            await volume.commit.aio()
            try:
                call = await enqueue_fusion(job_id, concept, frames)
                if getattr(call, "object_id", None):
                    state["call_id"] = call.object_id
                    save_json(marker, state)
                    await volume.commit.aio()
            except Exception as exc:
                logger.exception("Could not start fusion worker for %s", job_id)
                save_json(marker, {**state, "status": "failed", "error": "Could not start fusion. Retry."})
                await volume.commit.aio()
                raise HTTPException(503, "Could not start fusion. Retry.") from exc
            return {"status": "queued"}

    @api.get("/api/jobs/{job_id}/fusion", dependencies=[Depends(require_access)])
    async def get_fusion(job_id: str, concept: str, frame_indices: str):
        folder, manifest = await load_manifest(job_id)
        try:
            frames = [int(value) for value in frame_indices.split(",")]
        except ValueError as exc:
            raise HTTPException(400, "Invalid fusion frame indices.") from exc
        frames = fusion_request(manifest, concept, frames)
        concept = validate_analysis(manifest, concept, frames[0])
        target, marker = fusion_paths(folder, concept, frames)
        return await analysis_state(folder, target, marker, frames[0])

    @api.get("/api/jobs/{job_id}/fusion/geometry", dependencies=[Depends(require_access)])
    async def fusion_geometry(job_id: str, concept: str, frame_indices: str):
        folder, manifest = await load_manifest(job_id)
        try:
            frames = [int(value) for value in frame_indices.split(",")]
        except ValueError as exc:
            raise HTTPException(400, "Invalid fusion frame indices.") from exc
        frames = fusion_request(manifest, concept, frames)
        concept = validate_analysis(manifest, concept, frames[0])
        target, _ = fusion_paths(folder, concept, frames)
        geometry = target.with_suffix(".geometry.npz")
        if not target.is_file() or not geometry.is_file():
            raise HTTPException(409, "Complete tracked fusion before downloading object geometry.")
        return FileResponse(geometry, media_type="application/octet-stream",
                            filename=f"25worlds-{job_id[:8]}-objects.npz")

    @api.delete(
        "/api/jobs/{job_id}",
        dependencies=[Depends(require_access)],
    )
    async def delete(job_id: str):
        folder, manifest = await load_manifest(job_id)

        if manifest.get("status") not in {
            "completed",
            "failed",
        }:
            raise HTTPException(
                409,
                "A running job cannot be removed "
                "until it completes",
            )

        shutil.rmtree(folder)
        await volume.commit.aio()

        return {"deleted": True}

    return api
