"""HTTP interface for WorldPilot phase 1 (video -> Pi3X PLY).

No user data is shipped with this source tree. All inputs and outputs are held
in an opaque UUID directory on a mounted Modal Volume. The service is an
invitation-only hackathon demo, not a general multi-tenant media service.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

SERVICE_REVISION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
logger = logging.getLogger("uvicorn.error")

MAX_BYTES = 60 * 1024 * 1024
MAX_SECONDS = 60.0
MIN_SECONDS = 4.0
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
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
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "format=duration:stream=width,height,codec_name", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=25, check=True,
        )
        data = json.loads(proc.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        stream = data.get("streams", [{}])[0]
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    except (ValueError, IndexError, KeyError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, json.JSONDecodeError):
        raise HTTPException(400, "Could not decode this video. Export as H.264 MP4 or MOV.")
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        raise HTTPException(400, f"Walkthrough must be between {MIN_SECONDS:g} and {MAX_SECONDS:g} seconds.")
    if width < 160 or height < 160 or width > 7680 or height > 7680:
        raise HTTPException(400, "Unsupported video resolution.")
    return {"duration_seconds": round(duration, 1), "width": width, "height": height,
            "codec": stream.get("codec_name", "unknown")}


def extract_frames(video: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    # 0.4 fps -> 24 frames for a 60-second clip; no training or calibration.
    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
            "-vf", "fps=0.4,scale=640:-2", "-frames:v", "24", "-q:v", "3",
            str(output / "%04d.jpg")]
    try:
        subprocess.run(args, timeout=60, capture_output=True, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise HTTPException(400, "Frame extraction failed. Try a standard H.264 MP4.")
    count = len(list(output.glob("*.jpg")))
    if count < 2:
        raise HTTPException(400, "At least two usable video frames are required.")
    return count


def create_api(*, volume, enqueue: Callable[[str], Awaitable[object]],
               root: Path = Path("/data")) -> FastAPI:
    api = FastAPI(title="WorldPilot · Geometry Studio", version="0.1.0")

    @api.exception_handler(HTTPException)
    async def explain_http_error(request: Request, exc: HTTPException):
        if exc.status_code == 400:
            logger.warning("API revision %s: %s %s returned 400: %s",
                           SERVICE_REVISION, request.method, request.url.path, exc.detail)
        return await http_exception_handler(request, exc)

    def require_access(x_worldpilot_key: str | None = Header(default=None)) -> None:
        # Explicitly refuse an unprotected public deployment.
        import hmac
        expected = os.getenv("WORLD_ACCESS_CODE", "")
        if len(expected) < 10:
            raise HTTPException(503, "Demo access is not configured. Set WORLD_ACCESS_CODE in your Modal secret.")
        if not x_worldpilot_key or not hmac.compare_digest(x_worldpilot_key, expected):
            raise HTTPException(401, "Invalid access code. Enter the demo code to use the GPU service.")

    async def refresh() -> None:
        try:
            await volume.reload.aio()
        except FileNotFoundError:
            pass

    @api.get("/api/health")
    def health():
        return {"ok": True, "service": "worldpilot", "phase": "reconstruction",
                "revision": SERVICE_REVISION}

    @api.post("/api/jobs", dependencies=[Depends(require_access)], status_code=202)
    async def submit(video: UploadFile = File(...)):
        filename = Path(video.filename or "video.mp4").name
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, "Use an MP4, MOV, or M4V walkthrough.")
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
                        raise HTTPException(413, "Video must be 60 MB or smaller.")
                    out.write(chunk)
            if total == 0:
                raise HTTPException(400, "The video file is empty.")
            details = await run_in_threadpool(inspect_video, source)
            frame_count = await run_in_threadpool(extract_frames, source, folder / "frames")
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
            await volume.commit.aio()  # GPU worker sees the video + frames.
            await enqueue(job_id)  # Modal Function.spawn.aio(); return immediately.
            return manifest
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            await volume.commit.aio()
            raise
        finally:
            await video.close()

    async def load_manifest(job_id: str) -> tuple[Path, dict]:
        await refresh()
        folder = get_job_dir(root, job_id)
        path = folder / "manifest.json"
        if not path.is_file():
            raise HTTPException(404, "Reconstruction not found")
        try:
            return folder, json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise HTTPException(500, "Could not read reconstruction state")

    @api.get("/api/jobs/{job_id}", dependencies=[Depends(require_access)])
    async def status(job_id: str):
        _, manifest = await load_manifest(job_id)
        return manifest

    @api.get("/api/jobs/{job_id}/{resolution}.ply", dependencies=[Depends(require_access)])
    async def ply(job_id: str, resolution: str):
        if resolution not in {"preview", "full"}:
            raise HTTPException(404, "PLY not found")
        folder, manifest = await load_manifest(job_id)
        if manifest.get("status") != "completed":
            raise HTTPException(409, "Reconstruction is still processing")
        filename = "preview.ply" if resolution == "preview" else "full.ply"
        path = folder / filename
        if not path.is_file():
            raise HTTPException(404, "PLY not available")
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"worldpilot-{job_id[:8]}-{resolution}.ply",
                            content_disposition_type="attachment")

    @api.delete("/api/jobs/{job_id}", dependencies=[Depends(require_access)])
    async def delete(job_id: str):
        folder, manifest = await load_manifest(job_id)
        if manifest.get("status") not in {"completed", "failed"}:
            raise HTTPException(409, "A running job cannot be removed until it completes")
        shutil.rmtree(folder)
        await volume.commit.aio()
        return {"deleted": True}

    return api
