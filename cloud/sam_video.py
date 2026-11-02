"""Meta SAM 3.1 video transport, exact frame mapping, and durable mask caches."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

import numpy as np
from PIL import Image

MASK_VERSION = "aligned-video-v1"
logger = logging.getLogger(__name__)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def mask_cache_path(folder: Path, concept: str) -> Path:
    key = hashlib.sha256(concept.strip().lower().encode()).hexdigest()[:16]
    return folder / "analysis" / f"{MASK_VERSION}-{key}.json"


def pack_mask(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(bool).ravel(), bitorder="little")).decode("ascii")


def unpack_mask(record: dict) -> np.ndarray:
    height, width = record["shape"]
    bits = np.frombuffer(base64.b64decode(record["bits"], validate=True), dtype=np.uint8)
    if len(bits) != (height * width + 7) // 8:
        raise ValueError("Cached mask payload has the wrong size")
    return np.unpackbits(bits, bitorder="little", count=height * width).reshape(height, width).astype(bool)


def validate_masks(value: dict, frame_count: int, height: int, width: int) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("Invalid tracking cache")
    if value.get("version") != MASK_VERSION or value.get("frame_count") != frame_count or value.get("size") != [height, width]:
        raise ValueError("Tracking cache does not match the aligned reconstruction")
    seen = set()
    for record in value["records"]:
        frame = record["frame"]
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < frame_count:
            raise ValueError("SAM returned an invalid source-video frame marker")
        left, top, right, bottom = record["bounds"]
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Tracked mask is outside the aligned frame")
        if record["shape"] != [bottom-top, right-left]:
            raise ValueError("Tracked mask shape and bounds differ")
        identity = (frame, str(record["id"]))
        if identity in seen:
            raise ValueError("Duplicate object ID in one video frame")
        seen.add(identity)
        unpack_mask(record)
    return value


def record_to_mask(record, frame_count: int, height: int, width: int) -> dict | None:
    """Use explicit video markers, not record order; crop only encoding padding."""
    from meta_sam_parser import decode_mask_to_raster
    if record.frame is None:
        raise ValueError("SAM video mask is missing its frame marker")
    frame = record.frame.frame_index
    if not 0 <= frame < frame_count:
        raise ValueError("SAM video frame is outside the encoded clip")
    b = record.bounds
    coords = [b.left, b.top, b.right, b.bottom]
    if any(not float(v).is_integer() for v in coords):
        raise ValueError("SAM mask bounds must use integer pixels")
    left, top, right, bottom = map(int, coords)
    if not (0 <= left < right <= width + width % 2 and 0 <= top < bottom <= height + height % 2):
        raise ValueError("SAM mask dimensions do not match the encoded aligned clip")
    roi = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    if roi.size != (bottom-top) * (right-left):
        raise ValueError("SAM mask raster does not match its bounds")
    roi = roi.reshape(bottom-top, right-left).astype(bool)
    clipped_right, clipped_bottom = min(right, width), min(bottom, height)
    if left >= clipped_right or top >= clipped_bottom:
        return None
    roi = roi[:clipped_bottom-top, :clipped_right-left]
    return {"frame": frame, "id": str(record.object_id),
            "bounds": [left, top, clipped_right, clipped_bottom],
            "shape": list(roi.shape), "bits": pack_mask(roi), "confidence": record.confidence}


def build_aligned_clip(folder: Path, path: Path, frame_count: int, height: int, width: int):
    for frame in range(frame_count):
        image = folder / "aligned_frames" / f"{frame:03d}.jpg"
        with Image.open(image) as rgb:
            if rgb.size != (width, height):
                raise ValueError("Aligned RGB dimensions differ from the saved depth map")
    # Encode EVERY saved view, not only the selected grounding views. Padding
    # makes H.264 yuv420p legal for odd dimensions, without shifting any pixel.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", "5", "-start_number", "0", "-i", str(folder / "aligned_frames" / "%03d.jpg"),
        "-frames:v", str(frame_count), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path)], check=True, capture_output=True, timeout=180)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=60)
    stream = json.loads(probe.stdout)["streams"][0]
    if (int(stream["nb_read_frames"]) != frame_count or
        (stream["width"], stream["height"]) != (width+width%2, height+height%2)):
        raise ValueError("Encoded tracking clip changed the frame count or pixel grid")


async def track_concepts(folder: Path, concepts: list[str], frame_count: int, height: int, width: int,
                         report, commit, api_key: str | None = None) -> dict[str, dict]:
    """One video response per concept; object IDs are scoped to that response."""
    from openai import AsyncOpenAI
    from meta_sam_parser import SegmentationMaskRecord, parse_responses_stream, video_segmentation_format

    results = {}
    pending = []
    for concept in dict.fromkeys(concepts):
        cache = mask_cache_path(folder, concept)
        try:
            cached = validate_masks(json.loads(cache.read_text()), frame_count, height, width)
            if cached.get("concept") != concept:
                raise ValueError("Cached tracking concept does not match")
            results[concept] = cached
        except (OSError, ValueError, KeyError, TypeError):
            pending.append(concept)
    if not pending:
        return results
    key = api_key or os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY in the Modal Meta secret")
    async with AsyncOpenAI(api_key=key, base_url="https://api.meta.ai/v1", timeout=1800, max_retries=0) as client:
        remote_id = None
        try:
            with tempfile.TemporaryDirectory(prefix="sam-aligned-") as temp:
                clip = Path(temp) / "aligned.mp4"
                build_aligned_clip(folder, clip, frame_count, height, width)
                with clip.open("rb") as content:
                    remote = await client.files.create(file=content, purpose="user_data")
                remote_id = remote.id
                for concept in pending:
                    stage = "tracking_floor" if concept == "floor" else "tracking_object"
                    report(0, stage, tracking_frames=0, tracking_total=frame_count)
                    events = await client.responses.create(model="sam-3.1", stream=True,
                        input=[{"type": "message", "role": "user", "content": [
                            {"type": "input_text", "text": concept},
                            {"type": "input_video", "file_id": remote_id}]}],
                        metadata={"mask_encoding": "one_bit", "include_confidence": "true"})
                    parsed = parse_responses_stream(events, video_segmentation_format())
                    last_report = time.monotonic()
                    # Snapshots are cumulative. Report progress sparingly; only
                    # a COMPLETED response becomes a reusable identity cache.
                    async for snapshot in parsed:
                        if time.monotonic() - last_report >= 8:
                            indices = [r.frame.frame_index for r in snapshot.records
                                       if isinstance(r, SegmentationMaskRecord) and r.frame is not None]
                            report(0, stage, tracking_frames=max(indices, default=-1)+1, tracking_total=frame_count)
                            last_report = time.monotonic()
                    final = await parsed.final_result()
                    if final.outcome.status != "completed":
                        raise RuntimeError(f"SAM video tracking did not complete for {concept}")
                    if any(d.severity == "error" for d in final.diagnostics):
                        raise ValueError("SAM video output contains malformed segmentation records")
                    sizes = re.findall(r"<\|box;x1=\d+;y1=\d+;x2=\d+;y2=\d+;w=(\d+);h=(\d+)(?:;c=[0-9.]+)?\|>", final.raw_output)
                    if any((int(w), int(h)) != (width+width%2, height+height%2) for w, h in sizes):
                        raise ValueError("SAM returned a different pixel grid; refusing misaligned 3D grounding")
                    records = {}
                    for record in final.records:
                        if isinstance(record, SegmentationMaskRecord):
                            value = record_to_mask(record, frame_count, height, width)
                            if value is not None:
                                records[(value["frame"], value["id"])] = value
                    result = {"version": MASK_VERSION, "concept": concept, "frame_count": frame_count,
                              "size": [height, width], "records": [records[k] for k in sorted(records)]}
                    validate_masks(result, frame_count, height, width)
                    atomic_json(mask_cache_path(folder, concept), result)
                    commit()
                    results[concept] = result
                    report(0, stage, tracking_frames=frame_count, tracking_total=frame_count)
        finally:
            if remote_id:
                try:
                    await client.files.delete(remote_id)
                except Exception:
                    logger.warning("Could not remove uploaded SAM clip; provider retention policy applies")
    return results
