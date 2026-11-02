
"""25worlds: SAM 3.1 masks -> COLMAP world points; SAM-guided RANSAC.

Runs in the Modal CPU worker.
The RGB image and point map MUST have matching pixel coordinates.
"""

from __future__ import annotations

import base64
import io
import math
import os

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image


@dataclass
class SAMInstance:
    object_id: str | int
    label: str
    mask: np.ndarray
    bounds: tuple[int, int, int, int]
    sam_confidence: float | None


def unit(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)

    if (
        v.shape != (3,)
        or not np.isfinite(v).all()
        or np.linalg.norm(v) < 1e-10
    ):
        raise ValueError(
            "Expected a finite nonzero 3D vector"
        )

    return v / np.linalg.norm(v)


def sample_points(
    points: np.ndarray,
    count: int,
) -> list[list[float]]:
    if len(points) > count:
        points = points[
            np.linspace(
                0,
                len(points) - 1,
                count,
                dtype=int,
            )
        ]

    return np.round(points, 5).tolist()


def mask_png_data_uri(mask: np.ndarray) -> str:
    """Encode an exact SAM mask as a transparent PNG."""

    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError(
            "Expected a [H,W] SAM mask"
        )

    rgba = np.zeros(
        (*mask.shape, 4),
        dtype=np.uint8,
    )

    rgba[mask] = (
        255,
        176,
        153,
        180,
    )

    output = io.BytesIO()

    Image.fromarray(
        rgba,
        mode="RGBA",
    ).save(
        output,
        format="PNG",
    )

    return (
        "data:image/png;base64,"
        + base64.b64encode(
            output.getvalue()
        ).decode("ascii")
    )


def lift_mask(
    point_map: np.ndarray,
    pixel_mask: np.ndarray,
    valid_map: np.ndarray | None = None,
) -> np.ndarray:
    """Lift SAM pixels into COLMAP's 3D coordinate system."""

    points = np.asarray(point_map)
    mask = np.asarray(pixel_mask, dtype=bool)

    if (
        points.ndim != 3
        or points.shape[-1] != 3
        or points.shape[:2] != mask.shape
    ):
        raise ValueError(
            "Point map [H,W,3] and SAM mask [H,W] "
            "must align exactly"
        )

    valid = np.isfinite(points).all(axis=-1)

    if valid_map is not None:
        validity = np.asarray(valid_map)

        if validity.shape != mask.shape:
            raise ValueError(
                "valid_map shape must match image and mask"
            )

        valid &= validity.astype(bool)

    return np.asarray(
        points[mask & valid],
        dtype=np.float64,
    )


def fit_plane_ransac(
    xyz: np.ndarray,
    *,
    distance_threshold: float | None = None,
    iterations: int = 1000,
    min_points: int = 40,
    up_hint: Sequence[float] | None = None,
    seed: int = 25,
) -> dict[str, Any] | None:
    """Estimate a plane, not a verified walkable floor."""

    pts = np.asarray(
        xyz,
        dtype=np.float64,
    )

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            "xyz must be [N,3]"
        )

    pts = pts[
        np.isfinite(pts).all(axis=1)
    ]

    if len(pts) < max(3, min_points):
        return None

    extent = (
        np.percentile(pts, 95, axis=0)
        - np.percentile(pts, 5, axis=0)
    )

    diagonal = float(
        np.linalg.norm(extent)
    )

    threshold = (
        float(distance_threshold)
        if distance_threshold is not None
        else max(diagonal * 0.008, 1e-5)
    )

    if (
        not math.isfinite(threshold)
        or threshold <= 0
        or iterations < 1
    ):
        raise ValueError(
            "RANSAC threshold and iterations "
            "must be positive"
        )

    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts), dtype=bool)

    for _ in range(iterations):
        a, b, c = pts[
            rng.choice(
                len(pts),
                size=3,
                replace=False,
            )
        ]

        normal = np.cross(
            b - a,
            c - a,
        )

        length = np.linalg.norm(normal)

        if length < 1e-10:
            continue

        normal /= length

        inliers = (
            np.abs((pts - a) @ normal)
            < threshold
        )

        if inliers.sum() > best.sum():
            best = inliers

    if (
        best.sum() < min_points
        or best.mean() < 0.30
    ):
        return None

    for _ in range(2):
        cloud = pts[best]
        center = cloud.mean(axis=0)

        _, _, vh = np.linalg.svd(
            cloud - center,
            full_matrices=False,
        )

        normal = unit(vh[-1])
        offset = -float(normal @ center)

        best = (
            np.abs(pts @ normal + offset)
            < threshold
        )

        if best.sum() < min_points:
            return None

    if up_hint is not None:
        up = unit(up_hint)

        if abs(float(normal @ up)) < math.cos(
            math.radians(45)
        ):
            return None

        if float(normal @ up) < 0:
            normal = -normal
            offset = -offset

    inliers = pts[best]

    rms = float(
        np.sqrt(
            np.mean(
                (inliers @ normal + offset) ** 2
            )
        )
    )

    return {
        "normal": np.round(normal, 7).tolist(),
        "offset": float(offset),
        "distance_threshold": threshold,
        "inlier_count": int(len(inliers)),
        "input_count": int(len(pts)),
        "inlier_ratio": round(
            float(len(inliers) / len(pts)),
            4,
        ),
        "rms_residual": rms,
        "normal_oriented": up_hint is not None,
        "orientation_verified": up_hint is not None,
        "classification": "sam_guided_plane_estimate",
        "traversable": False,
        "sample_xyz": sample_points(
            inliers,
            1200,
        ),
    }


def ground_instances(
    instances: Sequence[SAMInstance],
    point_map: np.ndarray,
    valid_map: np.ndarray | None = None,
    floor: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Map SAM masks into 3D objects."""

    grounded = []

    for instance in instances:
        points = lift_mask(
            point_map,
            instance.mask,
            valid_map,
        )

        if len(points) < 12:
            continue

        center = np.median(
            points,
            axis=0,
        )

        distances = np.linalg.norm(
            points - center,
            axis=1,
        )

        points = points[
            distances
            <= np.percentile(distances, 96)
        ]

        if len(points) < 12:
            continue

        center = np.median(points, axis=0)

        obj = {
            "object_id": instance.object_id,
            "label": instance.label,
            "sam_confidence": instance.sam_confidence,
            "bounds_xyxy": list(instance.bounds),
            "mask_pixels": int(
                np.asarray(instance.mask).sum()
            ),
            "mask_png": mask_png_data_uri(
                instance.mask
            ),
            "grounded_points": len(points),
            "center_xyz": np.round(
                center,
                5,
            ).tolist(),
            "bbox_min_xyz": np.round(
                points.min(axis=0),
                5,
            ).tolist(),
            "bbox_max_xyz": np.round(
                points.max(axis=0),
                5,
            ).tolist(),
            "sample_xyz": sample_points(
                points,
                5000,
            ),
        }

        if floor is not None:
            normal = np.asarray(
                floor["normal"],
                dtype=np.float64,
            )

            signed = float(
                center @ normal
                + float(floor["offset"])
            )

            obj["distance_to_floor_plane"] = round(
                abs(signed),
                5,
            )

            if floor["normal_oriented"]:
                obj["signed_height_above_floor"] = round(
                    signed,
                    5,
                )

        grounded.append(obj)

    return grounded


async def segment_image(
    image_rgb: np.ndarray,
    concept: str,
    *,
    api_key: str | None = None,
) -> list[SAMInstance]:
    """Call Meta SAM 3.1 using the official parser."""

    from openai import AsyncOpenAI

    from meta_sam_parser import (
        SegmentationMaskRecord,
        decode_mask_to_raster,
        image_segmentation_format,
        parse_responses_stream,
    )

    rgb = np.asarray(image_rgb)

    if (
        rgb.ndim != 3
        or rgb.shape[-1] != 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError(
            "image_rgb must be uint8 RGB [H,W,3]"
        )

    if (
        not concept.strip()
        or len(concept) > 120
    ):
        raise ValueError(
            "SAM concept must be a short noun phrase"
        )

    key = (
        api_key
        or os.environ.get("MODEL_API_KEY")
    )

    if not key:
        raise RuntimeError(
            "Set MODEL_API_KEY on your Modal analysis worker"
        )

    output = io.BytesIO()

    Image.fromarray(rgb).save(
        output,
        format="JPEG",
        quality=92,
    )

    image_url = (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            output.getvalue()
        ).decode("ascii")
    )

    client = AsyncOpenAI(
        api_key=key,
        base_url="https://api.meta.ai/v1",
        timeout=120.0,
    )

    try:
        events = await client.responses.create(
            model="sam-3.1",
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": concept.strip(),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_url,
                        },
                    ],
                }
            ],
            stream=True,
            metadata={
                "mask_encoding": "one_bit",
                "include_confidence": "true",
            },
        )

        parsed = await parse_responses_stream(
            events,
            image_segmentation_format(),
        ).final_result()

    finally:
        await client.close()

    if parsed.outcome.status != "completed":
        raise RuntimeError(
            "SAM 3.1 segmentation did not complete"
        )

    h, w = rgb.shape[:2]
    instances = []

    for record in parsed.records:
        if not isinstance(
            record,
            SegmentationMaskRecord,
        ):
            continue

        bounds = record.bounds

        left = bounds.left
        top = bounds.top
        right = bounds.right
        bottom = bounds.bottom

        if not (
            0 <= left < right <= w
            and 0 <= top < bottom <= h
        ):
            raise ValueError(
                "SAM returned mask bounds outside "
                "the source image"
            )

        roi = np.frombuffer(
            decode_mask_to_raster(
                record.mask
            ),
            dtype=np.uint8,
        )

        if roi.size != (
            (bottom - top) * (right - left)
        ):
            raise ValueError(
                "SAM mask raster does not match "
                "its bounding box"
            )

        roi = roi.reshape(
            bottom - top,
            right - left,
        ).astype(bool)

        mask = np.zeros(
            (h, w),
            dtype=bool,
        )

        mask[top:bottom, left:right] = roi

        instances.append(
            SAMInstance(
                object_id=record.object_id,
                label=concept.strip(),
                mask=mask,
                bounds=(
                    left,
                    top,
                    right,
                    bottom,
                ),
                sam_confidence=record.confidence,
            )
        )

    return instances


async def analyze_frame(
    image_rgb: np.ndarray,
    point_map: np.ndarray,
    valid_map: np.ndarray | None,
    *,
    object_concept: str,
    up_hint: Sequence[float] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run SAM grounding and RANSAC on one frame."""

    if (
        np.asarray(image_rgb).shape[:2]
        != np.asarray(point_map).shape[:2]
    ):
        raise ValueError(
            "SAM frame and COLMAP point map "
            "must have identical pixels"
        )

    floor_instances = await segment_image(
        image_rgb,
        "floor",
        api_key=api_key,
    )

    floor_points = [
        lift_mask(
            point_map,
            item.mask,
            valid_map,
        )
        for item in floor_instances
    ]

    all_floor = (
        np.concatenate(floor_points)
        if floor_points
        else np.empty((0, 3))
    )

    plane = fit_plane_ransac(
        all_floor,
        up_hint=up_hint,
    )

    # A floor-only alignment request needs just one SAM call.
    detected = floor_instances if object_concept.strip().casefold() == "floor" else await segment_image(
        image_rgb,
        object_concept,
        api_key=api_key,
    )

    objects = ground_instances(
        detected,
        point_map,
        valid_map,
        plane,
    )

    warnings = []

    if plane is None:
        warnings.append(
            "SAM-guided RANSAC could not identify "
            "a reliable floor candidate."
        )

    elif up_hint is None:
        warnings.append(
            "The floor is a geometric candidate only: "
            "orientation, scale and traversability "
            "are not verified."
        )

    if not objects:
        warnings.append(
            "No grounded instances for the "
            "requested object in this frame."
        )

    return {
        "source": "sam-3.1 + COLMAP + RANSAC",
        "object_concept": object_concept,
        "floor": plane,
        "floor_status": (
            "not_detected"
            if plane is None
            else (
                "unverified_floor_candidate"
                if up_hint is None
                else "orientation_verified_plane_estimate"
            )
        ),
        "objects": objects,
        "floor_sam_instances": len(floor_instances),
        "object_sam_instances": len(detected),
        "navigation_ready": False,
        "warnings": warnings,
    }


def fuse_observations(
    results: Sequence[dict[str, Any]],
    *,
    scene_scale: float,
) -> dict[str, Any]:
    """Conservatively associate static objects already in shared COLMAP space.

    Only observed SAM surface samples are merged. This is not shape completion
    or motion tracking; ambiguous or disjoint surfaces remain separate tracks.
    """
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be positive")
    tolerance = scene_scale * 0.012
    voxel_size = scene_scale * 0.001
    tracks: list[dict[str, Any]] = []
    ambiguous = 0

    def clean_points(obj):
        pts = np.asarray(obj.get("sample_xyz", []), dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 3))
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Object samples must be [N,3]")
        return pts[np.isfinite(pts).all(axis=1)]

    def coverage(a, b):
        # Bound pairwise memory and work while keeping association deterministic.
        a = a[np.linspace(0, len(a)-1, min(len(a), 768), dtype=int)]
        b = b[np.linspace(0, len(b)-1, min(len(b), 768), dtype=int)]
        near_a = np.zeros(len(a), dtype=bool)
        near_b = np.zeros(len(b), dtype=bool)
        for start in range(0, len(a), 128):
            distances = np.sum((a[start:start+128, None] - b[None]) ** 2, axis=-1)
            near_a[start:start+128] = distances.min(axis=1) <= tolerance ** 2
            near_b |= distances.min(axis=0) <= tolerance ** 2
        return float(near_a.mean()), float(near_b.mean())

    for result in sorted(results, key=lambda r: r["frame_index"]):
        frame = int(result["frame_index"])
        for obj in result.get("objects", []):
            points = clean_points(obj)
            if len(points) < 12:
                continue
            label = str(obj["label"]).strip().casefold()
            center = np.median(points, axis=0)
            extent = np.linalg.norm(np.ptp(points, axis=0))
            candidates = []
            for track_index, track in enumerate(tracks):
                if track["label"] != label or frame in track["frames"]:
                    continue
                previous = track["points"]
                old_center = np.median(previous, axis=0)
                old_extent = np.linalg.norm(np.ptp(previous, axis=0))
                if np.linalg.norm(center - old_center) > max(max(extent, old_extent) * .65, tolerance * 2):
                    continue
                gap = np.maximum(np.maximum(points.min(0) - previous.max(0), previous.min(0) - points.max(0)), 0)
                if np.linalg.norm(gap) > tolerance * 2:
                    continue
                a, b = coverage(points, previous)
                if min(a, b) < .08 or (a + b) / 2 < .2:
                    continue
                candidates.append(((a+b)/2, track_index))
            candidates.sort(reverse=True)
            if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .12:
                # Keep a distinct hypothesis rather than assigning an uncertain ID.
                ambiguous += 1
                candidates = []
            observation = {
                "frame_index": frame,
                "object_id": obj["object_id"],
                "grounded_points": int(obj.get("grounded_points", len(points))),
                "sample_points": len(points),
            }
            if candidates:
                score, index = candidates[0]
                track = tracks[index]
                track["points"] = np.concatenate([track["points"], points])
                track["frames"].add(frame)
                track["observations"].append(observation)
                track["scores"].append(score)
            else:
                tracks.append({"label": label, "points": points, "frames": {frame},
                               "observations": [observation], "scores": []})

    objects = []
    for index, track in enumerate(tracks):
        pts = track["points"]
        # Deduplicate samples at a scene-relative resolution, without metric claims.
        _, keep = np.unique(np.floor(pts / voxel_size).astype(np.int64), axis=0, return_index=True)
        unique = pts[np.sort(keep)]
        objects.append({
            "object_id": f"fused-{index + 1:03d}",
            "label": track["label"],
            "sam_confidence": None,
            "bounds_xyxy": [0, 0, 0, 0],
            "mask_pixels": 0,
            "grounded_points": len(unique),
            "center_xyz": np.round(np.median(unique, axis=0), 5).tolist(),
            "bbox_min_xyz": np.round(unique.min(0), 5).tolist(),
            "bbox_max_xyz": np.round(unique.max(0), 5).tolist(),
            "sample_xyz": sample_points(unique, 5000),
            "observation_count": len(track["observations"]),
            "frame_indices": sorted(track["frames"]),
            "observations": track["observations"],
            "unique_sample_points": len(unique),
            "association_overlap": min(track["scores"]) if track["scores"] else None,
        })
    floors = [np.asarray(r["floor"]["sample_xyz"], dtype=np.float64)
              for r in results if r.get("floor") and r["floor"].get("sample_xyz")]
    floor = fit_plane_ransac(np.concatenate(floors), iterations=500) if floors else None
    first = min(results, key=lambda r: r["frame_index"]) if results else {}
    warnings = ["Fused geometry contains observed surface samples only; hidden surfaces are not reconstructed."]
    if ambiguous:
        warnings.append(f"{ambiguous} ambiguous association(s) were kept separate to avoid merging nearby objects.")
    if not objects:
        warnings.append("No grounded objects were found in the sampled frames. Try another prompt or frame range.")
    return {
        "source": "SAM 3.1 + COLMAP multi-frame geometric fusion",
        "frame_index": first.get("frame_index", 0),
        "frame_indices": sorted(r["frame_index"] for r in results),
        "image_width": first.get("image_width", 1),
        "image_height": first.get("image_height", 1),
        "object_concept": first.get("object_concept", ""),
        "objects": objects,
        "floor": floor,
        "warnings": warnings,
        "association_tolerance": tolerance,
        "voxel_size": voxel_size,
        "navigation_ready": False,
    }
