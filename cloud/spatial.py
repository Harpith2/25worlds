
"""25worlds: SAM 3.1 masks -> Pi3X world points; SAM-guided RANSAC.

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
    """Lift SAM pixels into Pi3X's 3D coordinate system."""

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
                1000,
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
            "SAM frame and Pi3X point map "
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

    detected = await segment_image(
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
        "source": "sam-3.1 + Pi3X + RANSAC",
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
