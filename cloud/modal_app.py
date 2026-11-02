
"""25worlds: Pi3X reconstruction + SAM 3.1 + RANSAC on Modal.

Required files in cloud/:
    modal_app.py
    service.py
    spatial.py

Deploy:
    modal deploy modal_app.py
"""

import asyncio
import hashlib
import json
import re
import sys
import traceback
import uuid

from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).parent

app = modal.App("worldpilot-reconstruct")

data = modal.Volume.from_name(
    "worldpilot-jobs",
    create_if_missing=True,
)

weights = modal.Volume.from_name(
    "worldpilot-model-weights",
    create_if_missing=True,
)


# ---------------------------------------------------------
# FASTAPI IMAGE
# ---------------------------------------------------------

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi>=0.115,<1",
        "uvicorn>=0.30,<1",
        "python-multipart>=0.0.12",
    )
    .add_local_file(
        str(ROOT / "service.py"),
        remote_path="/opt/worldpilot/service.py",
        copy=True,
    )
)


# ---------------------------------------------------------
# PI3X GPU IMAGE
# ---------------------------------------------------------

gpu_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    )
    .apt_install(
        "git",
        "libglib2.0-0",
        "libgl1",
    )
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "pillow",
        "plyfile",
        "huggingface_hub",
        "safetensors",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/yyfz/Pi3.git /opt/Pi3",
        "cd /opt/Pi3 && pip install --no-deps -e .",
    )
    .env({
        "HF_HOME": "/model-cache",
        "PYTHONPATH": "/opt/Pi3",
    })
)


# ---------------------------------------------------------
# SAM 3.1 + RANSAC IMAGE
# ---------------------------------------------------------

# SAM inference runs through Meta Model API.
# RANSAC and 2D-to-3D grounding run locally on Modal CPU.
# No separate GPU is required for this analysis.

spatial_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "pillow",
        "openai>=1.99,<3",
        "meta-sam-parser",
    )
    .add_local_file(
        str(ROOT / "spatial.py"),
        remote_path="/opt/worldpilot/spatial.py",
        copy=True,
    )
)


# ---------------------------------------------------------
# JOB MANIFEST
# ---------------------------------------------------------

def update_manifest(
    folder: Path,
    status: str,
    message: str,
    **extra,
) -> None:

    path = folder / "manifest.json"

    manifest = json.loads(
        path.read_text(encoding="utf-8")
    )

    manifest.update(
        status=status,
        message=message,
        updated_at=datetime.now(
            timezone.utc
        ).isoformat(),
        **extra,
    )

    tmp = folder / "manifest.tmp"

    tmp.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    tmp.replace(path)

    data.commit()


# ---------------------------------------------------------
# PI3X RECONSTRUCTION
# ---------------------------------------------------------

@app.function(
    image=gpu_image,
    gpu="L40S",
    volumes={
        "/data": data,
        "/model-cache": weights,
    },
    timeout=900,
    max_containers=1,
    retries=0,
)
def reconstruct(job_id: str):

    import numpy as np
    import torch

    from PIL import Image
    from plyfile import PlyData, PlyElement

    from pi3.models.pi3x import Pi3X

    from pi3.utils.basic import (
        load_images_as_tensor,
        write_ply,
    )

    from pi3.utils.geometry import depth_normal_edge

    data.reload()

    folder = Path("/data") / job_id

    if not folder.is_dir():
        raise FileNotFoundError(
            "Job upload missing"
        )

    try:

        # -------------------------------------------------
        # 1. LOAD VIDEO FRAMES
        # -------------------------------------------------

        update_manifest(
            folder,
            "reconstructing",
            "Estimating 3D geometry from video frames.",
        )

        imgs = load_images_as_tensor(
            str(folder / "frames"),
            interval=1,
        ).to("cuda")

        if imgs.shape[0] < 2:
            raise ValueError(
                "At least two frames are required"
            )

        # -------------------------------------------------
        # 2. LOAD PI3X
        # -------------------------------------------------

        model = Pi3X.from_pretrained(
            "yyfz233/Pi3X"
        ).eval()

        model.disable_multimodal()

        model = model.to("cuda")

        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )

        # -------------------------------------------------
        # 3. RUN RECONSTRUCTION
        # -------------------------------------------------

        with torch.inference_mode(), torch.autocast(
            "cuda",
            dtype=dtype,
        ):
            result = model(imgs[None])

        # -------------------------------------------------
        # 4. FILTER NOISY GEOMETRY
        # -------------------------------------------------

        confident = (
            torch.sigmoid(
                result["conf"][..., 0]
            ) > 0.1
        )

        non_edge = ~depth_normal_edge(
            result["local_points"],
            rtol=0.03,
            mask=confident,
        )

        valid_gpu = (
            confident & non_edge
        )[0]

        if not valid_gpu.any().item():
            raise ValueError(
                "No reliable 3D points; "
                "record a steadier walkthrough"
            )

        update_manifest(
            folder,
            "exporting",
            "Saving point cloud and aligned spatial data.",
        )

        # -------------------------------------------------
        # 5. SAVE FULL PLY
        # -------------------------------------------------

        full_path = folder / "full.ply"

        write_ply(
            result["points"][0][valid_gpu]
            .float()
            .cpu(),

            imgs.permute(
                0, 2, 3, 1
            )[valid_gpu]
            .float()
            .cpu(),

            str(full_path),
        )

        cloud = PlyData.read(
            str(full_path)
        )

        vertex = cloud["vertex"].data

        total = len(vertex)

        if total == 0:
            raise ValueError(
                "Reconstruction produced an empty point cloud"
            )

        # -------------------------------------------------
        # 6. SAVE LIGHTWEIGHT PLY PREVIEW
        # -------------------------------------------------

        preview_limit = 75000

        if total > preview_limit:

            indices = (
                np.random.default_rng(2026)
                .choice(
                    total,
                    preview_limit,
                    replace=False,
                )
            )

            sampled = vertex[
                np.sort(indices)
            ]

        else:
            sampled = vertex

        PlyData([
            PlyElement.describe(
                sampled,
                "vertex",
            )
        ], text=False).write(
            str(folder / "preview.ply")
        )

        # -------------------------------------------------
        # 7. PRESERVE PI3X SPATIAL DATA
        # -------------------------------------------------

        # These arrays preserve the pixel-to-3D mapping.
        # SAM masks will use this correspondence.

        points = (
            result["points"][0]
            .float()
            .cpu()
            .numpy()
        )

        valid_mask = (
            valid_gpu
            .cpu()
            .numpy()
            .astype(bool)
        )

        camera_poses = (
            result["camera_poses"][0]
            .float()
            .cpu()
            .numpy()
        )

        images = (
            imgs.detach()
            .permute(0, 2, 3, 1)
            .mul(255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        n, h, w = valid_mask.shape

        if (
            points.shape != (n, h, w, 3)
            or images.shape != (n, h, w, 3)
        ):
            raise ValueError(
                "Pi3X images and point maps are not aligned"
            )

        if camera_poses.shape != (n, 4, 4):
            raise ValueError(
                "Unexpected Pi3X camera pose shape"
            )

        # -------------------------------------------------
        # 8. SAVE ALIGNED RGB FRAMES
        # -------------------------------------------------

        # IMPORTANT:
        # These frames have exactly the same pixel dimensions
        # as the Pi3X point maps.
        #
        # The frontend displays these images with SAM boxes.
        # spatial.py uses the corresponding RGB arrays
        # from spatial.npz for mask-to-3D grounding.

        aligned_dir = folder / "aligned_frames"

        aligned_dir.mkdir(
            exist_ok=True
        )

        for index, rgb in enumerate(images):

            Image.fromarray(rgb).save(
                aligned_dir / f"{index:03d}.jpg",
                format="JPEG",
                quality=90,
            )

        # -------------------------------------------------
        # 9. SAVE SPATIAL.NPZ
        # -------------------------------------------------

        tmp = folder / "spatial.partial.npz"

        np.savez_compressed(
            tmp,
            points=points,
            valid_mask=valid_mask,
            camera_poses=camera_poses,
            images=images,
        )

        tmp.replace(
            folder / "spatial.npz"
        )

        # -------------------------------------------------
        # 10. MARK RECONSTRUCTION COMPLETE
        # -------------------------------------------------

        update_manifest(
            folder,
            "completed",
            "Reconstruction complete; ready for SAM 3.1.",
            total_points=total,
            preview_points=len(sampled),
            spatial_ready=True,
            spatial_frames=n,
            spatial_height=h,
            spatial_width=w,
            finished_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        weights.commit()

        return {
            "job_id": job_id,
            "total_points": total,
            "spatial_frames": n,
        }

    except Exception as exc:

        update_manifest(
            folder,
            "failed",
            "Reconstruction failed.",
            error=(
                f"{type(exc).__name__}: "
                f"{str(exc)[:350]}"
            ),
        )

        print(
            traceback.format_exc()
        )

        raise


# ---------------------------------------------------------
# SAM 3.1 + RANSAC SPATIAL ANALYSIS
# ---------------------------------------------------------

@app.function(
    image=spatial_image,
    volumes={
        "/data": data,
    },
    secrets=[
        modal.Secret.from_name("25worlds-meta")
    ],
    timeout=360,
    max_containers=2,
    retries=0,
)
def analyze_scene(
    job_id: str,
    concept: str,
    frame_index: int = 0,
):

    import numpy as np

    # -----------------------------------------------------
    # 1. VALIDATE REQUEST
    # -----------------------------------------------------

    if not re.fullmatch(
        r"[0-9a-f]{32}",
        job_id,
    ):
        raise ValueError(
            "Invalid job ID"
        )

    concept = concept.strip()

    if (
        not concept
        or len(concept) > 120
    ):
        raise ValueError(
            "Concept must be a short noun phrase"
        )

    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
    ):
        raise ValueError(
            "frame_index must be an integer"
        )

    # -----------------------------------------------------
    # 2. LOAD RECONSTRUCTION
    # -----------------------------------------------------

    data.reload()

    folder = Path("/data") / job_id

    manifest_path = (
        folder / "manifest.json"
    )

    spatial_path = (
        folder / "spatial.npz"
    )

    if (
        not manifest_path.is_file()
        or not spatial_path.is_file()
    ):
        raise FileNotFoundError(
            "Job does not have saved Pi3X spatial data"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if manifest.get("status") != "completed":
        raise ValueError(
            "Reconstruction must finish before SAM analysis"
        )

    # -----------------------------------------------------
    # 3. LOAD ALIGNED FRAME AND POINT MAP
    # -----------------------------------------------------

    with np.load(
        spatial_path,
        allow_pickle=False,
    ) as saved:

        n = saved["points"].shape[0]

        if not 0 <= frame_index < n:
            raise ValueError(
                f"frame_index must be between 0 and {n - 1}"
            )

        rgb = saved["images"][
            frame_index
        ].copy()

        xyz = saved["points"][
            frame_index
        ].copy()

        valid = saved["valid_mask"][
            frame_index
        ].copy()

    # -----------------------------------------------------
    # 4. SAM 3.1 + RANSAC
    # -----------------------------------------------------

    sys.path.insert(
        0,
        "/opt/worldpilot",
    )

    from spatial import analyze_frame

    # SAM 3.1:
    #   identifies floor and object pixels.
    #
    # Pi3X:
    #   converts those pixels to reconstructed 3D points.
    #
    # RANSAC:
    #   fits the candidate floor plane.
    #
    # No reliable gravity vector is available here,
    # so floor orientation remains unverified.

    result = asyncio.run(
        analyze_frame(
            image_rgb=rgb,
            point_map=xyz,
            valid_map=valid,
            object_concept=concept,
            up_hint=None,
        )
    )

    # -----------------------------------------------------
    # 5. ADD SCENE METADATA
    # -----------------------------------------------------

    result["job_id"] = job_id

    result["frame_index"] = frame_index

    result["image_width"] = int(
        rgb.shape[1]
    )

    result["image_height"] = int(
        rgb.shape[0]
    )

    result["created_at"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    # -----------------------------------------------------
    # 6. SAVE ANALYSIS
    # -----------------------------------------------------

    analysis_dir = (
        folder / "analysis"
    )

    analysis_dir.mkdir(
        exist_ok=True
    )

    digest = hashlib.sha256(
        concept.lower().encode("utf-8")
    ).hexdigest()[:12]

    target = (
        analysis_dir
        / f"frame-{frame_index:03d}-{digest}.json"
    )

    temp = (
        analysis_dir
        / f".{uuid.uuid4().hex}.tmp"
    )

    temp.write_text(
        json.dumps(
            result,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temp.replace(target)

    data.commit()

    result["analysis_file"] = str(
        target.relative_to(folder)
    )

    return result


@app.function(
    image=spatial_image,
    volumes={"/data": data},
    secrets=[modal.Secret.from_name("25worlds-meta")],
    timeout=7200,
    max_containers=2,
    retries=0,
)
def fuse_scene(job_id: str, concept: str, frame_indices: list[int]):
    import numpy as np
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid job ID")
    concept = concept.strip().lower()
    if not concept or len(concept) > 120:
        raise ValueError("Invalid object concept")
    if (len(frame_indices) < 2 or
        any(isinstance(f, bool) or not isinstance(f, int) for f in frame_indices) or
        len(set(frame_indices)) != len(frame_indices)):
        raise ValueError("Choose at least 2 distinct integer frames")
    frames = sorted(frame_indices)
    data.reload()
    folder = Path("/data") / job_id
    digest = hashlib.sha256(json.dumps(
        {"concept": concept, "frame_indices": frames}, sort_keys=True,
    ).encode()).hexdigest()[:12]
    target = folder / "analysis" / f"fusion-{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    progress_file = target.with_suffix(".progress.json")

    def report(completed: int, stage: str, current_frame: int | None = None):
        progress = {
            "completed_frames": completed, "total_frames": len(frames),
            "stage": stage, "current_frame": current_frame,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temp = progress_file.parent / f".{uuid.uuid4().hex}.tmp"
        temp.write_text(json.dumps(progress), encoding="utf-8")
        temp.replace(progress_file)
        data.commit()

    report(0, "preparing")
    with np.load(folder / "spatial.npz", allow_pickle=False) as saved:
        xyz = saved["points"]
        valid = saved["valid_mask"].astype(bool) & np.isfinite(xyz).all(-1)
        if any(f < 0 or f >= len(xyz) for f in frames):
            raise ValueError("Frame index outside reconstruction")
        pts = xyz[valid]
        if len(pts) < 12:
            raise ValueError("Reconstruction has too few valid points")
        sample = pts[::max(1, len(pts) // 20000)]
        scene_scale = float(np.linalg.norm(np.percentile(sample, 98, axis=0) - np.percentile(sample, 2, axis=0)))
    del xyz, valid, pts
    if not np.isfinite(scene_scale) or scene_scale <= 1e-10:
        raise ValueError("Reconstruction has degenerate geometry")
    results = []
    digest = hashlib.sha256(concept.encode()).hexdigest()[:12]
    for index, frame in enumerate(frames):
        report(index, "analyzing", frame)
        cached = folder / "analysis" / f"frame-{frame:03d}-{digest}.json"
        if cached.is_file():
            results.append(json.loads(cached.read_text(encoding="utf-8")))
        else:
            # Reuse the existing worker locally; each result is still saved.
            results.append(analyze_scene.local(job_id, concept, frame))
        report(index + 1, "analyzing", frame)
    report(len(frames), "associating")
    sys.path.insert(0, "/opt/worldpilot")
    from spatial import fuse_observations
    result = fuse_observations(results, scene_scale=scene_scale)
    result["job_id"] = job_id
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    temp = target.parent / f".{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    temp.replace(target)
    report(len(frames), "completed")
    return result


# ---------------------------------------------------------
# FASTAPI WEB ENDPOINT
# ---------------------------------------------------------

@app.function(
    image=web_image,
    volumes={
        "/data": data,
    },
    secrets=[
        modal.Secret.from_name("world-auth")
    ],
    timeout=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():

    sys.path.insert(
        0,
        "/opt/worldpilot",
    )

    from service import (
        SERVICE_REVISION,
        create_api,
    )

    print(
        f"Starting reconstruction API "
        f"revision {SERVICE_REVISION}",
        flush=True,
    )

    return create_api(
        volume=data,

        # Existing Pi3X reconstruction worker
        enqueue=reconstruct.spawn.aio,

        # NEW: SAM 3.1 + RANSAC worker
        enqueue_analysis=analyze_scene.spawn.aio,
        enqueue_fusion=fuse_scene.spawn.aio,

        root=Path("/data"),
    )


# ---------------------------------------------------------
# AUTOMATIC JOB CLEANUP
# ---------------------------------------------------------

@app.function(
    image=web_image,
    volumes={
        "/data": data,
    },
    schedule=modal.Period(hours=24),
)
def clean_old_jobs():

    import shutil
    from datetime import timedelta

    data.reload()

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(hours=24)
    )

    root = Path("/data")

    for folder in root.iterdir():

        if not folder.is_dir():
            continue

        manifest = (
            folder / "manifest.json"
        )

        if not manifest.exists():
            continue

        try:

            meta = json.loads(
                manifest.read_text(
                    encoding="utf-8"
                )
            )

            created = datetime.fromisoformat(
                meta["created_at"]
            )

            if (
                created < cutoff
                and meta.get("status")
                in ("completed", "failed")
            ):
                shutil.rmtree(
                    folder
                )

        except (
            OSError,
            ValueError,
            KeyError,
        ):
            continue

    data.commit()


# ---------------------------------------------------------
# OPTIONAL MANUAL SAM TEST
# ---------------------------------------------------------

@app.local_entrypoint()
def test_spatial(
    job_id: str,
    concept: str = "chair",
    frame_index: int = 0,
):

    result = analyze_scene.remote(
        job_id,
        concept,
        frame_index,
    )

    print(
        json.dumps(
            {
                "job_id": result["job_id"],
                "frame_index": result["frame_index"],
                "floor_detected": (
                    result["floor"] is not None
                ),
                "objects": [
                    {
                        "label": obj["label"],
                        "center_xyz": obj["center_xyz"],
                    }
                    for obj in result["objects"]
                ],
                "warnings": result["warnings"],
                "analysis_file": result["analysis_file"],
            },
            indent=2,
        )
    )
