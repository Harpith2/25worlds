"""WorldPilot / Modal cloud deployment, using Modal's documented ASGI + GPU patterns.

Deploy this as the API + GPU service for the separate Next.js frontend:
    modal secret create world-auth WORLD_ACCESS_CODE="<a-long-random-code>"
    modal serve modal_app.py
    modal deploy modal_app.py

One shared Volume stores uploads and PLY results. Another caches Pi3X weights.
No pretrained model weights, demo footage, or personal-room data ship in this repo.
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

ROOT = Path(__file__).parent
app = modal.App("worldpilot-reconstruct")
data = modal.Volume.from_name("worldpilot-jobs", create_if_missing=True)
weights = modal.Volume.from_name("worldpilot-model-weights", create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi>=0.115,<1", "uvicorn>=0.30,<1", "python-multipart>=0.0.12")
    .add_local_file(str(ROOT / "service.py"), remote_path="/opt/worldpilot/service.py", copy=True)
)

# Upstream Pi3X explicitly supports torch 2.5.1, torchvision 0.20.1,
# numpy 1.26.4. The official PyTorch image provides compatible CUDA libraries.
gpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("git", "libglib2.0-0", "libgl1")
    .pip_install("numpy==1.26.4", "opencv-python-headless==4.10.0.84",
                 "pillow", "plyfile", "huggingface_hub", "safetensors")
    .run_commands("git clone --depth 1 https://github.com/yyfz/Pi3.git /opt/Pi3",
                  "cd /opt/Pi3 && pip install --no-deps -e .")
    .env({"HF_HOME": "/model-cache", "PYTHONPATH": "/opt/Pi3"})
)


def update_manifest(folder: Path, status: str, message: str, **extra):
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(status=status, message=message, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    tmp = folder / "manifest.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)
    data.commit()


@app.function(image=gpu_image, gpu="L40S", volumes={"/data": data, "/model-cache": weights},
              timeout=900, max_containers=1, retries=0)
def reconstruct(job_id: str):
    """GPU worker: Pi3X pretrained RGB-only inference -> full and preview PLY.

    `max_containers=1` limits concurrent GPU spending. Pi3X is not trained or
    fine-tuned for a user's room; a new map is built at inference time.
    """
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_images_as_tensor, write_ply
    from pi3.utils.geometry import depth_normal_edge

    data.reload()
    folder = Path("/data") / job_id
    if not folder.is_dir():
        raise FileNotFoundError("Job upload missing")
    try:
        update_manifest(folder, "reconstructing", "Estimating 3D geometry from your frames.")
        frames = folder / "frames"
        imgs = load_images_as_tensor(str(frames), interval=1).to("cuda")
        if imgs.shape[0] < 2:
            raise ValueError("At least two decoded frames are required")
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
        model = model.to("cuda")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            result = model(imgs[None])
        confidence = torch.sigmoid(result["conf"][..., 0]) > 0.1
        not_edge = ~depth_normal_edge(result["local_points"], rtol=0.03, mask=confidence)
        mask = (confidence & not_edge)[0]
        if not mask.any():
            raise ValueError("No reliable 3D points were reconstructed. Try a steadier walkthrough.")
        update_manifest(folder, "exporting", "Exporting the full point cloud and browser preview.")
        full_path = folder / "full.ply"
        write_ply(result["points"][0][mask].float().cpu(),
                  imgs.permute(0, 2, 3, 1)[mask].float().cpu(), str(full_path))
        cloud = PlyData.read(str(full_path))
        vertex = cloud["vertex"].data
        total = len(vertex)
        if total == 0:
            raise ValueError("Reconstruction produced an empty point cloud")
        max_points = 75000
        if total > max_points:
            # Stable uniformly distributed sample; not a fabricated preview.
            indices = np.random.default_rng(2026).choice(total, max_points, replace=False)
            sampled = vertex[np.sort(indices)]
        else:
            sampled = vertex
        PlyData([PlyElement.describe(sampled, "vertex")], text=False).write(str(folder / "preview.ply"))
        update_manifest(folder, "completed", "Reconstruction complete. Explore or download your PLY.",
                        total_points=total, preview_points=len(sampled),
                        finished_at=datetime.now(timezone.utc).isoformat())
        weights.commit()  # Persist newly downloaded model weights.
        return {"job_id":job_id, "total_points":total}
    except Exception as exc:
        # Job-level failure is stored where the frontend can read it.
        update_manifest(folder, "failed", "Reconstruction failed.",
                        error=f"{type(exc).__name__}: {str(exc)[:350]}")
        print(traceback.format_exc())
        raise


@app.function(image=web_image, volumes={"/data": data}, secrets=[modal.Secret.from_name("world-auth")],
              timeout=300, max_containers=2)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    sys.path.insert(0, "/opt/worldpilot")
    from service import SERVICE_REVISION, create_api
    print(f"Starting reconstruction API revision {SERVICE_REVISION} (async Volume I/O)", flush=True)
    # `.spawn.aio` is the official nonblocking Modal function-call pattern.
    return create_api(volume=data, enqueue=reconstruct.spawn.aio,
                      root=Path("/data"))


@app.function(image=web_image, volumes={"/data": data}, schedule=modal.Period(hours=24))
def clean_old_jobs():
    """Remove completed/failed files after 24 hours (not active jobs)."""
    from datetime import datetime, timedelta, timezone
    import shutil
    data.reload()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    root = Path("/data")
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        manifest = folder / "manifest.json"
        if not manifest.exists():
            continue
        try:
            meta = json.loads(manifest.read_text())
            created = datetime.fromisoformat(meta["created_at"])
            if created < cutoff and meta.get("status") in ("completed", "failed"):
                shutil.rmtree(folder)
        except (OSError, ValueError, KeyError):
            continue
    data.commit()
