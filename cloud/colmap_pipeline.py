"""Classical COLMAP SfM/MVS adapter for 25worlds (no learned reconstruction).

Pinned CLI: COLMAP 3.13.0. Public artifacts preserve the existing application
contract: full.ply, preview.ply, aligned_frames/NNN.jpg, and spatial.npz.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RegisteredImage:
    image_id: int
    camera_id: int
    name: str
    rotation: np.ndarray  # world -> camera
    translation: np.ndarray


def read_depth_map(path: Path) -> np.ndarray:
    """Read COLMAP's width&height&channels& plus planar float32 payload."""
    with path.open("rb") as stream:
        fields = []
        for _ in range(3):
            token = bytearray()
            while True:
                byte = stream.read(1)
                if not byte or len(token) > 20:
                    raise ValueError(f"Invalid COLMAP depth header: {path.name}")
                if byte == b"&":
                    break
                token.extend(byte)
            fields.append(int(token))
        width, height, channels = fields
        if width < 1 or height < 1 or channels != 1 or width * height > 20_000_000:
            raise ValueError(f"Unexpected depth dimensions: {fields}")
        payload = np.frombuffer(stream.read(), dtype="<f4")
    if payload.size != width * height:
        raise ValueError(f"Truncated COLMAP depth map: {path.name}")
    return payload.reshape(height, width).copy()


def quaternion_rotation(q: list[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid COLMAP pose quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        params = [float(v) for v in fields[4:]]
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError("Dense workspace must have undistorted PINHOLE cameras")
        if min(width, height, fx, fy) <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
            raise ValueError("Invalid undistorted camera calibration")
        cameras[camera_id] = Camera(width, height, fx, fy, cx, cy)
    return cameras


def read_images(path: Path) -> list[RegisteredImage]:
    images = []
    with path.open(encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Invalid COLMAP image pose record")
            if stream.readline() == "":  # Consume even an EMPTY keypoint line.
                raise ValueError("COLMAP image record is missing its keypoint line")
            name = fields[9].strip()
            if Path(name).name != name:
                raise ValueError("Expected flat extracted-frame filenames")
            translation = np.asarray([float(v) for v in fields[5:8]])
            if not np.isfinite(translation).all():
                raise ValueError("Invalid COLMAP camera translation")
            images.append(RegisteredImage(int(fields[0]), int(fields[8]), name,
                quaternion_rotation([float(v) for v in fields[1:5]]), translation))
    return sorted(images, key=lambda image: image.name)


def backproject_depth(depth: np.ndarray, camera: Camera, image: RegisteredImage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geometric camera-Z depth to world XYZ, matching COLMAP fusion."""
    height, width = depth.shape
    sx, sy = width / camera.width, height / camera.height
    fx, fy, cx, cy = camera.fx*sx, camera.fy*sy, camera.cx*sx, camera.cy*sy
    rows, cols = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0).astype(np.float64)
    # COLMAP dense fusion backprojects integer row/col coordinates (not ray length).
    xyz_camera = np.stack([(cols-cx)*z/fx, (rows-cy)*z/fy, z], axis=-1)
    xyz_world = (xyz_camera - image.translation) @ image.rotation
    with np.errstate(over="ignore", invalid="ignore"):
        xyz_world = xyz_world.astype(np.float32)
    valid &= np.isfinite(xyz_world).all(axis=-1)
    xyz_world[~valid] = 0
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = image.rotation.T
    pose[:3, 3] = -image.rotation.T @ image.translation
    return xyz_world, valid, pose.astype(np.float32)


def export_spatial(dense: Path, model_text: Path, folder: Path, source_names: list[str]) -> dict:
    cameras = read_cameras(model_text / "cameras.txt")
    registered = read_images(model_text / "images.txt")
    source_indices = {name: index for index, name in enumerate(source_names)}
    points, masks, poses, rgbs, names, indices = [], [], [], [], [], []
    skipped = []
    expected_shape = None
    aligned = folder / "aligned_frames"
    aligned.mkdir(exist_ok=True)
    for image in registered:
        depth_path = dense / "stereo" / "depth_maps" / f"{image.name}.geometric.bin"
        if not depth_path.is_file():
            skipped.append(image.name)
            continue
        depth = read_depth_map(depth_path)
        xyz, valid, pose = backproject_depth(depth, cameras[image.camera_id], image)
        if valid.sum() < 12:
            skipped.append(image.name)
            continue
        if expected_shape is not None and depth.shape != expected_shape:
            raise ValueError("Dense maps have different sizes; use one video camera and fixed image size")
        expected_shape = depth.shape
        height, width = depth.shape
        with Image.open(dense / "images" / image.name) as source:
            rgb_image = source.convert("RGB")
            # MVS can reduce resolution. SAM must see the depth-map pixel grid.
            rgb_image = rgb_image.resize((width, height), Image.Resampling.LANCZOS)
            rgb = np.asarray(rgb_image, dtype=np.uint8).copy()
        rgb_image.save(aligned / f"{len(points):03d}.jpg", quality=92)
        points.append(xyz); masks.append(valid); poses.append(pose); rgbs.append(rgb)
        names.append(image.name); indices.append(source_indices[image.name])
    if len(points) < 2:
        raise ValueError("MVS produced fewer than two usable depth maps. Record slower motion with more textured, overlapping views.")
    tmp = folder / "spatial.partial.npz"
    np.savez_compressed(tmp, points=np.stack(points), valid_mask=np.stack(masks),
        camera_poses=np.stack(poses), images=np.stack(rgbs),
        source_frame_indices=np.asarray(indices, dtype=np.int32), source_frame_names=np.asarray(names),
        reconstruction_engine=np.asarray("COLMAP"), camera_pose_convention=np.asarray("camera_to_world"))
    tmp.replace(folder / "spatial.npz")
    metadata = {
        "reconstruction_engine": "COLMAP 3.13.0 SfM + MVS",
        "spatial_frames": len(points), "spatial_height": expected_shape[0], "spatial_width": expected_shape[1],
        "registered_frames": len(registered), "input_frames": len(source_names),
        "source_frame_names": names, "source_frame_indices": indices,
        "unregistered_frame_names": [n for n in source_names if n not in {im.name for im in registered}],
        "unusable_depth_frame_names": skipped, "coordinate_system": "colmap_world",
        "camera_pose_convention": "camera_to_world",
    }
    (folder / "colmap_frames.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def choose_sparse_model(sparse: Path) -> Path:
    candidates = []
    for folder in sorted(sparse.iterdir()):
        if (folder / "images.bin").is_file():
            with (folder / "images.bin").open("rb") as stream:
                header = stream.read(8)
            if len(header) == 8:
                candidates.append((struct.unpack("<Q", header)[0], folder))
    if not candidates:
        raise ValueError("SfM could not register a connected camera model. Capture slower translation with textured overlapping views.")
    count, selected = max(candidates, key=lambda item: (item[0], str(item[1])))
    if count < 2:
        raise ValueError("SfM registered fewer than two cameras")
    return selected


def create_preview(full: Path, preview: Path, limit: int = 75000) -> tuple[int, int]:
    from plyfile import PlyData, PlyElement
    cloud = PlyData.read(str(full))
    vertices = cloud["vertex"].data
    if not all(name in vertices.dtype.names for name in ("x", "y", "z")):
        raise ValueError("MVS cloud is missing vertex coordinates")
    valid = np.isfinite(np.column_stack([vertices[key] for key in ("x", "y", "z")])).all(axis=1)
    if not valid.all():
        vertices = vertices[valid]
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(full))
    total = len(vertices)
    if total == 0:
        raise ValueError("MVS stereo fusion produced an empty cloud. Record more overlapping textured views.")
    indices = np.sort(np.random.default_rng(2026).choice(total, min(total, limit), replace=False))
    PlyData([PlyElement.describe(vertices[indices], "vertex")], text=False).write(str(preview))
    return total, len(indices)


def reconstruct_colmap(folder: Path, work: Path, notify: Callable[[str, str], None]) -> dict:
    """Run classical SIFT -> SfM -> undistortion -> PatchMatch -> fusion."""
    source = folder / "frames"
    source_names = sorted(path.name for path in source.glob("*.jpg"))
    if len(source_names) < 2:
        raise ValueError("At least two extracted frames are required")
    sparse = work / "sparse"
    dense = work / "dense"
    model_text = work / "dense_text"
    for path in (work, sparse, dense, model_text):
        path.mkdir(parents=True, exist_ok=True)
    database = work / "database.db"
    log_path = folder / "colmap.log"
    # Leave time for export/error reporting before Modal's two-hour deadline.
    deadline = time.monotonic() + 6600

    def run(stage: str, message: str, command: list[str]):
        notify(stage, message)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command) + "\n"); log.flush()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("COLMAP exceeded its reconstruction time budget")
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=False)
        if result.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            print(tail, flush=True)
            raise RuntimeError(f"COLMAP {command[1]} failed (exit {result.returncode}). See colmap.log / Modal logs. {tail[-300:]}")

    run("reconstructing", "COLMAP: extracting classical SIFT features.", ["colmap", "feature_extractor",
        "--database_path", str(database), "--image_path", str(source),
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0"])
    run("reconstructing", "COLMAP: matching SIFT features and verifying geometry.", ["colmap", "exhaustive_matcher",
        "--database_path", str(database), "--FeatureMatching.type", "SIFT", "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0"])
    run("reconstructing", "COLMAP: estimating camera poses and sparse structure (SfM).", ["colmap", "mapper",
        "--database_path", str(database), "--image_path", str(source), "--output_path", str(sparse)])
    model = choose_sparse_model(sparse)
    run("reconstructing", "COLMAP: undistorting registered images for dense stereo.", ["colmap", "image_undistorter",
        "--image_path", str(source), "--input_path", str(model), "--output_path", str(dense),
        "--output_type", "COLMAP", "--max_image_size", "960"])
    run("reconstructing", "COLMAP: preparing undistorted camera calibration.", ["colmap", "model_converter",
        "--input_path", str(dense / "sparse"), "--output_path", str(model_text), "--output_type", "TXT"])
    registered_count = len(read_images(model_text / "images.txt"))
    consistent = max(1, min(2, registered_count - 1))
    run("reconstructing", "COLMAP: calculating geometric depth maps (PatchMatch MVS).", ["colmap", "patch_match_stereo",
        "--workspace_path", str(dense), "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", "960",
        "--PatchMatchStereo.filter", "true", "--PatchMatchStereo.filter_min_num_consistent", str(consistent),
        "--PatchMatchStereo.gpu_index", "0", "--PatchMatchStereo.cache_size", "4"])
    fused = dense / "fused.ply"
    run("exporting", "COLMAP: fusing geometric depth maps into a colored point cloud.", ["colmap", "stereo_fusion",
        "--workspace_path", str(dense), "--workspace_format", "COLMAP", "--input_type", "geometric",
        "--output_path", str(fused), "--StereoFusion.min_num_pixels", str(min(3, registered_count)),
        "--StereoFusion.cache_size", "4"])
    shutil.copyfile(fused, folder / "full.ply")
    total, preview = create_preview(folder / "full.ply", folder / "preview.ply")
    notify("exporting", "COLMAP: exporting aligned RGB and pixel-to-world maps for SAM.")
    metadata = export_spatial(dense, model_text, folder, source_names)
    shutil.copytree(model_text, folder / "colmap_sparse", dirs_exist_ok=True)
    return {**metadata, "total_points": total, "preview_points": preview, "spatial_ready": True}
