"""Static object fusion from SAM video identities plus calibrated 3D evidence.

No learned association model is used. Tracking is evidence, not ground truth.
Missing depth and occluded surfaces are never invented.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property, lru_cache
from pathlib import Path
import shutil
import tempfile
import uuid
import zipfile

import numpy as np
from scipy.spatial import cKDTree

from fusion_cache import FUSION_VERSION
from sam_video import atomic_json, track_concepts, unpack_mask
from spatial import fit_plane_ransac, sample_points


def subset(points: np.ndarray, limit: int = 2048) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points)-1, limit, dtype=int)]


class WorldFrames:
    """Decompress arrays ONCE into local memmaps, avoiding N full NPZ loads."""
    def __init__(self, folder: Path):
        self.folder = folder

    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sam-world-")
        self.arrays = {}
        try:
            with zipfile.ZipFile(self.folder / "spatial.npz") as saved:
                for name in ("points", "valid_mask", "images", "camera_poses", "source_frame_indices", "source_frame_names", "camera_intrinsics"):
                    if f"{name}.npy" not in saved.namelist():
                        continue
                    path = Path(self.temp.name) / f"{name}.npy"
                    with saved.open(f"{name}.npy") as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target, length=4*1024*1024)
                    self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            self.points = self.arrays["points"]
            self.valid = self.arrays["valid_mask"]
            self.images = self.arrays["images"]
            self.poses = self.arrays["camera_poses"]
            self.count, self.height, self.width, channels = self.points.shape
            if channels != 3 or self.images.shape != self.points.shape or self.valid.shape != self.points.shape[:-1]:
                raise ValueError("Saved RGB, depth validity and world-point grids differ")
            if self.poses.shape != (self.count, 4, 4) or not np.isfinite(self.poses).all():
                raise ValueError("Missing or invalid COLMAP camera-to-world poses")
            if not np.allclose(self.poses[:, 3], [0, 0, 0, 1], atol=1e-5):
                raise ValueError("Invalid camera-to-world matrix")
            rotations = self.poses[:, :3, :3]
            if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3):
                raise ValueError("COLMAP camera rotations are not orthonormal")
            self.source_indices = self.arrays.get("source_frame_indices", np.arange(self.count))
            self.intrinsics = self.arrays.get("camera_intrinsics")
            if self.intrinsics is None:
                # Existing completed COLMAP jobs already preserve these text files.
                from colmap_pipeline import read_cameras, read_images
                model = self.folder / "colmap_sparse"
                if not model.is_dir() or "source_frame_names" not in self.arrays:
                    raise ValueError("This reconstruction lacks calibrated COLMAP metadata. Reconstruct once with the tracking update.")
                cameras = read_cameras(model / "cameras.txt")
                images = {im.name: im for im in read_images(model / "images.txt")}
                values = []
                for name in self.arrays["source_frame_names"]:
                    camera = cameras[images[str(name)].camera_id]
                    values.append([camera.fx*self.width/camera.width, camera.fy*self.height/camera.height,
                                   camera.cx*self.width/camera.width, camera.cy*self.height/camera.height])
                self.intrinsics = np.asarray(values)
            if self.intrinsics.shape != (self.count, 4) or not np.isfinite(self.intrinsics).all() or (self.intrinsics[:, :2] <= 0).any():
                raise ValueError("Invalid calibrated intrinsics for the depth grid")
            if len(self.source_indices) != self.count:
                raise ValueError("Invalid saved frame mapping")
            pool = []
            for frame in range(self.count):
                flat = self.points[frame].reshape(-1, 3)
                indices = np.linspace(0, len(flat)-1, min(1000, len(flat)), dtype=int)
                valid = self.valid[frame].reshape(-1)[indices].astype(bool)
                pts = flat[indices][valid]
                pool.append(pts[np.isfinite(pts).all(-1)])
            pool = np.concatenate(pool)
            if len(pool) < 12:
                raise ValueError("Too few reconstructed points for fusion")
            self.scale = float(np.linalg.norm(np.percentile(pool, 98, axis=0)-np.percentile(pool, 2, axis=0)))
            if not np.isfinite(self.scale) or self.scale <= 1e-8:
                raise ValueError("Degenerate reconstructed geometry")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @lru_cache(maxsize=4)
    def depth(self, frame: int) -> np.ndarray:
        pose = self.poses[frame]
        depth = np.einsum("...j,j->...", self.points[frame]-pose[:3, 3], pose[:3, 2])
        return np.where(self.valid[frame] & np.isfinite(depth) & (depth > 0), depth, np.nan)

    def __exit__(self, *_):
        self.depth.cache_clear()
        for array in getattr(self, "arrays", {}).values():
            if getattr(array, "_mmap", None) is not None:
                array._mmap.close()
        if hasattr(self, "temp"):
            self.temp.cleanup()


@dataclass
class Observation:
    frame: int
    sam_id: str
    tracklet: str
    points: np.ndarray
    colors: np.ndarray
    mask: np.ndarray  # ROI, in the original aligned image coordinates
    bounds: tuple[int, int, int, int]
    confidence: float | None = None


@dataclass
class Track:
    observations: list[Observation]
    temporal_merges: int = 0
    geometric_merges: int = 0
    ambiguous: bool = False

    @cached_property
    def frames(self):
        return {o.frame for o in self.observations}

    @cached_property
    def key(self):
        return tuple(sorted((o.frame, o.sam_id, o.tracklet) for o in self.observations))

    @cached_property
    def cloud(self):
        # Association uses bounded representatives; storage uses ALL valid points.
        points = np.concatenate([subset(o.points, 10000) for o in self.observations])
        return subset(np.unique(points, axis=0), 30000)

    @cached_property
    def tree(self):
        return cKDTree(self.cloud)

    @cached_property
    def spacing(self):
        distances = self.tree.query(subset(self.cloud, 512), k=2, workers=1)[0][:, 1]
        return float(np.median(distances[np.isfinite(distances)]))

    @cached_property
    def bounds(self):
        return np.percentile(self.cloud, [2, 98], axis=0)


def projection_evidence(points: np.ndarray, target: Observation, world) -> tuple[int, int]:
    """Return mask support and observable samples. Occluded points are ignored."""
    pose = world.poses[target.frame]
    camera = (subset(points).astype(np.float64)-pose[:3, 3]) @ pose[:3, :3]
    good = np.isfinite(camera).all(-1) & (camera[:, 2] > 1e-8)
    camera = camera[good]
    if not len(camera):
        return 0, 0
    fx, fy, cx, cy = world.intrinsics[target.frame]
    u = np.rint(fx*camera[:, 0]/camera[:, 2]+cx).astype(np.int64)
    v = np.rint(fy*camera[:, 1]/camera[:, 2]+cy).astype(np.int64)
    inside = (u >= 0) & (u < world.width) & (v >= 0) & (v < world.height)
    u, v, z = u[inside], v[inside], camera[inside, 2]
    if not len(u):
        return 0, 0
    # One contribution per camera ray; use the nearest projected surface.
    order = np.argsort(z, kind="stable")
    _, first = np.unique(v[order]*world.width+u[order], return_index=True)
    keep = order[first]
    u, v, z = u[keep], v[keep], z[keep]
    depth = world.depth(target.frame)[v, u]
    tolerance = np.maximum(world.scale*.003, np.abs(depth)*.015)
    known = np.isfinite(depth)
    observable = known & (z <= depth+tolerance)  # Behind a surface = occluded, NOT a negative vote.
    on_surface = known & (np.abs(z-depth) <= tolerance)
    left, top, right, bottom = target.bounds
    in_box = (u >= left) & (u < right) & (v >= top) & (v < bottom)
    foreground = np.zeros(len(u), dtype=bool)
    foreground[in_box] = target.mask[v[in_box]-top, u[in_box]-left]
    return int((observable & on_surface & foreground).sum()), int(observable.sum())


def nearby(a: Track, b: Track, scale: float) -> bool:
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    gap = np.maximum(np.maximum(amin-bmax, bmin-amax), 0)
    extent = max(np.linalg.norm(amax-amin), np.linalg.norm(bmax-bmin), scale*.005)
    return np.linalg.norm(gap) <= max(scale*.01, extent*.4) and np.linalg.norm(
        (amax+amin-bmax-bmin)/2) <= extent*1.25+scale*.01


def representative_observations(track: Track):
    ordered = sorted(track.observations, key=lambda o: o.frame)
    return [ordered[i] for i in np.unique(np.linspace(0, len(ordered)-1, min(3, len(ordered)), dtype=int))]


def evidence(a: Track, b: Track, world) -> dict:
    if a.frames & b.frames:
        return {"conflict": True, "reason": "distinct_instances_in_same_frame", "score": 0.}
    if not nearby(a, b, world.scale):
        return {"conflict": True, "reason": "inconsistent_world_location", "score": 0.}
    projections = ([projection_evidence(a.cloud, ob, world) for ob in representative_observations(b)] +
                   [projection_evidence(b.cloud, ob, world) for ob in representative_observations(a)])
    reliable = [(hit, visible) for hit, visible in projections if visible >= 12]
    positives = [(hit, visible) for hit, visible in reliable if hit >= 12 and hit/visible >= .55]
    conflict = bool(reliable and not positives and all(hit/visible < .05 for hit, visible in reliable))
    tol = max(world.scale*.006, 1e-7)
    aa, bb = subset(a.cloud), subset(b.cloud)
    # Account for representative spacing; otherwise subsampling phase alone
    # makes two coincident surfaces appear to have no overlap.
    cover_a = float((b.tree.query(aa, workers=1)[0] <= max(tol, b.spacing*.8)).mean())
    cover_b = float((a.tree.query(bb, workers=1)[0] <= max(tol, a.spacing*.8)).mean())
    # Containment handles a small partial view inside a much larger accumulated view.
    overlap = max(cover_a, cover_b)
    geometric = not conflict and ((overlap >= .35 and min(cover_a, cover_b) >= .03 and len(positives) >= 1)
                                 or (overlap >= .1 and len(positives) >= 2))
    return {"conflict": conflict, "reason": "visible_mask_or_depth_disagreement" if conflict else "",
            "geometric": geometric, "positive_views": len(positives), "containment": overlap,
            "score": .6*overlap+.4*max((h/v for h, v in positives), default=0.)}


def combine(a: Track, b: Track, *, geometric: bool) -> Track:
    return Track(a.observations+b.observations,
                 a.temporal_merges+b.temporal_merges+(0 if geometric else 1),
                 a.geometric_merges+b.geometric_merges+(1 if geometric else 0),
                 a.ambiguous or b.ambiguous)


def associate(observations: list[Observation], world) -> tuple[list[Track], dict]:
    """Track continuity first; globally reconcile fragments with cannot-link constraints."""
    tracks = []
    active = {}
    splits = 0
    by_frame = {}
    for observation in observations:
        by_frame.setdefault(observation.frame, []).append(observation)
    for observation in sorted(observations, key=lambda o: (o.frame, o.sam_id)):
        single = Track([observation])
        previous = active.get(observation.tracklet)
        if previous is not None:
            match = evidence(tracks[previous], single, world)
            if not match["conflict"]:
                contested = False
                # Consider all CURRENT masks before accepting a tracker ID.
                # An earlier combined mask must not absorb either of two now
                # separately visible instances just because its ID was reused.
                for rival_observation in by_frame[observation.frame]:
                    if rival_observation.sam_id == observation.sam_id:
                        continue
                    alternative = evidence(tracks[previous], Track([rival_observation]), world)
                    if alternative.get("geometric") and (not match.get("geometric") or alternative["score"] >= match["score"]-.08):
                        contested = True
                        single.ambiguous = True
                        break
                if match.get("geometric"):
                    for rival, other in enumerate(tracks):
                        if rival == previous or not other.frames & tracks[previous].frames or not nearby(other, single, world.scale):
                            continue
                        alternative = evidence(other, single, world)
                        if alternative.get("geometric") and abs(alternative["score"]-match["score"]) <= .08:
                            contested = True
                            single.ambiguous = True
                            break
                if not contested:
                    tracks[previous] = combine(tracks[previous], single, geometric=bool(match.get("geometric")))
                    continue
            splits += 1
        active[observation.tracklet] = len(tracks)
        tracks.append(single)
    initial = len(tracks)
    pair_cache = {}
    # Recompute after each merge: an intermediate view can connect two fragments
    # which could not be associated before it was accumulated.
    while True:
        edges = []
        for i, a in enumerate(tracks):
            for j in range(i+1, len(tracks)):
                b = tracks[j]
                if a.frames & b.frames or not nearby(a, b, world.scale):
                    continue
                key = tuple(sorted((a.key, b.key)))
                if key not in pair_cache:
                    pair_cache[key] = evidence(a, b, world)
                match = pair_cache[key]
                if match.get("geometric"):
                    edges.append((match["score"], i, j))
        if not edges:
            break
        edges.sort(key=lambda x: (-x[0], x[1], x[2]))
        selected = None
        for score, i, j in edges:
            contested = False
            for other_score, x, y in edges:
                if (x, y) == (i, j) or score-other_score > .08:
                    continue
                for center, partner in ((i, j), (j, i)):
                    if center in (x, y):
                        rival = y if x == center else x
                        if rival != partner and tracks[rival].frames & tracks[partner].frames:
                            contested = True
                            tracks[center].ambiguous = True
            if not contested:
                selected = (i, j)
                break
        if selected is None:
            break
        i, j = selected
        tracks[i] = combine(tracks[i], tracks[j], geometric=True)
        del tracks[j]
    return tracks, {"initial_tracklets": initial, "geometric_splits": splits,
                    "reconciled_fragments": initial-len(tracks), "ambiguous_groups": sum(t.ambiguous for t in tracks)}


def ground_records(records: list[dict], world, frames: list[int], report) -> tuple[list[Observation], dict]:
    selected = set(frames)
    last_seen, runs, assignments = {}, {}, {}
    ordered = sorted(records, key=lambda r: (r["frame"], str(r["id"])))
    # Use ALL tracked frames to determine continuity even for a 3-view grounding request.
    for record in ordered:
        frame, identity = record["frame"], str(record["id"])
        previous = last_seen.get(identity)
        if previous is None or frame != previous+1 or world.source_indices[frame] != world.source_indices[previous]+1:
            runs[identity] = runs.get(identity, 0)+1
        assignments[(frame, identity)] = f"{identity}:{runs[identity]}"
        last_seen[identity] = frame
    observations = []
    rejected = 0
    for index, frame in enumerate(frames):
        report(index, "grounding", frame)
        for record in ordered:
            if record["frame"] != frame:
                continue
            left, top, right, bottom = record["bounds"]
            mask = unpack_mask(record)
            points = world.points[frame, top:bottom, left:right]
            valid = world.valid[frame, top:bottom, left:right].astype(bool) & np.isfinite(points).all(-1)
            pixels = mask & valid
            if pixels.sum() < 12:
                rejected += 1
                continue
            observations.append(Observation(frame, str(record["id"]), assignments[(frame, str(record["id"]))],
                np.asarray(points[pixels], dtype=np.float32),
                np.asarray(world.images[frame, top:bottom, left:right][pixels], dtype=np.uint8),
                mask, tuple(record["bounds"]), record.get("confidence")))
        report(index+1, "grounding", frame)
    return observations, {"sam_masks": sum(r["frame"] in selected for r in records),
                           "grounded_observations": len(observations), "insufficient_depth_masks": rejected}


def floor_from_records(records: list[dict], world, frames: list[int]):
    selected = set(frames)
    samples = []
    for record in records:
        frame = record["frame"]
        if frame not in selected:
            continue
        left, top, right, bottom = record["bounds"]
        points = world.points[frame, top:bottom, left:right]
        valid = unpack_mask(record) & world.valid[frame, top:bottom, left:right].astype(bool) & np.isfinite(points).all(-1)
        samples.append(subset(points[valid], 2000))
    return fit_plane_ransac(np.concatenate(samples), iterations=500) if samples else None


def export_tracks(tracks: list[Track], concept: str, world, target: Path) -> tuple[list[dict], int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    objects, clouds, colors, labels = [], [], [], []
    voxel = max(world.scale*.0005, 1e-8)
    for index, track in enumerate(tracks):
        points = np.concatenate([o.points for o in track.observations])
        rgb = np.concatenate([o.colors for o in track.observations])
        _, keep = np.unique(np.floor(points.astype(np.float64)/voxel).astype(np.int64), axis=0, return_index=True)
        keep.sort()
        points, rgb = points[keep], rgb[keep]
        identity = f"object-{index+1:03d}"
        status = ("ambiguous" if track.ambiguous else "single_observation" if len(track.observations) == 1
                  else "geometry_supported" if track.geometric_merges else "tracking_supported")
        objects.append({"object_id": identity, "label": concept, "sam_confidence": None,
            "bounds_xyxy": [0, 0, 0, 0], "mask_pixels": sum(int(o.mask.sum()) for o in track.observations),
            "grounded_points": len(points), "center_xyz": np.median(points, axis=0).tolist(),
            "bbox_min_xyz": points.min(0).tolist(), "bbox_max_xyz": points.max(0).tolist(),
            "sample_xyz": sample_points(points, 5000), "unique_sample_points": len(points),
            "observation_count": len(track.observations), "frame_indices": sorted(track.frames),
            "association_overlap": None, "association_status": status,
            "tracking_ids": sorted({o.sam_id for o in track.observations}),
            "association_evidence": {"tracking_merges": track.temporal_merges, "geometry_merges": track.geometric_merges},
            "observations": [{"frame_index": o.frame, "object_id": o.sam_id, "grounded_points": len(o.points),
                              "sample_points": min(5000, len(o.points))} for o in track.observations]})
        clouds.append(points)
        colors.append(rgb)
        labels.append(np.full(len(points), index, dtype=np.int32))
    temp = target.parent / f".{uuid.uuid4().hex}.npz"
    np.savez_compressed(temp, points=np.concatenate(clouds) if clouds else np.empty((0, 3), np.float32),
        colors=np.concatenate(colors) if colors else np.empty((0, 3), np.uint8),
        object_index=np.concatenate(labels) if labels else np.empty(0, np.int32),
        object_ids=np.asarray([o["object_id"] for o in objects], dtype="U32"), version=np.asarray(FUSION_VERSION),
        concept=np.asarray(concept), scene_scale=np.asarray(world.scale), voxel_size=np.asarray(voxel))
    temp.replace(target.with_suffix(".geometry.npz"))
    return objects, sum(len(p) for p in clouds)


async def run_tracked_fusion(folder: Path, concept: str, frames: list[int], target: Path, report, commit):
    with WorldFrames(folder) as world:
        if frames != sorted(set(frames)) or any(f < 0 or f >= world.count for f in frames):
            raise ValueError("Invalid reconstructed frame selection")
        masks = await track_concepts(folder, [concept, "floor"], world.count, world.height, world.width, report, commit)
        observations, diagnostics = ground_records(masks[concept]["records"], world, frames, report)
        report(len(frames), "associating")
        tracks, association = associate(observations, world)
        diagnostics.update(association)
        report(len(frames), "exporting")
        objects, full_count = export_tracks(tracks, concept, world, target)
        floor = floor_from_records(masks["floor"]["records"], world, frames)
        warnings = ["Only observed surfaces with valid COLMAP depth are included; hidden surfaces are not completed.",
                    "SAM tracking IDs can change after occlusion or leaving view; uncertain identities remain separate."]
        if not objects:
            warnings.append("No grounded matches. Check the SAM mask and insufficient-depth counts; all reconstructed views may already be selected.")
        if diagnostics["geometric_splits"]:
            warnings.append(f'{diagnostics["geometric_splits"]} tracking association(s) were split because their geometry disagreed.')
        if diagnostics["ambiguous_groups"]:
            warnings.append(f'{diagnostics["ambiguous_groups"]} ambiguous group(s) were kept separate.')
        result = {"version": FUSION_VERSION, "source": "SAM 3.1 video tracking + COLMAP visibility-verified fusion",
            "job_id": folder.name, "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_index": frames[0], "frame_indices": frames, "image_width": world.width, "image_height": world.height,
            "tracking_frames": world.count, "object_concept": concept, "objects": objects, "floor": floor,
            "warnings": warnings, "navigation_ready": False, "association_tolerance": world.scale*.006,
            "voxel_size": world.scale*.0005, "full_geometry_points": full_count, "diagnostics": diagnostics,
            "geometry_file": target.with_suffix(".geometry.npz").name}
        atomic_json(target, result)
        commit()
        report(len(frames), "completed")
        return result
