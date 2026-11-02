"""Shared, dependency-free cache names for the API and tracked-fusion worker."""
import hashlib
import json
from pathlib import Path

FUSION_VERSION = "video-world-v1"


def fusion_paths(folder: Path, concept: str, frames: list[int]):
    digest = hashlib.sha256(json.dumps({
        "version": FUSION_VERSION, "concept": concept.strip().lower(),
        "frame_indices": sorted(frames),
    }, sort_keys=True).encode()).hexdigest()[:12]
    target = folder / "analysis" / f"fusion-{FUSION_VERSION}-{digest}.json"
    return target, target.with_suffix(".request.json")
