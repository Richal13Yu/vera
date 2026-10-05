"""Convert the nine official MimicGen HDF5s to RGB/trajectory-only VERA NPZs.

Preserves recorded camera images, observation timestamps and task/demo order.
This is a new pack (native source images are 84x84), not a reproduction of the
unreleased high-resolution MegaFlow pack. No optical flow is generated.
Run from the repository root with python -m scripts.data.pack_mimicgen.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image

from scripts.data.download_mimicgen import TASKS

VIEWS = ("agentview_image", "robot0_eye_in_hand_image")
LOW_DIM = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def pack_task(job):
    source_root, output_root, task, limit, quality = job
    source_root, output_root = Path(source_root), Path(output_root)
    shard = output_root / f"shard_{TASKS.index(task):06d}"
    shard.mkdir(parents=True, exist_ok=True)
    relative_paths = []
    started = time.monotonic()
    with h5py.File(source_root / "core" / f"{task}.hdf5", "r") as handle:
        demos = sorted(handle["data"], key=lambda name: int(name.split("_")[-1]))
        if len(demos) != 1000:
            raise ValueError(f"{task}: expected 1,000 demonstrations, got {len(demos)}")
        if limit is not None:
            demos = demos[:limit]
        for ordinal, demo in enumerate(demos):
            path = shard / f"{demo}.npz"
            relative_paths.append(path.relative_to(output_root).as_posix())
            source_id = f"core/{task}.hdf5:data/{demo}"
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    meta = json.loads(saved["__packed_episode_metadata__"].tobytes())
                    if meta.get("source_relative_path") != source_id:
                        raise ValueError(f"Existing episode has a different source: {path}")
                    if any(meta["rgb_entries"][view]["quality"] != quality for view in VIEWS):
                        raise ValueError(f"Existing episode has different JPEG settings: {path}")
                continue
            group = handle["data"][demo]
            count = int(group.attrs["num_samples"])
            if count < 2:
                raise ValueError(f"{source_id}: fewer than two frames")
            metadata = {
                "episode_id": demo, "source_relative_path": source_id,
                "num_frames": count, "views": list(VIEWS),
                "rgb_entries": {}, "flow_entries": {}, "trajectory_entries": {},
                "notes": ["official_recorded_rgb", "no_megaflow", "no_rerendering"],
            }
            arrays = {}
            for key in LOW_DIM:
                logical = f"low_dim/{key}"
                packed = "traj_" + logical.replace("/", "__")
                array = np.asarray(group["obs"][key], dtype=np.float32)
                if array.shape[0] != count or not np.isfinite(array).all():
                    raise ValueError(f"{source_id}: invalid {key}")
                arrays[packed] = array
                metadata["trajectory_entries"][logical] = {
                    "key": packed, "shape": list(array.shape), "dtype": str(array.dtype),
                }
            for view in VIEWS:
                frames = np.asarray(group["obs"][view])
                if frames.shape[0] != count or frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                    raise ValueError(f"{source_id}: invalid RGB field {view}: {frames.shape}/{frames.dtype}")
                keys = []
                for index, frame in enumerate(frames):
                    key = f"frame_{index:06d}_view_{view}"
                    stream = io.BytesIO()
                    Image.fromarray(frame).save(stream, format="JPEG", quality=quality, subsampling=0)
                    arrays[key] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                    keys.append(key)
                metadata["rgb_entries"][view] = {
                    "codec": "jpeg", "quality": quality, "num_frames": count,
                    "height": int(frames.shape[1]), "width": int(frames.shape[2]), "keys": keys,
                }
            arrays["__packed_episode_metadata__"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
            temporary = path.with_suffix(".npz.partial")
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(path)
            if (ordinal + 1) % 50 == 0 or ordinal == 0:
                print(f"{task}: {ordinal + 1}/{len(demos)} episodes, {time.monotonic() - started:.1f}s", flush=True)
    atomic_json(shard / "completed.json", {"task": task, "episodes": relative_paths})
    print(f"Finished {task}: {len(relative_paths)} episodes", flush=True)
    return relative_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/mimicgen_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/datasets/jacobian/mimicgen_official_rgb_warp"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if args.limit_per_task is not None and args.limit_per_task < 1:
        parser.error("--limit-per-task must be positive")
    if args.workers < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("workers must be positive; JPEG quality must be in [1,100]")
    tasks = [task for task in TASKS if task in args.tasks]
    manifest = json.loads((args.source_root / "download_manifest.json").read_text())
    for task in tasks:
        path = args.source_root / "core" / f"{task}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"Download must finish first: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.source_root), str(args.output_root), task, args.limit_per_task, args.jpeg_quality) for task in tasks]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        shards = list(executor.map(pack_task, jobs))
    entries = [entry for shard in shards for entry in shard]
    atomic_json(args.output_root / "index.json", entries)
    atomic_json(args.output_root / "provenance.json", {
        "source": manifest, "task_order": tasks, "episodes": len(entries),
        "format": "VERA RGB/trajectory-only NPZ", "jpeg_quality": args.jpeg_quality,
        "rgb": "recorded native-resolution source images; no geometric transforms",
        "flow": None, "action": "VERA SE3QuatDeltaAction from aligned low_dim observations",
        "warning": "Not the author's mimicgen_packed_v3_megaflow pack; not bitwise equivalent.",
    })
    print(f"Pack ready: {args.output_root}, {len(entries)} episodes", flush=True)


if __name__ == "__main__":
    main()
