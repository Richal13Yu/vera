"""Recover a portable Omega-Warp teacher config and a restricted-load checkpoint.

Loads only tensors, basic containers and the explicitly listed NumPy RNG types.
The source checkpoint is never modified. Optimizer/RNG state is unnecessary for
teacher inference and is omitted from the derived checkpoint.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import tomli_w
import torch


def fingerprint(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def primitive(value):
    if isinstance(value, str):
        return str(value)  # torch.torch_version.TorchVersion is a str subclass.
    if isinstance(value, dict):
        return {primitive(key): primitive(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(primitive(item) for item in value)
    if isinstance(value, list):
        return [primitive(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    allowed_names = {"torch.torch_version.TorchVersion", "numpy.core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype"}
    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)) - allowed_names
    if unknown:
        raise ValueError(f"Checkpoint contains unsupported pickle globals: {sorted(unknown)}")
    allowed = [
        torch.torch_version.TorchVersion,
        (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray, np.dtype, type(np.dtype("uint32")),
    ]
    with torch.serialization.safe_globals(allowed):
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if source.get("project") != "vggt-omega-warp":
        raise ValueError("Expected a vggt-omega-warp checkpoint")
    expected = source["backbone"]["sha256"]
    if fingerprint(args.backbone) != expected:
        raise ValueError("Backbone SHA-256 does not match the Warp training checkpoint")
    skip = {"optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state", "rng_state_by_rank"}
    inference = primitive({key: value for key, value in source.items() if key not in skip})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "best_inference.pt"
    temporary = target.with_suffix(".pt.partial")
    torch.save(inference, temporary)
    if torch.serialization.get_unsafe_globals_in_checkpoint(temporary):
        raise ValueError("Derived inference checkpoint still requires nonstandard pickle globals")
    reloaded = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
    original_head, exported_head = source["head_state_dict"], reloaded["head_state_dict"]
    assert original_head.keys() == exported_head.keys()
    assert all(torch.equal(value, exported_head[key]) for key, value in original_head.items())
    temporary.replace(target)
    config = {key: dict(source["experiment_config"][key]) for key in ("model", "head", "loss", "inference")}
    config["model"]["backbone_checkpoint"] = str(args.backbone.resolve())
    (args.output_dir / "teacher.toml").write_text(tomli_w.dumps(config))
    record = {
        "source_checkpoint": str(args.checkpoint.resolve()), "source_sha256": fingerprint(args.checkpoint),
        "inference_checkpoint": str(target.resolve()), "inference_sha256": fingerprint(target),
        "backbone": str(args.backbone.resolve()), "backbone_sha256": expected,
        "head_tensors_identical": True, "epoch": source["epoch"], "global_step": source["global_step"],
    }
    (args.output_dir / "teacher_artifacts.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
