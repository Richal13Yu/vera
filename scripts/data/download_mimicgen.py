"""Download and checksum the nine official MimicGen tasks used by VERA.

These are source HDF5 files, not the unpublished VERA MegaFlow pack.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

TASKS = (
    "coffee_d0", "coffee_d1", "square_d0", "square_d1", "square_d2",
    "stack_d0", "stack_d1", "stack_three_d0", "stack_three_d1",
)
REPO = "amandlek/mimicgen_datasets"
DEFAULT_REVISION = "33016f8a62c02334f929f2913af8fdd2a8a129e1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry, revision, root):
    path = root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = entry["lfs"]["oid"]
    size = entry["size"]
    if path.exists():
        if path.stat().st_size == size and sha256(path) == expected:
            print(f"Verified existing {path.name}", flush=True)
            return
        raise RuntimeError(f"Existing file does not match official checksum: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}"
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Server returned an unexpected byte range")
                    with partial.open("ab" if append else "wb") as stream:
                        for block in response.iter_content(chunk_size=8 << 20):
                            stream.write(block)
            if partial.stat().st_size != size or sha256(partial) != expected:
                raise RuntimeError(f"Size/checksum mismatch: {partial}")
            partial.replace(path)
            print(f"Downloaded and verified {path.name}: {size:,} bytes", flush=True)
            return
        except requests.RequestException as exc:
            if attempt == 4:
                raise
            print(f"Retry {path.name}: {type(exc).__name__}", flush=True)
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/mimicgen_raw"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Pinned dataset revision (default: the version used in our runs)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    revision = args.revision
    listing = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/core", timeout=30)
    listing.raise_for_status()
    files = {Path(entry["path"]).stem: entry for entry in listing.json()}
    entries = [files[task] for task in TASKS]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"repo": REPO, "revision": revision, "task_order": TASKS, "files": entries}
    (args.output_root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloading {len(entries)} files, {sum(e['size'] for e in entries):,} bytes, revision {revision}", flush=True)
    # Stack first makes a small representative task available for conversion checks.
    ordered = sorted(entries, key=lambda entry: entry["path"] != "core/stack_d0.hdf5")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(lambda entry: download(entry, revision, args.output_root), ordered))
    (args.output_root / "DOWNLOAD_COMPLETE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All nine source files passed SHA-256 verification.", flush=True)


if __name__ == "__main__":
    main()
