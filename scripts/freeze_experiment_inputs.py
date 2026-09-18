"""Create one immutable input bundle from a completed reference run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_core.artifacts import FROZEN_INPUT_FILES, load_frozen_inputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(source: Path, destination: Path) -> None:
    source_config = source / "config" if (source / "config").is_dir() else source
    status_path = source_config / "completion_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise ValueError(f"Source run is not complete: {status.get('status')}")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in FROZEN_INPUT_FILES:
        source_path = source_config / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target = destination / name
        shutil.copy2(source_path, target)
        hashes[name] = sha256(target)
    load_frozen_inputs(str(destination))
    (destination / "bundle_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "sha256": hashes,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze(args.source, args.output)
    print(f"Frozen input bundle created: {args.output}")


if __name__ == "__main__":
    main()
