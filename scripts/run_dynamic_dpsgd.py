"""Run the isolated dynamic synchronous D-PSGD baseline."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import ray
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.dynamic_dpsgd import run
from fl_core.config import FLConfig
from fl_core.runtime import RayFLRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    fields = set(FLConfig.__dataclass_fields__)
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unsupported Dynamic D-PSGD config fields: {sorted(unknown)}")
    cfg = FLConfig(**{key: value for key, value in raw.items() if key in fields})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = str(Path(cfg.output_dir) / f"{cfg.program_name}_{stamp}")
    runtime = RayFLRuntime(cfg)
    runtime.start()
    try:
        run(runtime, cfg)
        runtime.save_completion_status("COMPLETE")
    except BaseException as exc:
        runtime.save_completion_status("FAILED", error=repr(exc))
        raise
    finally:
        runtime.shutdown()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
