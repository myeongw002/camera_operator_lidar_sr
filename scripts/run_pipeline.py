#!/usr/bin/env python3
"""Run a config-driven Camera Operator LiDAR SR experiment pipeline."""
import argparse
from pathlib import Path

from camera_operator_sr.pipeline.config import load_config
from camera_operator_sr.pipeline.runner import PipelineRunner


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True, type=Path); parser.add_argument("--resume", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-stage"); parser.add_argument("--to-stage"); parser.add_argument("--only-stage"); parser.add_argument("--force-stage", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--skip-path-validation", action="store_true")
    args = parser.parse_args()
    if args.skip_path_validation and not args.dry_run: parser.error("--skip-path-validation is allowed only with --dry-run")
    config = load_config(args.config)
    runner = PipelineRunner(config, args.config, resume=args.resume, overwrite=args.overwrite, dry_run=args.dry_run, skip_path_validation=args.skip_path_validation, force_stages=set(args.force_stage))
    return runner.run(from_stage=args.from_stage, to_stage=args.to_stage, only_stage=args.only_stage)


if __name__ == "__main__": raise SystemExit(main())
