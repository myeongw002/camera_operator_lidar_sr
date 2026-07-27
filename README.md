# Camera-Guided Local Operator LiDAR SR

The repository implements a 64→16 synthetic-input LiDAR super-resolution baseline, a camera-guided teacher, and anchor-operator distillation primitives. The student never accepts camera tensors; the teacher samples depth features only at observed 16-beam points.

## Fixed data contract

`prepare_range_images.py` emits one frame directory with target/input range, intensity, validity arrays and `meta.npz`. `meta.npz` holds the versioned input/target beam angles, azimuth bin centres, `K`, and `T_cam_lidar`. The current KITTI configuration is an MVP synthetic beam table; replace it with the calibrated scanner table before reporting sensor-specific results.

L0 uses the full-FOV 16-row contract `[0, 4, 8, 13, 17, 21, 25, 29, 34, 38, 42, 46, 50, 55, 59, 63]`. Existing processed data created with the prior row list is stale for L0; rerun `prepare_range_images.py` before training or evaluating L0.

Generated rows are the 48 target rows not exactly present in the synthetic 16-row input. During inference, `fuse_observed_rows` preserves the real observed rows.

## Run

Inside the supplied container after `pip install -e .`:

```bash
python -m pytest -q
python scripts/prepare_range_images.py --scan-root /data/velodyne --output-root /data/processed/00 --target-elevation elevations.npy --input-row-indices 0,4,8,13,17,21,25,29,34,38,42,46,50,55,59,63
python scripts/train_student.py --dataset-root /data/processed --output-root outputs --experiment-name student_baseline --seed 42
python scripts/infer.py --checkpoint outputs/student_baseline/seed_42/checkpoints/best.ckpt --frame-root /data/processed/00/000000 --output results/frame.npz
```

An L0-only run uses the same preparation contract, then trains and evaluates
the relation checkpoint directly:

```bash
python scripts/prepare_range_images.py ... --input-row-indices 0,4,8,13,17,21,25,29,34,38,42,46,50,55,59,63
python scripts/train_relation_l0.py --dataset-root /data/processed --train-split splits/train.txt --val-split splits/validation.txt --output-root outputs --experiment-name relation_l0 --seed 42
python scripts/evaluate_relation.py --checkpoint outputs/relation_l0/seed_42/checkpoints/best.ckpt --dataset-root /data/processed --split-file splits/test.txt --output-root outputs/relation_l0_eval --device cpu --distance-bins 0 10 20 40 inf
python scripts/infer.py --checkpoint outputs/relation_l0/seed_42/checkpoints/best.ckpt --frame-root /data/processed/00/000000 --output results/relation_l0.npz
```

The L0 row contract requires regenerating processed data created with the old
input-row list. The recommended run always supplies both train and validation
splits, so `best.ckpt` is produced for the evaluate and infer commands above.
If validation is omitted, only `last.ckpt` may be written; use that checkpoint
instead or rerun training with a validation split.

## Resumable experiments

The three training entry points write isolated experiments under
`<output-root>/<experiment-name>/seed_<seed>/`.  `--output` is not supported;
use `--output-root`, `--experiment-name`, and `--seed` for every training run.

```bash
python scripts/train_student.py \
  --dataset-root /data/processed \
  --output-root outputs \
  --experiment-name student_baseline \
  --seed 42

# Continue an existing experiment to a total of 30 epochs.
python scripts/train_student.py \
  --dataset-root /data/processed \
  --output-root outputs \
  --experiment-name student_baseline \
  --seed 42 \
  --epochs 30 \
  --resume
```

Each experiment records immutable `config.json`, `manifest.json`, append-only
`invocations.jsonl`, epoch metrics, and `checkpoints/{best,last}.ckpt`.

## Resume verification

Run the synthetic end-to-end resume verification after installing the project
and its test dependencies:

```bash
PYTHONPATH=src python scripts/verify_resume_reproducibility.py
```

The command has no `pytest-xdist` dependency. It exits non-zero on any failed
check and prints `RESUME_REPRODUCIBILITY_VERIFICATION: PASS` only on success.

## End-to-End Pipeline Runner

`run_pipeline.py` connects preprocessing, split generation, student and teacher
training, distillation, evaluation, inference, and a final artifact summary.
It records commands, stdout/stderr logs, state transitions, and summary files
under `outputs/pipelines/<pipeline-name>/`.

The pilot configuration is intended for KITTI-style data. Inspect the plan
without accessing dataset paths with:

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/pipeline/kitti_pilot.yaml \
  --dry-run --skip-path-validation
```

Run a configured pilot, resume validated completed stages, or select stages:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/pipeline/kitti_pilot.yaml
PYTHONPATH=src python scripts/run_pipeline.py --config configs/pipeline/kitti_pilot.yaml --resume
PYTHONPATH=src python scripts/run_pipeline.py --config configs/pipeline/kitti_pilot.yaml --from-stage P08_evaluate_teachers
PYTHONPATH=src python scripts/run_pipeline.py --config configs/pipeline/kitti_pilot.yaml --force-stage P03_precompute_depth
```

`dataset.max_frames_per_sequence` selects a single per-sequence frame manifest
under `<pipeline-root>/frame_manifests/`; range preparation, depth preparation,
and generated splits consume exactly that same list. `P03_precompute_depth`
skips individually valid depth outputs, repairs only missing/corrupt frames,
and uses `depth.overwrite: true` (or `--force-stage P03_precompute_depth`) to
recompute selected frames. The shipped depth backend accepts
`depth.batch_size: 1` only, so this setting is never silently ignored.

`--force-stage` invalidates the selected stage and graph-derived downstream
stages. Forced training stages overwrite only their own experiment directory;
the complete pipeline output is retained. A failed or interrupted training
stage resumes from `manifest.json` plus `checkpoints/last.ckpt` when both are
valid. Otherwise its incomplete seed directory is quarantined as
`seed_<n>.failed-<timestamp>` before a fresh run.

Resume verifies expected outputs, not merely directories. Missing range/depth
arrays, evaluation CSVs, or individual inference `.npz` files cause the owning
stage (and its downstream graph) to run again. `--dry-run` is read-only and
prints commands, dependencies, automatic student→teacher and
teacher→distillation checkpoint wiring, and expected outputs. The
`--skip-path-validation` escape hatch is permitted only with `--dry-run`.

Stage enablement is explicit: teacher evaluation requires all of
`stages.evaluate_teachers`, `evaluation.enabled`, and
`evaluation.evaluate_teachers`; SR evaluation uses the matching student and
distillation switches; inference requires both `stages.inference` and
`inference.enabled`. Invalid combinations, such as distillation selecting a
disabled teacher, are rejected while loading the config.

`summary/pipeline_summary.json` includes `scientific_checks` derived from
teacher comparison, SR summaries, and distillation metrics. These checks are
diagnostics: a surprising metric direction or inactive KD creates a warning
without converting an otherwise valid pipeline run into a failure.

The KITTI pilot uses a calibrated `configs/kitti/hdl64_elevations.npy` when it
is available. When it is absent and `dataset.auto_estimate_elevation: true`,
P01 derives a clearly marked 64-beam radian estimate from real Velodyne scan
vertical angles and writes it to
`<pipeline-root>/calibration/estimated_hdl64_elevations.npy` with provenance
JSON. Inspect or replace that estimate with a calibrated scanner table before
reporting sensor-specific results. Preflight checks frame stems, supported
calibration keys and matrix sanity, elevation shape/range, and the Python
dependencies needed by enabled preprocessing stages.

The standard KITTI Odometry downloads keep Velodyne, color images, and
calibration in separate archives. `kitti_pilot.yaml` therefore uses
`dataset.scan_root`, `dataset.image_root`, and `dataset.calibration_root`.
Mount/extract those roots so each contains either `sequences/<seq>/...` or
`dataset/sequences/<seq>/...`; no manual merging is needed.

For a dependency-free smoke test that performs actual training and evaluation on
a generated tiny dataset:

```bash
PYTHONPATH=src python scripts/run_pipeline.py --config configs/pipeline/synthetic_test.yaml
PYTHONPATH=src python scripts/verify_pipeline_runner.py
```

If a stage fails, consult `pipeline_state.json`, `commands/`, and `logs/` in
the pipeline output, correct the underlying input or configuration, then rerun
with `--resume`. A failed stage is preserved as `FAILED`; dependent stages are
recorded as `BLOCKED` rather than reported as successful.

Precompute a fixed relative-depth model with `scripts/precompute_relative_depth.py`; copy its frame output into the corresponding processed frame directory before teacher training. Teacher training requires `relative_depth.npy` and `depth_valid.npy` per processed frame. Training uses GT z-buffer visibility only for teacher supervision/KD masking; it is never supplied to either model forward path.
