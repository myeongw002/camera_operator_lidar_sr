"""Stateful pipeline runner that delegates all model work to repository CLIs."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import commands
from .state import PipelineState, now, write_json_atomic
from .stages import DEPENDENCIES, STAGES, downstream
from .summary import build_summary
from .validation import RANGE_FILES, frame_names, validate_csv_metrics, validate_depth_frame, validate_inference, validate_range_frame
from .elevation import estimate_elevations


def sha256(path: Path) -> str:
    digest = hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


@dataclass
class PipelineContext:
    config: dict
    root: Path
    config_path: Path
    device: str
    artifacts: dict = field(default_factory=lambda: {"teacher_checkpoints": {}})
    warnings: list[str] = field(default_factory=list)
    generated_elevation_file: Path | None = None
    train_entries: list[str] = field(default_factory=list)
    validation_entries: list[str] = field(default_factory=list)
    test_entries: list[str] = field(default_factory=list)

    @property
    def processed_root(self) -> Path: return Path(self.config["dataset"]["processed_root"])
    @property
    def experiments_root(self) -> Path: return self.root / "experiments"
    @property
    def evaluations_root(self) -> Path: return self.root / "evaluations"
    @property
    def summary_root(self) -> Path: return self.root / "summary"
    @property
    def splits_root(self) -> Path: return Path(self.config["splits"]["root"])
    @property
    def frame_manifests_root(self) -> Path: return self.root / "frame_manifests"
    def frame_manifest(self, sequence: str) -> Path: return self.frame_manifests_root / f"{sequence}.txt"
    def _sequence_root(self, root: str | Path, sequence: str) -> Path:
        """Support both KITTI archive layouts: <root>/sequences and <root>/dataset/sequences."""
        root = Path(root)
        direct, archive = root / "sequences" / sequence, root / "dataset" / "sequences" / sequence
        return direct if direct.exists() or not archive.exists() else archive
    def scan_directory(self, sequence: str) -> Path:
        data = self.config["dataset"]
        return self._sequence_root(data.get("scan_root") or data["raw_root"], sequence) / "velodyne"
    def image_directory(self, sequence: str) -> Path:
        data = self.config["dataset"]
        return self._sequence_root(data.get("image_root") or data["raw_root"], sequence) / "image_2"
    def calibration_file(self, sequence: str) -> Path:
        data = self.config["dataset"]
        return self._sequence_root(data.get("calibration_root") or data["raw_root"], sequence) / "calib.txt"
    @property
    def target_elevation_file(self) -> Path:
        generated = self.root / "calibration" / "estimated_hdl64_elevations.npy"
        configured = self.config["dataset"].get("target_elevation_file")
        return self.generated_elevation_file or (generated if generated.exists() else Path(configured) if configured else self.root / "calibration" / "not_configured.npy")
    @property
    def train_split(self) -> Path: return self.splits_root / self.config["splits"]["train_file"]
    @property
    def validation_split(self) -> Path: return self.splits_root / self.config["splits"]["validation_file"]
    @property
    def test_split(self) -> Path: return self.splits_root / self.config["splits"]["test_file"]


class PipelineRunner:
    def __init__(self, config: dict, config_path: Path, *, resume: bool = False, overwrite: bool = False, dry_run: bool = False, skip_path_validation: bool = False, force_stages: set[str] | None = None):
        root = Path(config["pipeline"]["output_root"]) / config["pipeline"]["name"]
        self.context = PipelineContext(config, root, config_path, str(config["pipeline"]["device"]))
        self.resume, self.overwrite, self.dry_run, self.skip_path_validation = resume or config["pipeline"].get("resume", False), overwrite or config["pipeline"].get("overwrite", False), dry_run, skip_path_validation
        self.force_stages = set(force_stages or ())
        self._active_process: subprocess.Popen | None = None
        self._active_stage: str | None = None
        unknown = self.force_stages - set(STAGES)
        if unknown: raise ValueError("unknown stage name: " + sorted(unknown)[0])
        if self.overwrite and root.exists() and not dry_run: shutil.rmtree(root)
        self.state = PipelineState(root / "pipeline_state.json", config["pipeline"]["name"], resume=self.resume)
        self.disabled = self._disabled_stages()
        self.invalidated = set().union(*(downstream(stage) | {stage} for stage in self.force_stages)) if self.force_stages else set()

    @staticmethod
    def _debug(stage: str, message: str) -> None:
        print(f"[debug:{stage}] {message}", flush=True)

    def _disabled_stages(self) -> set[str]:
        stage = self.context.config["stages"]
        disabled = set()
        mapping = {"P02_prepare_range_images": "prepare_range_images", "P03_precompute_depth": "precompute_depth", "P04_create_splits": "create_splits", "P05_train_student": "train_student", "P06_train_teacher_correct": "train_teachers", "P07_train_teacher_controls": "train_teachers", "P08_evaluate_teachers": "evaluate_teachers", "P09_train_distillation": "train_distillation", "P10_evaluate_sr": "evaluate_sr", "P11_inference": "inference"}
        for stage_id, key in mapping.items():
            if not stage.get(key, False): disabled.add(stage_id)
        cfg = self.context.config
        if not cfg["student"].get("enabled", True): disabled.add("P05_train_student")
        if not cfg["depth"].get("enabled", False): disabled.add("P03_precompute_depth")
        if not cfg["teachers"].get("correct", {}).get("enabled", False): disabled.add("P06_train_teacher_correct")
        if not any(value.get("enabled", False) for name, value in cfg["teachers"].items() if name != "correct"): disabled.add("P07_train_teacher_controls")
        if not cfg["distillation"].get("enabled", True): disabled.add("P09_train_distillation")
        evaluation = cfg["evaluation"]
        if not (evaluation.get("enabled", True) and evaluation.get("evaluate_teachers", True)): disabled.add("P08_evaluate_teachers")
        if not (evaluation.get("enabled", True) and (evaluation.get("evaluate_student", True) or evaluation.get("evaluate_distilled", True))): disabled.add("P10_evaluate_sr")
        if not cfg["inference"].get("enabled", True): disabled.add("P11_inference")
        return disabled

    def _fingerprint(self, stage_id: str) -> str:
        c = self.context
        paths: list[Path] = []
        if stage_id == "P02_prepare_range_images": paths = [c.target_elevation_file, *[c.frame_manifest(sequence) for sequence in self._all_sequences()]]
        elif stage_id == "P03_precompute_depth": paths = [c.frame_manifest(sequence) for sequence in self._all_sequences()]
        elif stage_id in {"P05_train_student"}: paths = [path for path in (c.train_split, c.validation_split) if path.exists()]
        elif stage_id in {"P06_train_teacher_correct", "P07_train_teacher_controls"}: paths = [c.artifacts.get("student_best_checkpoint")] if c.artifacts.get("student_best_checkpoint") else []
        elif stage_id == "P08_evaluate_teachers": paths = [c.artifacts.get("student_best_checkpoint"), *c.artifacts.get("teacher_checkpoints", {}).values()]
        elif stage_id == "P09_train_distillation": paths = [c.artifacts.get("student_best_checkpoint"), c.artifacts.get("teacher_checkpoints", {}).get(c.config["distillation"]["teacher"])]
        elif stage_id == "P10_evaluate_sr": paths = [c.artifacts.get("student_best_checkpoint"), c.artifacts.get("distillation_best_checkpoint"), c.test_split]
        elif stage_id == "P11_inference": paths = [c.artifacts.get("student_best_checkpoint") if c.config["inference"].get("checkpoint") == "student" else c.artifacts.get("distillation_best_checkpoint"), c.test_split]
        elif stage_id == "P12_summary": paths = [c.artifacts.get("student_best_checkpoint"), c.artifacts.get("distillation_best_checkpoint")]
        inputs = {str(path): sha256(path) for path in paths if isinstance(path, Path) and path.exists()}
        encoded = json.dumps({"stage": stage_id, "config": self.context.config, "inputs": inputs}, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _record_event(self, payload: dict) -> None:
        if self.dry_run: return
        path = self.context.root / "pipeline_events.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle: handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _run_command(self, stage_id: str, command: list[str]) -> int:
        root = self.context.root; (root / "commands").mkdir(parents=True, exist_ok=True); (root / "logs").mkdir(parents=True, exist_ok=True)
        write_json_atomic(root / "commands" / f"{stage_id}.json", {"command": command, "created_at_utc": now()})
        environment = dict(os.environ); environment["PYTHONPATH"] = "src" + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""); environment["PYTHONUNBUFFERED"] = "1"; environment.setdefault("CAMERA_OPERATOR_SR_DEBUG", "1"); environment.setdefault("PYTHONFAULTHANDLER", "1"); environment.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
        print(f"{stage_id}: LAUNCH {' '.join(command)}", flush=True)
        with (root / "logs" / f"{stage_id}.stdout.log").open("w") as stdout, (root / "logs" / f"{stage_id}.stderr.log").open("w") as stderr:
            # A process group contains model-download helpers and other
            # grandchildren as well as the script itself.  It is terminated as
            # one unit on Ctrl+C, docker stop (SIGTERM), or runner interruption.
            process = subprocess.Popen(command, cwd=".", env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
            self._active_process, self._active_stage = process, stage_id
            counts = {"stdout": 0, "stderr": 0}

            def relay(pipe, log, terminal, stream_name: str) -> None:
                assert pipe is not None
                for line in iter(pipe.readline, ""):
                    counts[stream_name] += len(line)
                    log.write(line); log.flush()
                    terminal.write(line); terminal.flush()
                pipe.close()

            workers = [
                threading.Thread(target=relay, args=(process.stdout, stdout, sys.stdout, "stdout"), daemon=True),
                threading.Thread(target=relay, args=(process.stderr, stderr, sys.stderr, "stderr"), daemon=True),
            ]
            for worker in workers: worker.start()
            previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

            def interrupt(number, _frame):
                name = signal.Signals(number).name
                print(f"{stage_id}: received {name}; stopping stage process group", file=sys.stderr, flush=True)
                self._terminate_process_group(process)
                raise KeyboardInterrupt

            for number in previous: signal.signal(number, interrupt)
            code: int | None = None
            try:
                code = process.wait()
            except KeyboardInterrupt:
                self._terminate_process_group(process)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(process, force=True); process.wait()
                raise RuntimeError(f"{stage_id} interrupted by user or container stop")
            finally:
                for number, handler in previous.items(): signal.signal(number, handler)
                for worker in workers: worker.join()
                self._active_process, self._active_stage = None, None
            assert code is not None
            if code == 0:
                print(f"{stage_id}: EXIT code=0 stdout_bytes={counts['stdout']} stderr_bytes={counts['stderr']}", flush=True)
            else:
                reason = f"signal={signal.Signals(-code).name}" if code < 0 else f"code={code}"
                print(f"{stage_id}: EXIT {reason} stdout_bytes={counts['stdout']} stderr_bytes={counts['stderr']}", file=sys.stderr, flush=True)
                if counts["stderr"] == 0:
                    print(f"{stage_id}: child produced no stderr; SIGKILL commonly indicates container/host OOM or an external kill. Inspect Docker OOM state and host kernel logs.", file=sys.stderr, flush=True)
            return code

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen, *, force: bool = False) -> None:
        if process.poll() is not None: return
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _experiment_checkpoint(self, section: dict) -> Path:
        return self.context.experiments_root / section["experiment_name"] / f"seed_{self.context.config['pipeline']['seed']}" / "checkpoints" / "best.ckpt"

    def _expected(self, stage_id: str) -> list[Path]:
        c = self.context
        if stage_id == "P01_dataset_validation": return [c.frame_manifest(sequence) for sequence in self._all_sequences()] + ([c.target_elevation_file] if c.target_elevation_file.exists() else [])
        if stage_id == "P02_prepare_range_images":
            return [c.processed_root / sequence / name / file for sequence in self._all_sequences() for name in self._frame_names(sequence) for file in RANGE_FILES]
        if stage_id == "P03_precompute_depth":
            return [c.processed_root / sequence / name / file for sequence in self._all_sequences() for name in self._frame_names(sequence) for file in ("relative_depth.npy", "depth_valid.npy")]
        if stage_id == "P04_create_splits": return [c.train_split, c.validation_split, c.test_split]
        if stage_id == "P05_train_student": return [self._experiment_checkpoint(c.config["student"])]
        if stage_id == "P06_train_teacher_correct": return [self._experiment_checkpoint(c.config["teachers"]["correct"])]
        if stage_id == "P07_train_teacher_controls": return [self._experiment_checkpoint(value) for name, value in c.config["teachers"].items() if name != "correct" and value.get("enabled")]
        if stage_id == "P08_evaluate_teachers": return [c.evaluations_root/"teachers"/"teacher_comparison.csv", c.evaluations_root/"teachers"/"teacher_superiority.json"]
        if stage_id == "P09_train_distillation": return [self._experiment_checkpoint(c.config["distillation"])]
        if stage_id == "P10_evaluate_sr":
            names = []
            evaluation = c.config["evaluation"]
            if evaluation.get("evaluate_student", True): names.append("student_baseline")
            if evaluation.get("evaluate_distilled", True): names.append(c.config["distillation"]["experiment_name"])
            return [c.evaluations_root/name/file for name in names for file in ("summary.json", "region_metrics.csv", "beam_metrics.csv", "distance_metrics.csv", "operator_metrics.csv")]
        if stage_id == "P11_inference":
            choice = c.config["inference"].get("checkpoint", "distillation")
            return [c.root / "inference" / choice / Path(entry).parent.name / f"{Path(entry).name}.npz" for entry in c.test_entries[:int(c.config["inference"].get("max_frames", 1))]]
        if stage_id == "P12_summary": return [c.summary_root/"pipeline_summary.json", c.summary_root/"artifact_index.json", c.summary_root/"metric_comparison.csv"]
        return []

    def _validate_outputs(self, stage_id: str) -> None:
        missing = [str(path) for path in self._expected(stage_id) if not path.exists()]
        if missing: raise FileNotFoundError(f"{stage_id} missing expected output: {missing[0]}")
        if stage_id in {"P05_train_student", "P06_train_teacher_correct", "P07_train_teacher_controls", "P09_train_distillation"}:
            for path in self._expected(stage_id):
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if not checkpoint.get("loss_config"): raise ValueError(f"{stage_id} checkpoint has empty loss_config")
        if stage_id == "P02_prepare_range_images":
            for sequence in self._all_sequences():
                for name in self._frame_names(sequence): validate_range_frame(self.context.processed_root / sequence / name)
        if stage_id == "P03_precompute_depth":
            for sequence in self._all_sequences():
                for name in self._frame_names(sequence): validate_depth_frame(self.context.processed_root / sequence / name)
        if stage_id == "P10_evaluate_sr":
            for path in self._expected(stage_id):
                if path.name == "summary.json": json.loads(path.read_text())
                else: validate_csv_metrics(path)
        if stage_id == "P11_inference":
            for path in self._expected(stage_id): validate_inference(path)

    def _all_sequences(self) -> list[str]:
        return list(dict.fromkeys(sequence for values in self.context.config["dataset"]["sequences"].values() for sequence in values))

    def _frame_names(self, sequence: str) -> list[str]:
        return frame_names(self.context.frame_manifest(sequence))

    def _create_splits(self) -> None:
        c, sequences = self.context, self.context.config["dataset"]["sequences"]
        if not c.config["splits"].get("generate", True):
            for path in (c.train_split, c.validation_split, c.test_split):
                if not path.exists(): raise FileNotFoundError(f"configured split does not exist: {path}")
            c.train_entries, c.validation_entries, c.test_entries = (c.train_split.read_text().splitlines(), c.validation_split.read_text().splitlines(), c.test_split.read_text().splitlines())
            return
        c.splits_root.mkdir(parents=True, exist_ok=True)
        groups = (("train", c.train_split), ("validation", c.validation_split), ("test", c.test_split))
        for group, path in groups:
            entries = []
            for sequence in sequences.get(group, []):
                entries.extend(f"{sequence}/{name}" for name in self._frame_names(sequence) if (c.processed_root / sequence / name / "target_range.npy").exists())
            if not entries: raise ValueError(f"split {group} has no processed frames")
            path.write_text("\n".join(entries) + "\n")
            setattr(c, f"{group if group != 'validation' else 'validation'}_entries", entries)
            self._debug("P04", f"split={group} entries={len(entries)} path={path}")
        c.train_entries = c.train_split.read_text().splitlines(); c.validation_entries = c.validation_split.read_text().splitlines(); c.test_entries = c.test_split.read_text().splitlines()

    def _write_frame_manifests(self) -> None:
        """Make the one selected frame set shared by P02, P03 and P04."""
        c = self.context; c.frame_manifests_root.mkdir(parents=True, exist_ok=True)
        data = c.config["dataset"]
        for sequence in self._all_sequences():
            if data["type"] in {"synthetic", "processed_synthetic"}:
                stems = sorted(path.name for path in (c.processed_root / sequence).glob("*") if path.is_dir())
            else:
                lidar = {path.stem for path in c.scan_directory(sequence).glob("*.bin")}
                images = {path.stem for extension in ("*.png", "*.jpg", "*.jpeg") for path in c.image_directory(sequence).glob(extension)}
                stems = sorted(lidar & images)
                if not stems: raise ValueError(f"sequence {sequence} has no common LiDAR/image frame stems")
                missing = 1 - len(stems) / max(len(lidar | images), 1)
                if missing > 0.5: c.warnings.append(f"sequence {sequence}: {missing:.1%} non-common raw frames")
            limit = data.get("max_frames_per_sequence")
            stems = stems[:limit] if limit is not None else stems
            if not stems: raise ValueError(f"sequence {sequence} selected frame manifest is empty")
            c.frame_manifest(sequence).write_text("\n".join(stems) + "\n")

    def _preflight(self) -> None:
        self._debug("P00", f"python={sys.version.split()[0]} requested_device={self.context.device} dataset_type={self.context.config['dataset']['type']}")
        if sys.version_info < (3, 10): raise RuntimeError("pipeline requires Python 3.10+")
        if self.context.device.startswith("cuda") and not torch.cuda.is_available():
            if self.context.config["pipeline"].get("allow_device_fallback"): self.context.device = "cpu"
            else: raise RuntimeError("pipeline requested CUDA but CUDA is unavailable")
        # These imports fail early, before an expensive preprocessing/training subprocess.
        for module in ("numpy", "torch"):
            __import__(module)
        if self.context.config["dataset"]["type"] not in {"synthetic", "processed_synthetic"}:
            __import__("PIL")
        if self.context.config["depth"].get("enabled"):
            __import__("transformers")
        for script in ("prepare_range_images.py", "precompute_relative_depth.py", "train_student.py", "train_teacher.py", "train_distill.py", "evaluate_sr.py", "evaluate_teacher.py", "infer.py"):
            if not (Path("scripts") / script).exists(): raise FileNotFoundError(Path("scripts") / script)
        if not self.skip_path_validation and not self.context.processed_root.exists() and self.context.config["dataset"]["type"] == "processed_synthetic": raise FileNotFoundError(self.context.processed_root)
        self.context.root.mkdir(parents=True, exist_ok=True)
        probe = self.context.root / ".write_probe"; probe.write_text("ok"); probe.unlink()
        self._debug("P00", f"active_device={self.context.device} cuda_available={torch.cuda.is_available()} output_root={self.context.root}")

    def _dataset_validation(self) -> None:
        self._debug("P01", f"dataset_type={self.context.config['dataset']['type']} processed_root={self.context.processed_root}")
        if self.context.config["dataset"]["type"] == "synthetic": self._create_synthetic_dataset()
        if self.context.config["dataset"]["type"] in {"processed_synthetic", "synthetic"}:
            if not any(self.context.processed_root.glob("*/*/meta.npz")): raise FileNotFoundError("processed dataset has no meta.npz frames")
            self._write_frame_manifests()
            self._debug("P01", f"processed frames validated; manifests={self.context.frame_manifests_root}")
            return
        data = self.context.config["dataset"]
        elevation_path = self.context.target_elevation_file
        if not elevation_path.exists():
            if not data.get("auto_estimate_elevation", True):
                raise FileNotFoundError(f"target elevation file missing: {elevation_path}; provide calibrated HDL-64E elevations in radians")
            scans = [path for sequence in self._all_sequences() for path in sorted(self.context.scan_directory(sequence).glob("*.bin"))[:5]]
            if not scans: raise FileNotFoundError("target elevation file missing and no KITTI scans are available for estimation")
            output = self.context.root / "calibration" / "estimated_hdl64_elevations.npy"
            output.parent.mkdir(parents=True, exist_ok=True)
            estimate = estimate_elevations(scans)
            np.save(output, estimate)
            write_json_atomic(output.with_suffix(".json"), {"method": "data_derived_1d_kmeans_vertical_angle", "units": "radians", "beam_count": 64, "source_scans": [str(path) for path in scans], "min": float(estimate.min()), "max": float(estimate.max())})
            self.context.generated_elevation_file = output
            self.context.warnings.append(f"target elevation file missing; generated data-derived estimate: {output}")
            print(f"P01_dataset_validation: generated data-derived 64-beam elevation table: {output}", flush=True)
            elevation_path = output
        elevation = np.load(elevation_path)
        if elevation.shape != (64,) or not np.isfinite(elevation).all() or np.max(np.abs(elevation)) > np.pi / 2 or np.any(np.diff(elevation) == 0): raise ValueError("target elevation must be finite ordered radians shape [64]")
        self._debug("P01", f"elevation={elevation_path} min={float(elevation.min()):.6f} max={float(elevation.max()):.6f}")
        from camera_operator_sr.data.kitti_calibration import load_kitti_calibration
        for sequence in self._all_sequences():
            scan_dir, image_dir, calibration_path = self.context.scan_directory(sequence), self.context.image_directory(sequence), self.context.calibration_file(sequence)
            for path in (scan_dir, image_dir, calibration_path):
                if not path.exists(): raise FileNotFoundError(path)
            lidar = {path.stem for path in scan_dir.glob("*.bin")}
            images = {path.stem for extension in ("*.png", "*.jpg", "*.jpeg") for path in image_dir.glob(extension)}
            self._debug("P01", f"sequence={sequence} scans={len(lidar)} images={len(images)} common={len(lidar & images)}")
            if not lidar or not images or not (lidar & images): raise ValueError(f"sequence {sequence} has no common LiDAR/image frame stems")
            calibration = load_kitti_calibration(calibration_path, data.get("camera_id", 2))
            if not np.isfinite(calibration.K).all() or not np.isfinite(calibration.T_cam_lidar).all() or abs(np.linalg.det(calibration.T_cam_lidar[:3, :3])) < 1e-6: raise ValueError(f"invalid calibration: {calibration_path}")
        self._write_frame_manifests()
        self._debug("P01", f"frame manifests written: {self.context.frame_manifests_root}")

    def _create_synthetic_dataset(self) -> None:
        """Create the tiny, deterministic processed dataset used by the shipped pilot config."""
        root = self.context.processed_root
        if any(root.glob("*/*/meta.npz")): return
        for index in range(2):
            frame = root / "00" / f"{index:06d}"; frame.mkdir(parents=True, exist_ok=True)
            width = 8; input_range = np.ones((2, width), np.float32) * (10 + index); target_range = np.ones((3, width), np.float32) * (10 + index)
            for name, value in (("input_range.npy", input_range), ("input_intensity.npy", np.zeros_like(input_range)), ("input_valid.npy", np.ones_like(input_range)), ("target_range.npy", target_range), ("target_valid.npy", np.ones_like(target_range)), ("relative_depth.npy", np.ones((16, 16), np.float32)), ("depth_valid.npy", np.ones((16, 16), np.float32))): np.save(frame / name, value)
            k = np.array([[10, 0, 8], [0, 10, 8], [0, 0, 1]], np.float32); t = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], np.float32)
            np.savez(frame / "meta.npz", input_elevation=np.array([-.2, .2], np.float32), target_elevation=np.array([-.2, 0, .2], np.float32), azimuth=np.linspace(-.3, .3, width, dtype=np.float32), K=k, T_cam_lidar=t, image_size=np.array([16, 16]))

    def _training_sections(self, stage_id: str) -> list[dict]:
        c = self.context.config
        if stage_id == "P05_train_student": return [c["student"]]
        if stage_id == "P06_train_teacher_correct": return [c["teachers"]["correct"]]
        if stage_id == "P07_train_teacher_controls": return [value for name, value in c["teachers"].items() if name != "correct" and value.get("enabled")]
        if stage_id == "P09_train_distillation": return [c["distillation"]]
        return []

    def _experiment_root(self, section: dict) -> Path:
        return self.context.experiments_root / section["experiment_name"] / f"seed_{self.context.config['pipeline']['seed']}"

    def _recover_training_mode(self, stage_id: str) -> str | None:
        """Force means overwrite; failed/running resume is safe only with a valid last checkpoint."""
        if stage_id in self.invalidated: return "overwrite"
        previous = self.state.value["stages"].get(stage_id, {})
        if not (self.resume and previous.get("status") in {"FAILED", "RUNNING"}): return None
        modes: list[str] = []
        for section in self._training_sections(stage_id):
            root = self._experiment_root(section); manifest, checkpoint = root / "manifest.json", root / "checkpoints" / "last.ckpt"
            valid = False
            if manifest.exists() and checkpoint.exists():
                try:
                    metadata = json.loads(manifest.read_text()); loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    valid = bool(metadata.get("experiment_name") == section["experiment_name"] and loaded.get("model") and loaded.get("optimizer"))
                except Exception:
                    valid = False
            if valid:
                modes.append("resume")
            elif root.exists():
                quarantine = root.with_name(root.name + f".failed-{now().replace(':', '').replace('+', '').replace('-', '')}")
                shutil.move(str(root), str(quarantine)); self.context.warnings.append(f"quarantined stale experiment: {quarantine}")
                modes.append("overwrite")
            else:
                modes.append(None)
        # A grouped controls stage must be internally consistent: a fresh
        # invocation is safest if one control needs replacement.
        return "resume" if modes and all(mode == "resume" for mode in modes) else ("overwrite" if "overwrite" in modes else None)

    def _execute_stage(self, stage_id: str) -> None:
        c = self.context
        if stage_id == "P00_preflight": self._preflight(); return
        if stage_id == "P01_dataset_validation": self._dataset_validation(); return
        if stage_id == "P02_prepare_range_images":
            for sequence in sum((list(values) for values in c.config["dataset"]["sequences"].values()), []):
                self._debug("P02", f"sequence={sequence} selected_frames={len(self._frame_names(sequence))} output={c.processed_root / sequence}")
                if self._run_command(stage_id + "_" + sequence, commands.prepare(c, sequence)): raise RuntimeError("range preprocessing failed")
            return
        if stage_id == "P03_precompute_depth":
            if not c.config["depth"].get("enabled"): return
            for sequence in sum((list(values) for values in c.config["dataset"]["sequences"].values()), []):
                self._debug("P03", f"sequence={sequence} selected_frames={len(self._frame_names(sequence))} model={c.config['depth']['model']} device={c.config['depth']['device']}")
                if self._run_command(stage_id + "_" + sequence, commands.depth(c, sequence, force=stage_id in self.force_stages)): raise RuntimeError("depth preprocessing failed")
            return
        if stage_id == "P04_create_splits": self._create_splits(); return
        if stage_id == "P05_train_student":
            self._debug("P05", f"experiment={c.config['student']['experiment_name']} train_split={c.train_split} validation_split={c.validation_split}")
            if self._run_command(stage_id, commands.student(c, self._recover_training_mode(stage_id))): raise RuntimeError("student training failed")
            c.artifacts["student_best_checkpoint"] = self._experiment_checkpoint(c.config["student"]); return
        if stage_id == "P06_train_teacher_correct":
            self._debug("P06", f"baseline={c.artifacts.get('student_best_checkpoint')} depth_mode=correct")
            if self._run_command(stage_id, commands.teacher(c, "correct", self._recover_training_mode(stage_id))): raise RuntimeError("teacher correct training failed")
            c.artifacts["teacher_checkpoints"]["correct"] = self._experiment_checkpoint(c.config["teachers"]["correct"]); return
        if stage_id == "P07_train_teacher_controls":
            for name, settings in c.config["teachers"].items():
                if name != "correct" and settings.get("enabled"):
                    self._debug("P07", f"control={name} baseline={c.artifacts.get('student_best_checkpoint')}")
                    if self._run_command(stage_id + "_" + name, commands.teacher(c, name, self._recover_training_mode(stage_id))): raise RuntimeError(f"teacher {name} training failed")
                    c.artifacts["teacher_checkpoints"][name] = self._experiment_checkpoint(settings)
            return
        if stage_id == "P08_evaluate_teachers":
            self._debug("P08", f"teacher_checkpoints={c.artifacts.get('teacher_checkpoints', {})}")
            if self._run_command(stage_id, commands.teacher_evaluation(c)): raise RuntimeError("teacher evaluation failed")
            return
        if stage_id == "P09_train_distillation":
            self._debug("P09", f"baseline={c.artifacts.get('student_best_checkpoint')} teacher={c.artifacts.get('teacher_checkpoints', {}).get(c.config['distillation']['teacher'])}")
            if self._run_command(stage_id, commands.distillation(c, self._recover_training_mode(stage_id))): raise RuntimeError("distillation training failed")
            c.artifacts["distillation_best_checkpoint"] = self._experiment_checkpoint(c.config["distillation"]); return
        if stage_id == "P10_evaluate_sr":
            targets = []
            if c.config["evaluation"].get("evaluate_student", True): targets.append((c.artifacts["student_best_checkpoint"], "student_baseline"))
            if c.config["evaluation"].get("evaluate_distilled", True): targets.append((c.artifacts["distillation_best_checkpoint"], c.config["distillation"]["experiment_name"]))
            for checkpoint, name in targets:
                self._debug("P10", f"evaluation={name} checkpoint={checkpoint} split={c.test_split}")
                if self._run_command(stage_id + "_" + name, commands.sr_evaluation(c, checkpoint, name)): raise RuntimeError("SR evaluation failed")
            return
        if stage_id == "P11_inference":
            choice = c.config["inference"].get("checkpoint", "distillation")
            checkpoint = c.artifacts["student_best_checkpoint"] if choice == "student" else c.artifacts["distillation_best_checkpoint"]
            frames = [c.processed_root / entry for entry in c.test_entries[:int(c.config["inference"].get("max_frames", 1))]]
            print(f"[P11_inference] selected frames: {len(frames)}", flush=True)
            for index, frame in enumerate(frames, start=1):
                output = c.root / "inference" / choice / frame.parent.name / f"{frame.name}.npz"
                print(f"[P11_inference] frame {index}/{len(frames)}: {frame.parent.name}/{frame.name}", flush=True)
                if self._run_command(stage_id + "_" + frame.name, commands.inference(c, checkpoint, frame, output)): raise RuntimeError("inference failed")
            return
        if stage_id == "P12_summary": self._debug("P12", f"summary_root={c.summary_root}"); build_summary(c, self.state.value); return

    def _hydrate_artifacts(self) -> None:
        c = self.context
        for path in (c.train_split, c.validation_split, c.test_split):
            if path.exists(): setattr(c, {c.train_split: "train_entries", c.validation_split: "validation_entries", c.test_split: "test_entries"}[path], path.read_text().splitlines())
        student = self._experiment_checkpoint(c.config["student"])
        if student.exists(): c.artifacts["student_best_checkpoint"] = student
        for name, settings in c.config["teachers"].items():
            path = self._experiment_checkpoint(settings)
            if path.exists(): c.artifacts["teacher_checkpoints"][name] = path
        distilled = self._experiment_checkpoint(c.config["distillation"])
        if distilled.exists(): c.artifacts["distillation_best_checkpoint"] = distilled

    def _planned_artifacts(self) -> None:
        c = self.context
        c.artifacts["student_best_checkpoint"] = self._experiment_checkpoint(c.config["student"])
        c.artifacts["teacher_checkpoints"] = {name: self._experiment_checkpoint(value) for name, value in c.config["teachers"].items() if value.get("enabled")}
        c.artifacts["distillation_best_checkpoint"] = self._experiment_checkpoint(c.config["distillation"])

    def _dry_commands(self, stage: str) -> list[list[str]]:
        c = self.context
        if stage in self.disabled: return []
        if stage == "P02_prepare_range_images": return [commands.prepare(c, sequence) for sequence in self._all_sequences()]
        if stage == "P03_precompute_depth": return [commands.depth(c, sequence) for sequence in self._all_sequences()]
        if stage == "P05_train_student": return [commands.student(c)]
        if stage == "P06_train_teacher_correct": return [commands.teacher(c, "correct")]
        if stage == "P07_train_teacher_controls": return [commands.teacher(c, name) for name, value in c.config["teachers"].items() if name != "correct" and value.get("enabled")]
        if stage == "P08_evaluate_teachers": return [commands.teacher_evaluation(c)]
        if stage == "P09_train_distillation": return [commands.distillation(c)]
        if stage == "P10_evaluate_sr":
            values = []
            if c.config["evaluation"].get("evaluate_student", True): values.append(commands.sr_evaluation(c, c.artifacts["student_best_checkpoint"], "student_baseline"))
            if c.config["evaluation"].get("evaluate_distilled", True): values.append(commands.sr_evaluation(c, c.artifacts["distillation_best_checkpoint"], c.config["distillation"]["experiment_name"]))
            return values
        if stage == "P11_inference":
            choice = c.config["inference"].get("checkpoint", "distillation")
            checkpoint = c.artifacts["student_best_checkpoint"] if choice == "student" else c.artifacts["distillation_best_checkpoint"]
            entries = c.test_entries or [f"{c.config['dataset']['sequences'].get('test', ['<sequence>'])[0]}/<selected-frame>"]
            return [commands.inference(c, checkpoint, c.processed_root / entry, c.root / "inference" / choice / Path(entry).parent.name / f"{Path(entry).name}.npz") for entry in entries[:int(c.config["inference"].get("max_frames", 1))]]
        return []

    def _print_dry_run(self, selected: list[str]) -> None:
        self._planned_artifacts(); c = self.context
        print(f"pipeline: {c.config['pipeline']['name']}")
        print(f"sequences: {self._all_sequences()}  max_frames_per_sequence: {c.config['dataset'].get('max_frames_per_sequence')}")
        print(f"depth model: {c.config['depth'].get('model')}")
        for stage in selected:
            status = "DISABLED" if stage in self.disabled else ("WOULD_RERUN" if stage in self.invalidated else "WOULD_RUN")
            print(f"{stage}\n  status: {status}\n  dependencies: {', '.join(DEPENDENCIES[stage]) or '-'}")
            if stage == "P06_train_teacher_correct": print(f"  baseline checkpoint: {c.artifacts['student_best_checkpoint']}")
            if stage == "P09_train_distillation": print(f"  teacher checkpoint: {c.artifacts['teacher_checkpoints'].get(c.config['distillation']['teacher'])}")
            try:
                expected = self._expected(stage)
            except Exception:
                expected = []
            if expected: print("  expected outputs: " + ", ".join(map(str, expected[:3])) + (" ..." if len(expected) > 3 else ""))
            if stage == "P11_inference" and not expected:
                print(f"  expected outputs: {c.root / 'inference' / c.config['inference'].get('checkpoint', 'distillation') / '<sequence>' / '<selected-frame>.npz'}")
            for command in self._dry_commands(stage): print("  command: " + " ".join(command))

    def run(self, *, from_stage: str | None = None, to_stage: str | None = None, only_stage: str | None = None) -> int:
        selected = list(STAGES)
        for name in (from_stage, to_stage, only_stage):
            if name and name not in STAGES: raise ValueError(f"unknown stage name: {name}")
        if only_stage: selected = [only_stage]
        else:
            if from_stage: selected = selected[selected.index(from_stage):]
            if to_stage: selected = selected[:selected.index(to_stage)+1]
        if self.dry_run:
            self._print_dry_run(selected)
            return 0
        self.context.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.context.config_path, self.context.root / f"pipeline_config{self.context.config_path.suffix}")
        write_json_atomic(self.context.root/"pipeline_manifest.json", {"schema_version": 1, "config_sha256": sha256(self.context.config_path), "created_at_utc": now(), "hash_policy": "full SHA-256 for config/splits/checkpoints"})
        self._hydrate_artifacts()
        failed = False
        total_stages = len(selected)
        for index, stage in enumerate(selected, start=1):
            prefix = f"[{index}/{total_stages}] {stage}"
            if stage in self.disabled:
                print(f"{prefix}: SKIPPED (disabled by config)", flush=True)
                self.state.update(stage, "SKIPPED", reason="disabled by config"); continue
            if failed or any(self.state.value["stages"].get(dep, {}).get("status") in {"FAILED", "BLOCKED"} for dep in DEPENDENCIES[stage] if dep in selected):
                print(f"{prefix}: BLOCKED (upstream failed)", flush=True)
                self.state.update(stage, "BLOCKED", reason="upstream failed"); continue
            fingerprint = self._fingerprint(stage)
            previous = self.state.value["stages"].get(stage, {})
            if self.resume and stage not in self.invalidated and previous.get("status") in {"SUCCEEDED", "SKIPPED"} and previous.get("fingerprint") == fingerprint:
                try:
                    self._validate_outputs(stage); print(f"{prefix}: SKIPPED (validated resume)", flush=True); self.state.update(stage, "SKIPPED", reason="validated resume", fingerprint=fingerprint); continue
                except Exception as error:
                    self.invalidated |= downstream(stage)
                    self.context.warnings.append(f"{stage} output validation failed; rerunning: {error}")
            print(f"{prefix}: STARTED", flush=True)
            self.state.update(stage, "RUNNING", fingerprint=fingerprint, command=[]); self._record_event({"stage": stage, "status": "RUNNING", "time": now()})
            try:
                self._execute_stage(stage); self._validate_outputs(stage); self.state.update(stage, "SUCCEEDED", fingerprint=fingerprint, outputs=[str(path) for path in self._expected(stage)], return_code=0); self._record_event({"stage": stage, "status": "SUCCEEDED", "time": now()}); print(f"{prefix}: SUCCEEDED", flush=True)
            except Exception as error:
                message = f"{stage}: FAILED: {error}"
                print(message, file=sys.stderr, flush=True)
                logs = self.context.root / "logs"; logs.mkdir(parents=True, exist_ok=True)
                details = traceback.format_exc()
                print(details, file=sys.stderr, end="", flush=True)
                with (logs / f"{stage}.stderr.log").open("a") as handle: handle.write(message + "\n" + details)
                self.state.update(stage, "FAILED", error=str(error), return_code=1); self._record_event({"stage": stage, "status": "FAILED", "error": str(error), "time": now()}); failed = True
                if self.context.config["pipeline"].get("fail_fast", True):
                    for later_index, later in enumerate(selected[index:], start=index + 1):
                        print(f"[{later_index}/{total_stages}] {later}: BLOCKED (blocked by {stage})", flush=True)
                        self.state.update(later, "BLOCKED", reason=f"blocked by {stage}")
                    break
        self.state.finish("FAILED" if failed else "SUCCEEDED")
        build_summary(self.context, self.state.value)
        print(f"pipeline status: {'FAILED' if failed else 'SUCCEEDED'}", flush=True)
        return 1 if failed else 0
