"""Experiment isolation, manifest, and resumable-output primitives."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib, json, shutil, warnings
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path; checkpoints: Path; best_checkpoint: Path; last_checkpoint: Path
    config: Path; manifest: Path; metrics: Path; logs: Path; invocations: Path


def _name(name: str) -> str:
    if not name or not name.strip() or Path(name).is_absolute() or "/" in name or "\\" in name or ".." in name:
        raise ValueError("experiment_name must be a non-empty path-safe name")
    return name


def prepare_experiment(*, output_root: Path, experiment_name: str, seed: int, resume: bool, overwrite: bool) -> ExperimentPaths:
    if resume and overwrite: raise ValueError("--resume and --overwrite cannot be used together")
    root = Path(output_root) / _name(experiment_name) / f"seed_{seed}"
    existing = root.exists() and any(root.iterdir())
    if existing and not resume and not overwrite:
        raise FileExistsError(f"Experiment directory already exists: {root}. Use --resume or --overwrite explicitly.")
    if overwrite and root.exists():
        warnings.warn(f"Overwriting existing experiment directory: {root}", UserWarning, stacklevel=2); shutil.rmtree(root)
    if resume and not (root / "manifest.json").exists(): raise FileNotFoundError(f"Cannot resume without manifest.json: {root}")
    checkpoints=root/"checkpoints"; logs=root/"logs"
    checkpoints.mkdir(parents=True, exist_ok=True); logs.mkdir(exist_ok=True)
    return ExperimentPaths(root, checkpoints, checkpoints/"best.ckpt", checkpoints/"last.ckpt", root/"config.json", root/"manifest.json", root/"metrics.jsonl", logs, root/"invocations.jsonl")


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); tmp.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle: handle.write(json.dumps(payload, sort_keys=True)+"\n")


def source_checkpoint_metadata(**paths: Path) -> dict:
    return {name:{"path":str(path),"sha256":hash_file_sha256(path)} for name,path in paths.items()}


def hash_file_sha256(path: str | Path | None) -> str | None:
    if path is None: return None
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_metadata(path: str | Path | None) -> dict | None:
    return None if path is None else {"path": str(path), "sha256": hash_file_sha256(path)}


def build_manifest(*, experiment_name: str, experiment_type: str, seed: int, deterministic: bool, dataset_root: str, train_split: str | None, validation_split: str | None, train_frames: int, validation_frames: int, model_config: dict, geometry: dict, depth_mode: str, advantage_config: dict | None) -> dict:
    return {"schema_version":1,"experiment_name":experiment_name,"experiment_type":experiment_type,"created_at_utc":datetime.now(timezone.utc).isoformat(),"seed":seed,"deterministic":deterministic,"dataset":{"root":str(dataset_root),"training_frames":train_frames,"validation_frames":validation_frames},"splits":{"train":split_metadata(train_split),"validation":split_metadata(validation_split),"test":None},"model_config":model_config,"geometry":geometry,"depth_mode":depth_mode,"advantage_config":advantage_config}


def assert_resume_compatible(existing: dict, requested: dict) -> None:
    for key in ("experiment_name","experiment_type","seed","deterministic","model_config","geometry","depth_mode","advantage_config"):
        if existing.get(key) != requested.get(key): raise ValueError(f"Resume configuration mismatch: {key}\nexisting: {existing.get(key)}\nrequested: {requested.get(key)}")
    if existing.get("dataset",{}).get("root") != requested.get("dataset",{}).get("root"): raise ValueError("Resume configuration mismatch: dataset_root")
    if existing.get("splits") != requested.get("splits"): raise ValueError("Resume configuration mismatch: splits")
    for name, value in requested.get("source_checkpoints", {}).items():
        old_hash = existing.get("source_checkpoints", {}).get(name, {}).get("sha256")
        new_hash = value.get("sha256")
        if old_hash != new_hash:
            raise ValueError(
                f"Resume source checkpoint mismatch: {name}\n"
                f"existing SHA-256: {old_hash}\nrequested SHA-256: {new_hash}"
            )


def invocation_record(*, invocation_index: int, resume: bool, overwrite: bool, arguments: dict) -> dict:
    """Build the append-only record that is also embedded in checkpoints."""
    return {
        "invocation_index": int(invocation_index),
        "resume": bool(resume),
        "overwrite": bool(overwrite),
        "arguments": arguments,
    }
