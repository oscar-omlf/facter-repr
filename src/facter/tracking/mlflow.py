import os
import re
import platform
import subprocess
import tempfile
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import mlflow


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str = "sqlite:///./mlflow.db"
    experiment_name: str = "facter-repro"
    run_name: Optional[str] = None


def _sanitize_key(key: str) -> str:
    """
    Removes characters not allowed by MLflow:
    Allowed: alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/)
    """
    # This regex replaces any character NOT in the allowed set with an underscore
    return re.sub(r'[^a-zA-Z0-9._\-\/ ]', '_', key)


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def _get_git_dirty() -> str:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return "dirty" if out.strip() else "clean"
    except Exception:
        return "UNKNOWN"

def setup_mlflow(cfg: MLflowConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    
    # Use current working directory to ensure it stays within your GPFS allocation
    base_dir = Path.cwd().resolve()
    artifact_path = (base_dir / "mlruns").as_uri() # Converts to 'file:///path/to/mlruns'
    
    experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
    if experiment is None:
        # Create new experiment with explicit local artifact location
        mlflow.create_experiment(cfg.experiment_name, artifact_location=artifact_path)
    elif experiment.artifact_location.startswith("/home/ozzy"):
        # If the experiment exists but points to the wrong user, you must rename or delete it
        print(f"WARNING: Experiment {cfg.experiment_name} has a corrupted path. Creating a new one.")
        mlflow.create_experiment(f"{cfg.experiment_name}_{int(time.time())}", artifact_location=artifact_path)
    
    mlflow.set_experiment(cfg.experiment_name)

# def setup_mlflow(cfg: MLflowConfig) -> None:
#     mlflow.set_tracking_uri(cfg.tracking_uri)
#     mlflow.set_experiment(cfg.experiment_name)


@contextmanager
def start_run(cfg: MLflowConfig, tags: Optional[Dict[str, Any]] = None) -> Iterator[str]:
    """
    Context manager that starts an MLflow run and logs basic provenance.
    Returns the run_id.
    """
    setup_mlflow(cfg)

    base_tags: Dict[str, Any] = {
        "git.commit": _get_git_commit(),
        "git.state": _get_git_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.set_tags(base_tags)
        # Useful default: record working directory (helps debugging)
        mlflow.log_param("cwd", os.getcwd())
        yield run.info.run_id


def log_params(params: Dict[str, Any]) -> None:
    # Sanitize keys AND ensure values are MLflow-safe
    safe_params = {
        _sanitize_key(k): (v if isinstance(v, (str, int, float, bool)) else str(v)) 
        for k, v in params.items()
    }
    mlflow.log_params(safe_params)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    # Sanitize keys before logging
    safe_metrics = {_sanitize_key(k): float(v) for k, v in metrics.items()}
    for k, v in safe_metrics.items():
        mlflow.log_metric(k, v, step=step)


def log_text(text: str, artifact_path: str) -> None:
    """
    Save a small text artifact (e.g., config dump).
    artifact_path should include a filename, e.g., 'configs/run_config.yaml'
    """
    mlflow.log_text(text, artifact_file=artifact_path)


def log_dataframe(df: Any, artifact_path: str, format: str = "parquet") -> None:
    """
    Log a pandas DataFrame as an artifact.
    
    Args:
        df: pandas DataFrame to log
        artifact_path: Path where the artifact will be saved (e.g., 'data/calibration.parquet')
        format: File format - 'parquet', 'csv', or 'json' (default: 'parquet')
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / Path(artifact_path).name
        
        if format == "parquet":
            df.to_parquet(tmp_path, index=False)
        elif format == "csv":
            df.to_csv(tmp_path, index=False)
        elif format == "json":
            df.to_json(tmp_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}. Choose from 'parquet', 'csv', or 'json'.")
        
        mlflow.log_artifact(str(tmp_path), artifact_path=str(Path(artifact_path).parent) if Path(artifact_path).parent != Path(".") else None)
