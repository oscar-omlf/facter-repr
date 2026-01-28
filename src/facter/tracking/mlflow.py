"""Provide small MLflow helpers for experiment setup and artifact logging.

This module wraps common MLflow operations used by this repository:

- Configure the tracking URI and experiment.
- Start a run with basic provenance tags.
- Log parameters, metrics, and small artifacts with minimal key sanitization.

The functions here are intentionally lightweight: they delegate to MLflow and rely
on the caller to provide correctly-typed objects (e.g., DataFrames).
"""

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
import time


@dataclass(frozen=True)
class MLflowConfig:
    """Configure MLflow tracking and naming defaults.

    Attributes:
        tracking_uri (str): MLflow tracking URI. Defaults to a local SQLite DB.
        experiment_name (str): Experiment name to use/create.
        run_name (Optional[str]): Optional default run name.
    """

    tracking_uri: str = "sqlite:///./mlflow.db"
    experiment_name: str = "facter-repro"
    run_name: Optional[str] = None


def _sanitize_key(key: str) -> str:
    """Sanitize a tag/param/metric key to match MLflow's allowed characters.

    Removes characters not allowed by MLflow.
    Allowed: alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ),
    and slashes (/).

    Args:
        key (str): Candidate key to sanitize.

    Returns:
        str: Sanitized key with disallowed characters replaced by underscores.
    """
    # This regex replaces any character NOT in the allowed set with an underscore
    return re.sub(r'[^a-zA-Z0-9._\-\/ ]', '_', key)


def _get_git_commit() -> str:
    """Get the current git commit hash.

    Returns:
        str: The current commit hash, or "UNKNOWN" if git is unavailable.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def _get_git_dirty() -> str:
    """Get the working tree state from git.

    Returns:
        str: "dirty" if there are uncommitted changes, "clean" if not, or
        "UNKNOWN" if git is unavailable.
    """
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return "dirty" if out.strip() else "clean"
    except Exception:
        return "UNKNOWN"

def setup_mlflow(cfg: MLflowConfig) -> None:
    """Configure MLflow tracking URI and ensure the experiment exists.

    This function sets the tracking URI, resolves a local artifact location under
    the current working directory, and then sets (and potentially creates) the
    configured experiment.

    Args:
        cfg (MLflowConfig): MLflow tracking configuration.
    """
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
    """Start an MLflow run and log basic provenance tags.

    This context manager configures MLflow via `setup_mlflow`, starts a run, sets
    default tags (git revision/state and platform metadata), logs the current
    working directory as a parameter, and yields the run ID.

    Args:
        cfg (MLflowConfig): MLflow tracking configuration.
        tags (Optional[Dict[str, Any]]): Optional extra tags to merge into the
            default tag set.

    Yields:
        str: The MLflow run ID.
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
    """Log parameters to the active MLflow run with sanitized keys.

    Keys are sanitized to satisfy MLflow restrictions. Values that are not
    primitive MLflow-safe types are converted to strings.

    Args:
        params (Dict[str, Any]): Parameter dictionary.
    """
    # Sanitize keys AND ensure values are MLflow-safe
    safe_params = {
        _sanitize_key(k): (v if isinstance(v, (str, int, float, bool)) else str(v)) 
        for k, v in params.items()
    }
    mlflow.log_params(safe_params)

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the active MLflow run with sanitized keys.

    Args:
        metrics (Dict[str, float]): Metric dictionary.
        step (Optional[int]): Optional step value forwarded to MLflow.
    """
    # Sanitize keys before logging
    safe_metrics = {_sanitize_key(k): float(v) for k, v in metrics.items()}
    for k, v in safe_metrics.items():
        mlflow.log_metric(k, v, step=step)


def log_text(text: str, artifact_path: str) -> None:
    """Log a small text artifact to the active MLflow run.

    The `artifact_path` should include a filename, e.g.,
    "configs/run_config.yaml".

    Args:
        text (str): Text content to log.
        artifact_path (str): MLflow artifact file path (including filename).
    """
    mlflow.log_text(text, artifact_file=artifact_path)


def log_dataframe(df: Any, artifact_path: str, format: str = "parquet") -> None:
    """Log a DataFrame-like object as an MLflow artifact.

    The object is written to a temporary file and then logged as an artifact.
    The exact methods called depend on `format`:

    - ``parquet``: calls ``df.to_parquet``
    - ``csv``: calls ``df.to_csv``
    - ``json``: calls ``df.to_json`` (with ``orient='records'`` and ``lines=True``)

    Args:
        df (Any): DataFrame-like object.
        artifact_path (str): Artifact path including filename (e.g.,
            "data/calibration.parquet").
        format (str): File format: "parquet", "csv", or "json".

    Raises:
        ValueError: If `format` is not one of the supported values.
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
