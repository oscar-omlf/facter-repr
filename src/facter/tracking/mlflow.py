import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import mlflow


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str = "sqlite:///./mlflow.db"
    experiment_name: str = "facter-repro"
    run_name: Optional[str] = None


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
    mlflow.set_experiment(cfg.experiment_name)


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
    # MLflow requires values to be simple (str/int/float/bool)
    safe = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items()}
    mlflow.log_params(safe)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v), step=step)


def log_text(text: str, artifact_path: str) -> None:
    """
    Save a small text artifact (e.g., config dump).
    artifact_path should include a filename, e.g., 'configs/run_config.yaml'
    """
    mlflow.log_text(text, artifact_file=artifact_path)
