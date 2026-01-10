import os
from pathlib import Path

import mlflow

from facter.tracking.mlflow import MLflowConfig, start_run, log_metrics, log_params


def test_mlflow_run_creates_artifacts(tmp_path: Path):
    # Use an isolated file store
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    cfg = MLflowConfig(tracking_uri=tracking_uri, experiment_name="pytest-exp", run_name="test-run")

    with start_run(cfg, tags={"purpose": "unit-test"}) as run_id:
        assert isinstance(run_id, str) and len(run_id) > 0
        log_params({"alpha": 0.1, "lambda": 0.7})
        log_metrics({"loss": 1.23}, step=0)

    # Verify run is present
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("pytest-exp")
    assert exp is not None

    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.data.tags.get("purpose") == "unit-test"
    assert run.data.params.get("alpha") == "0.1"
    assert "loss" in run.data.metrics
