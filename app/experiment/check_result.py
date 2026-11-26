import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


# Metric history does not include the test configuration. Match entries to
# the order of N_NODES in script/experiment/test_model.sh.
N_NODES = (20, 50, 100, 500, 1000)
METRICS = (
    "test_acc",
    "test_ap",
    "test_pos_recall",
    "test_neg_recall",
    "test_balanced_acc",
)
RUN_PARAMS = ["params.model_name", "params.seed", "params.lr"]
TRACKING_DB = Path("../data/TR-NE/tracking/experiments.db")


def main(experiment_name: str) -> None:
    if not TRACKING_DB.is_file():
        raise SystemExit(f"MLflow tracking database not found: {TRACKING_DB.resolve()}")

    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if not set(RUN_PARAMS).issubset(runs.columns):
        raise SystemExit("No training runs with model_name, seed, and lr were found.")

    # test_model.py writes to the latest training run for each configuration.
    runs = runs.dropna(subset=RUN_PARAMS).drop_duplicates(subset=RUN_PARAMS)
    rows = []
    for run_id, model in runs[["run_id", "params.model_name"]].itertuples(index=False, name=None):
        histories = {
            metric: sorted(
                client.get_metric_history(run_id, metric),
                key=lambda point: point.timestamp,
            )
            for metric in METRICS
        }
        if all(not history for history in histories.values()):
            continue
        if any(
            len(history) != len(N_NODES)
            or len({point.timestamp for point in history}) != len(N_NODES)
            for history in histories.values()
        ):
            print(
                f"Skipping run {run_id}: expected {len(N_NODES)} distinctly timed "
                "entries per test metric.",
                file=sys.stderr,
            )
            continue

        for index, n_node in enumerate(N_NODES):
            rows.append({
                "model": model,
                "n_node": n_node,
                **{metric: histories[metric][index].value for metric in METRICS},
            })

    if not rows:
        raise SystemExit("No complete test metric histories were found.")

    summary = (
        pd.DataFrame(rows)
        .groupby(["model", "n_node"])[list(METRICS)]
        .agg(["mean", "std", "count"])
    )
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize test metrics by model and node count across seeds and learning rates."
    )
    parser.add_argument("--experiment_name", default="TR-NE")
    main(experiment_name=parser.parse_args().experiment_name)
