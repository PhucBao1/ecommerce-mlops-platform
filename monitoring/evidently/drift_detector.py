import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


def detect_and_report(
    reference_path: str = "monitoring/evidently/reference/reference.parquet",
    current_path: str = "monitoring/evidently/reference/valid_df.parquet",
    output_path: str = "monitoring/evidently/reports/drift_report.html",
    drift_threshold: float = 0.3,
) -> float:
    """Run Evidently drift report and raise if drift exceeds threshold.

    Returns the drift share score (fraction of drifted features).
    Raises ValueError if drift_score > drift_threshold so Airflow marks the task as failed.
    """
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    report.run(
        reference_data=pd.read_parquet(reference_path),
        current_data=pd.read_parquet(current_path),
    )
    report.save_html(output_path)

    result = report.as_dict()
    drift_score: float = result["metrics"][0]["result"]["dataset_drift_share"]

    print(f"Drift score: {drift_score:.2%} (threshold: {drift_threshold:.2%})")

    if drift_score > drift_threshold:
        raise ValueError(
            f"Data drift detected: {drift_score:.2%} of features drifted "
            f"(threshold: {drift_threshold:.2%}). Trigger retrain."
        )

    return drift_score


if __name__ == "__main__":
    detect_and_report()
