"""
A/B Test Statistical Significance Job

Reads gold_ab_test_results from Iceberg, runs two-proportion z-test,
logs results to MLflow. Answers: "Is the CTR lift statistically significant?"
"""

import argparse
import json
import logging
import os
from datetime import date

import mlflow
import numpy as np
from scipy.stats import norm, proportions_ztest

from src.data_pipeline.spark.session import create_spark_session

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ABSignificanceJob")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ICEBERG_TABLE = "lakehouse.gold.gold_ab_test_results"


def compute_significance(
    control_clicks: int,
    control_total: int,
    exp_clicks: int,
    exp_total: int,
    alpha: float = 0.05,
) -> dict:
    """Run two-proportion z-test and return full result dict."""
    control_ctr = control_clicks / control_total if control_total > 0 else 0.0
    exp_ctr = exp_clicks / exp_total if exp_total > 0 else 0.0

    z_stat, p_value = proportions_ztest(
        count=[exp_clicks, control_clicks],
        nobs=[exp_total, control_total],
        alternative="two-sided",
    )

    # 95% confidence interval for the lift
    se = np.sqrt(
        (exp_ctr * (1 - exp_ctr) / exp_total)
        + (control_ctr * (1 - control_ctr) / control_total)
    )
    diff = exp_ctr - control_ctr
    z_critical = norm.ppf(1 - alpha / 2)
    ci_low = diff - z_critical * se
    ci_high = diff + z_critical * se

    lift_pct = (diff / control_ctr * 100) if control_ctr > 0 else 0.0

    return {
        "control_ctr": round(float(control_ctr), 6),
        "exp_ctr": round(float(exp_ctr), 6),
        "z_stat": round(float(z_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "lift_pct": round(float(lift_pct), 2),
        "confidence_interval_95": [round(float(ci_low), 6), round(float(ci_high), 6)],
        "control_n": int(control_total),
        "exp_n": int(exp_total),
    }


def run(spark, run_date: str, exp_group: str, control_group: str) -> dict:
    logger.info(f"Loading {ICEBERG_TABLE} for date={run_date}")

    df = spark.read.format("iceberg").table(ICEBERG_TABLE)

    # Filter by date if provided, otherwise aggregate all available data
    if run_date:
        from pyspark.sql.functions import col, to_date

        df = df.filter(to_date(col("date")) <= run_date)

    from pyspark.sql.functions import col
    from pyspark.sql.functions import sum as spark_sum

    agg = (
        df.groupBy("experiment_group")
        .agg(
            spark_sum("total_recommendations").alias("total"),
            spark_sum("clicks").alias("clicks"),
            spark_sum("purchases").alias("purchases"),
        )
        .collect()
    )

    groups = {row["experiment_group"]: row for row in agg}

    if control_group not in groups:
        raise ValueError(
            f"Control group '{control_group}' not found. Available: {list(groups.keys())}"
        )
    if exp_group not in groups:
        raise ValueError(
            f"Experiment group '{exp_group}' not found. Available: {list(groups.keys())}"
        )

    ctrl = groups[control_group]
    exp = groups[exp_group]

    result = compute_significance(
        control_clicks=int(ctrl["clicks"]),
        control_total=int(ctrl["total"]),
        exp_clicks=int(exp["clicks"]),
        exp_total=int(exp["total"]),
    )
    result["run_date"] = run_date
    result["exp_group"] = exp_group
    result["control_group"] = control_group

    return result


def log_to_mlflow(result: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ab_significance")

    with mlflow.start_run(run_name=f"ab_sig_{result['run_date']}"):
        mlflow.log_params(
            {
                "exp_group": result["exp_group"],
                "control_group": result["control_group"],
                "run_date": result["run_date"],
                "significant": str(result["significant"]),
            }
        )
        mlflow.log_metrics(
            {
                "ab_control_ctr": result["control_ctr"],
                "ab_exp_ctr": result["exp_ctr"],
                "ab_p_value": result["p_value"],
                "ab_z_stat": result["z_stat"],
                "ab_lift_pct": result["lift_pct"],
                "ab_ci_low": result["confidence_interval_95"][0],
                "ab_ci_high": result["confidence_interval_95"][1],
            }
        )
        mlflow.log_dict(result, "ab_result.json")

    logger.info(f"Logged to MLflow: experiment=ab_significance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B test statistical significance")
    parser.add_argument(
        "--date", default=str(date.today()), help="Cutoff date (YYYY-MM-DD)"
    )
    parser.add_argument("--experiment-group", default="experiment", dest="exp_group")
    parser.add_argument("--control-group", default="control", dest="control_group")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    spark = None
    try:
        spark = create_spark_session()
        result = run(spark, args.date, args.exp_group, args.control_group)

        print(json.dumps(result, indent=2))
        logger.info(
            f"p_value={result['p_value']:.4f} | significant={result['significant']} | "
            f"lift={result['lift_pct']:+.2f}% | "
            f"control_CTR={result['control_ctr']:.4f} | exp_CTR={result['exp_ctr']:.4f}"
        )

        if not args.no_mlflow:
            log_to_mlflow(result)

    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise
    finally:
        if spark:
            spark.stop()
