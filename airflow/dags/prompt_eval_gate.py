"""
Airflow DAG: Automated prompt evaluation gate with rollback.

Triggered after deploying a new prompt version (manual trigger or CI/CD).
Runs DeepEval, compares to baseline, rolls back if quality drops.

Flow:
  1. run_deepeval_current  → score new prompt version
  2. run_deepeval_previous → score previous version (baseline)
  3. compare_and_gate      → if new_score < baseline * ROLLBACK_THRESHOLD → rollback
  4. notify                → log result to MLflow

ROLLBACK_THRESHOLD default 0.97 = allow max 3% regression.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_ROLLBACK_THRESHOLD = float(os.getenv("PROMPT_ROLLBACK_THRESHOLD", "0.97"))

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 0,
    "email_on_failure": False,
}


def _eval_version(version: str, **context) -> float:
    """Run DeepEval for a prompt version and push score via XCom."""
    from src.serving.agent_api.eval.deepeval_eval import run_eval

    scores = run_eval(prompt_version=version)
    composite = (
        scores.get("answer_relevancy", 0) * 0.4
        + scores.get("faithfulness", 0) * 0.4
        + scores.get("contextual_recall", 0) * 0.2
    )
    key = f"score_{version.replace('.', '_')}"
    context["ti"].xcom_push(key=key, value=round(composite, 4))
    print(f"[prompt_eval_gate] {version} composite={composite:.4f}")
    return composite


def _eval_current(**context) -> None:
    from src.serving.agent_api.prompt_registry import PromptRegistry

    version = PromptRegistry().get_active()
    context["ti"].xcom_push(key="current_version", value=version)
    _eval_version(version, **context)


def _eval_previous(**context) -> None:
    from src.serving.agent_api.prompt_registry import PromptRegistry

    registry = PromptRegistry()
    history = registry.history()
    previous = history[0] if history else registry.get_active()
    context["ti"].xcom_push(key="previous_version", value=previous)
    _eval_version(previous, **context)


def _compare_and_gate(**context) -> None:
    """Roll back if new version scores below threshold * baseline."""
    ti = context["ti"]
    current_v = ti.xcom_pull(key="current_version") or "v1"
    previous_v = ti.xcom_pull(key="previous_version") or "v1"

    current_score = ti.xcom_pull(key=f"score_{current_v.replace('.','_')}") or 0.0
    previous_score = ti.xcom_pull(key=f"score_{previous_v.replace('.','_')}") or 0.0

    print(f"[prompt_eval_gate] current={current_v} score={current_score:.4f}")
    print(f"[prompt_eval_gate] baseline={previous_v} score={previous_score:.4f}")
    print(f"[prompt_eval_gate] threshold={_ROLLBACK_THRESHOLD}")

    import mlflow

    mlflow.set_tracking_uri(_MLFLOW_URI)
    with mlflow.start_run(run_name="prompt_eval_gate"):
        mlflow.log_param("current_version", current_v)
        mlflow.log_param("previous_version", previous_v)
        mlflow.log_metric("current_score", current_score)
        mlflow.log_metric("baseline_score", previous_score)
        mlflow.log_metric("threshold", _ROLLBACK_THRESHOLD)

        if previous_score > 0 and current_score < previous_score * _ROLLBACK_THRESHOLD:
            from src.serving.agent_api.prompt_registry import PromptRegistry

            rolled_back_to = PromptRegistry().rollback()
            mlflow.log_param("action", "rollback")
            mlflow.log_param("rolled_back_to", rolled_back_to)
            print(
                f"[prompt_eval_gate] ROLLBACK: {current_v} ({current_score:.4f}) "
                f"< {previous_v} ({previous_score:.4f}) * {_ROLLBACK_THRESHOLD} "
                f"→ rolled back to {rolled_back_to}"
            )
        else:
            mlflow.log_param("action", "promote")
            print(f"[prompt_eval_gate] PROMOTE: {current_v} passes quality gate")


with DAG(
    dag_id="prompt_eval_gate",
    description="Evaluate new prompt version vs baseline; auto-rollback on regression",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # triggered manually or via CI/CD
    catchup=False,
    tags=["prompt", "eval", "rollback", "llm"],
) as dag:

    t_curr = PythonOperator(
        task_id="eval_current_version", python_callable=_eval_current
    )
    t_prev = PythonOperator(
        task_id="eval_previous_version", python_callable=_eval_previous
    )
    t_gate = PythonOperator(
        task_id="compare_and_gate", python_callable=_compare_and_gate
    )

    [t_curr, t_prev] >> t_gate
