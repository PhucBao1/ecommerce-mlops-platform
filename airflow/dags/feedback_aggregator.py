"""
Airflow DAG: Aggregate user feedback from Kafka → Iceberg silver layer.

Schedule: daily at 2 AM (low-traffic window).

Flow:
  Kafka topic 'agent_feedback' (click/purchase/ignore events)
    → Consume via confluent_kafka Consumer
    → Aggregate per (customer_id, product_id): compute training signal weight
    → Append to Iceberg silver.user_feedback_events
    → Log n_events to MLflow
    → Trigger retraining DAG when n_events >= MIN_EVENTS_FOR_RETRAIN

Training signal weights (same scale as LTR labels in train_ltr.py):
  purchase → 3  |  click → 1  |  ignore → 0
"""

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "broker:29092")
_CATALOG_URI = os.getenv("ICEBERG_CATALOG_URI", "http://localhost:8080")
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
_MIN_EVENTS = int(os.getenv("MIN_EVENTS_FOR_RETRAIN", "500"))

_ACTION_WEIGHTS = {"purchase": 3, "click": 1, "ignore": 0}

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def _consume_feedback(**context) -> int:
    """Poll 'agent_feedback' Kafka topic and push events via XCom."""
    from confluent_kafka import Consumer, KafkaError

    conf = {
        "bootstrap.servers": _BOOTSTRAP,
        "group.id": "feedback-aggregator-dag",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    consumer = Consumer(conf)
    consumer.subscribe(["agent_feedback"])

    events: list[dict] = []
    empty_polls = 0
    while empty_polls < 3:
        msg = consumer.poll(timeout=5.0)
        if msg is None:
            empty_polls += 1
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                empty_polls += 1
            continue
        try:
            ev = json.loads(msg.value().decode())
            ev["weight"] = _ACTION_WEIGHTS.get(ev.get("action", "ignore"), 0)
            events.append(ev)
            empty_polls = 0
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    consumer.commit()
    consumer.close()

    context["ti"].xcom_push(key="events", value=events)
    context["ti"].xcom_push(key="n_events", value=len(events))
    print(f"[feedback_aggregator] consumed {len(events)} events")
    return len(events)


def _write_to_iceberg(**context) -> None:
    """Append feedback events to silver.user_feedback_events Iceberg table."""
    import pandas as pd
    from pyiceberg.catalog import load_catalog

    events = context["ti"].xcom_pull(key="events")
    if not events:
        print("[feedback_aggregator] no events to write, skipping")
        return

    df = pd.DataFrame(events)
    df["ds"] = datetime.utcnow().strftime("%Y-%m-%d")
    df = df[
        [
            "customer_id",
            "product_id",
            "action",
            "weight",
            "source",
            "session_id",
            "ts",
            "ds",
        ]
    ]

    catalog = load_catalog("lakehouse", uri=_CATALOG_URI)
    table = catalog.load_table("silver.user_feedback_events")
    table.append(df)
    print(f"[feedback_aggregator] wrote {len(df)} rows to silver.user_feedback_events")


def _log_to_mlflow(**context) -> None:
    """Log feedback count to MLflow for tracking."""
    import mlflow

    n_events = context["ti"].xcom_pull(key="n_events") or 0
    mlflow.set_tracking_uri(_MLFLOW_URI)
    with mlflow.start_run(run_name="feedback_aggregator"):
        mlflow.log_metric("n_feedback_events", n_events)
        mlflow.log_metric("feedback_eligible_for_retrain", int(n_events >= _MIN_EVENTS))
    print(f"[feedback_aggregator] logged {n_events} events to MLflow")


def _maybe_trigger_retrain(**context) -> None:
    """Trigger recsys_retrain DAG when accumulated feedback >= MIN_EVENTS_FOR_RETRAIN."""
    from airflow.api.client.local_client import Client

    n_events = context["ti"].xcom_pull(key="n_events") or 0
    if n_events >= _MIN_EVENTS:
        client = Client(None, None)
        client.trigger_dag(
            dag_id="recsys_retrain",
            run_id=f"feedback_trigger_{context['ds_nodash']}",
        )
        print(
            f"[feedback_aggregator] triggered recsys_retrain with {n_events} feedback events"
        )
    else:
        print(
            f"[feedback_aggregator] {n_events} events < {_MIN_EVENTS} threshold, skipping retrain"
        )


with DAG(
    dag_id="feedback_aggregator",
    description="Aggregate Kafka agent_feedback -> Iceberg silver + optional retrain trigger",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["rlhf", "feedback", "silver"],
) as dag:

    t1 = PythonOperator(task_id="consume_feedback", python_callable=_consume_feedback)
    t2 = PythonOperator(task_id="write_to_iceberg", python_callable=_write_to_iceberg)
    t3 = PythonOperator(task_id="log_to_mlflow", python_callable=_log_to_mlflow)
    t4 = PythonOperator(
        task_id="maybe_trigger_retrain", python_callable=_maybe_trigger_retrain
    )

    t1 >> t2 >> t3 >> t4
