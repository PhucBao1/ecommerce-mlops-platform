from datetime import datetime

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


def promote_model(**context):
    """Transition the latest registered version to Production stage."""
    import mlflow

    client = mlflow.MlflowClient(tracking_uri="http://mlflow:5000")
    versions = client.get_latest_versions("recsys-two-tower", stages=["None"])
    if versions:
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.transition_model_version_stage(
            name="recsys-two-tower",
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Promoted version {latest.version} to Production")
    else:
        print("No registered versions found — skipping promotion")


def check_promotion(**context):
    import mlflow

    client = mlflow.MlflowClient(tracking_uri="http://mlflow:5000")

    # NDCG of current production model (0.0 if none deployed yet)
    prod_ndcg = 0.0
    prod_versions = client.get_latest_versions(
        "recsys-two-tower", stages=["Production"]
    )
    if prod_versions:
        prod_ndcg = client.get_run(prod_versions[0].run_id).data.metrics.get(
            "valid_ndcg_at_10", 0.0
        )

    # NDCG of the latest training run
    runs = mlflow.search_runs(
        experiment_names=["two_tower_recsys"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    new_ndcg = (
        float(runs.iloc[0]["metrics.valid_ndcg_at_10"]) if not runs.empty else 0.0
    )

    print(f"Production NDCG@10: {prod_ndcg:.4f} | New NDCG@10: {new_ndcg:.4f}")

    return "export_embeddings" if new_ndcg > prod_ndcg else "notify_degradation"


with DAG(
    dag_id="recsys_retraining_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:

    train_model = DockerOperator(
        task_id="train_model",
        image="recsys-training:latest",
        command="python src/ml_models/recsys/train_model.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
    )

    evaluate_model = DockerOperator(
        task_id="evaluate_model",
        image="recsys-training:latest",
        command="python src/ml_models/recsys/evaluation/evaluate.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
        mounts=[
            Mount(
                source="/home/bao/BaoBao/Ecommerce/artifacts",
                target="/app/artifacts",
                type="bind",
            )
        ],
    )

    promotion_gate = BranchPythonOperator(
        task_id="promotion_gate",
        python_callable=check_promotion,
    )

    notify_degradation = PythonOperator(
        task_id="notify_degradation",
        python_callable=lambda **ctx: print(
            "New model NDCG did not improve — skipping deployment."
        ),
        provide_context=True,
    )

    export_embeddings = DockerOperator(
        task_id="export_embeddings",
        image="recsys-training:latest",
        command="python src/ml_models/recsys/retrieval/export_embeddings.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
    )

    build_faiss = DockerOperator(
        task_id="build_faiss",
        image="recsys-training:latest",
        command="python src/ml_models/recsys/retrieval/build_faiss.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
    )

    promote_model_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    train_model >> evaluate_model >> promotion_gate
    promotion_gate >> [export_embeddings, notify_degradation]
    export_embeddings >> build_faiss >> promote_model_task
