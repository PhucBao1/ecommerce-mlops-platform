import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.task_group import TaskGroup
from docker.types import Mount

default_args = {
    "owner": "baobao",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# ECR_REGISTRY được set trong environment của container Airflow (docker-compose.batch_dev.yml)
# — nếu trống (local dev), fallback về tag local "ecommerce-local", khớp đúng
# pattern trong docker-compose.infra.yml/batch_dev.yml (${ECR_REGISTRY:-ecommerce-local}/...)
_ECR_REGISTRY = os.getenv("ECR_REGISTRY", "ecommerce-local")
SPARK_IMAGE = f"{_ECR_REGISTRY}/spark:latest"
DBT_IMAGE = f"{_ECR_REGISTRY}/dbt:latest"

aws_env = {
    "AWS_REGION": Variable.get("AWS_REGION", default_var="ap-southeast-1"),
    "AWS_DEFAULT_REGION": Variable.get("AWS_REGION", default_var="ap-southeast-1"),
}
spark_catalog_env = {
    "POSTGRES_HOST": os.getenv("POSTGRES_ICEBERG_HOST", "postgres-catalog"),
    "POSTGRES_PORT": os.getenv("POSTGRES_ICEBERG_PORT", "5432"),
    "POSTGRES_ICEBERG_USER": os.getenv("POSTGRES_ICEBERG_USER", "admin"),
    "POSTGRES_ICEBERG_PASSWORD": os.getenv("POSTGRES_ICEBERG_PASSWORD", "password"),
    "POSTGRES_ICEBERG_DB": os.getenv("POSTGRES_ICEBERG_DB", "iceberg_metadata"),
    "CATALOG_WAREHOUSE": Variable.get(
        "CATALOG_WAREHOUSE", default_var="s3a://warehouse/"
    ),
    "S3_ENDPOINT_URL": Variable.get("S3_ENDPOINT_URL", default_var="http://minio:9000"),
}

docker_common_args = {
    "api_version": "auto",
    "docker_url": "unix://var/run/docker.sock",
    "network_mode": "my_shared_network",  # Đồng nhất 1 mạng duy nhất
    "auto_remove": "success",
    "mount_tmp_dir": False,
}


with DAG(
    "daily_bronze_to_silver_job_2",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 5, 4),
    catchup=False,  # Không chạy bù các ngày trong quá khứ nếu bỏ lỡ
    tags=["etl", "spark", "ecommerce"],
) as dag:

    # ==========================================
    # TASK 1: CHẠY CRAWLER (Data Ingestion)
    # ==========================================
    """crawl_tiki_task = DockerOperator(
        task_id='crawl_tiki_data',
        image='ecommerce-crawler:latest', # Thay bằng tên image crawler của bạn
        command='python /opt/workspace/src/crawler/spiders/tiki_crawl.py --date {{ ds }}',

        environment={
            'AWS_ACCESS_KEY_ID': 'admin',
            'AWS_SECRET_ACCESS_KEY': 'password',
            'AWS_DEFAULT_REGION': 'us-east-1',
            'AWS_ENDPOINT_URL': 'http://minio:9000' # CHỈ CẦN DÒNG NÀY ĐỂ PYTHON BOTO3 TÌM THẤY MINIO
        },
        **docker_common_args
    )"""

    # ==========================================
    # STEP 1: BRONZE → SILVER
    # ==========================================

    # Khối lệnh này sẽ nói Airflow tự động sinh ra container và chạy Spark
    bronze_to_silver = DockerOperator(
        task_id="bronze_to_silver_task",
        image=SPARK_IMAGE,
        # entrypoint=["/bin/bash", "-c"],
        # Sức mạnh của Airflow nằm ở đây: {{ ds }} sẽ tự động thay bằng ngày chạy (YYYY-MM-DD)
        command="spark-submit /opt/workspace/src/data_pipeline/jobs/bronze_to_silver.py --date {{ ds }}",
        environment={
            **aws_env,
            **spark_catalog_env,
        },
        **docker_common_args,
    )

    # ==========================================
    # STEP 2: ML INFERENCE
    # ==========================================
    inference_task = DockerOperator(
        task_id="spark_ml_inference",
        image=SPARK_IMAGE,
        command="spark-submit /opt/workspace/src/data_pipeline/jobs/inference_job.py --date {{ ds }}",
        environment={
            **aws_env,
            **spark_catalog_env,
        },
        **docker_common_args,
    )

    # ==========================================
    # STEP 2.5: ICEBERG TABLE MAINTENANCE
    # ==========================================
    iceberg_maintenance = DockerOperator(
        task_id="iceberg_maintenance",
        image=SPARK_IMAGE,
        command=(
            "spark-submit /opt/workspace/src/data_pipeline/jobs/iceberg_maintenance.py "
            "--date {{ ds }}"
        ),
        environment={
            **aws_env,
            **spark_catalog_env,
        },
        **docker_common_args,
    )

    # ==========================================
    # PHASE 2: DATA TRANSFORMATION (dbt)
    # ==========================================
    # Lưu ý: Cần cd vào đúng thư mục dbt_project trước khi chạy lệnh dbt
    dbt_project_dir = "/opt/workspace/dbt_project"

    with TaskGroup("dbt", tooltip="All dbt tasks") as dbt_group:

        dbt_deps = DockerOperator(
            task_id="dbt_deps",
            image=DBT_IMAGE,
            command="dbt deps --target prod_airflow",
            working_dir=dbt_project_dir,  # Trỏ tới thư mục chứa dbt_project.yml trong container
            **docker_common_args,
        )

        dbt_run_staging = DockerOperator(
            task_id="dbt_run_staging",
            image=DBT_IMAGE,
            command="dbt run --select tag:staging --target prod_airflow",
            working_dir=dbt_project_dir,
            **docker_common_args,
        )

        dbt_snapshot = DockerOperator(
            task_id="dbt_snapshot",
            image=DBT_IMAGE,
            command="dbt snapshot --target prod_airflow",
            working_dir=dbt_project_dir,
            **docker_common_args,
        )

        dbt_run_mart_staging = DockerOperator(
            task_id="dbt_run_mart_staging",
            image=DBT_IMAGE,
            command="dbt run --select tag:mart_stg --exclude gold_ab_test_results --target prod_airflow",
            working_dir=dbt_project_dir,
            **docker_common_args,
        )

        dbt_run_mart_snapshot = DockerOperator(
            task_id="dbt_run_mart_snapshot",
            image=DBT_IMAGE,
            command="dbt run --select tag:mart_snp --target prod_airflow",
            working_dir=dbt_project_dir,
            **docker_common_args,
        )

        dbt_test = DockerOperator(
            task_id="dbt_test",
            image=DBT_IMAGE,
            command="dbt test --exclude gold_ab_test_results source:lakehouse_silver.prediction_events --target prod_airflow",
            working_dir=dbt_project_dir,
            **docker_common_args,
        )

        dbt_deps >> dbt_run_staging

        dbt_run_staging >> dbt_snapshot

        dbt_run_staging >> dbt_run_mart_staging

        dbt_snapshot >> dbt_run_mart_snapshot

        [dbt_run_mart_snapshot, dbt_run_mart_staging] >> dbt_test

    # ==========================================
    # STEP 3: DATA DRIFT DETECTION (Evidently)
    # ==========================================
    check_drift = DockerOperator(
        task_id="check_drift",
        image="ecommerce-monitoring:latest",
        command="python /opt/workspace/monitoring/evidently/drift_detector.py",
        **docker_common_args,
    )

    # ==========================================
    # STEP 4: AUTO-TRIGGER RETRAIN ON DRIFT
    # trigger_rule="all_failed": only fires when check_drift task fails
    # (drift_detector.py exits non-zero when drift > threshold)
    # ==========================================
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain_on_drift",
        trigger_dag_id="recsys_retraining_pipeline",
        conf={"reason": "data_drift_detected"},
        trigger_rule="all_failed",
    )

    # ==========================================
    # ĐỊNH NGHĨA LUỒNG CHẠY (DEPENDENCIES)
    # ==========================================
    # iceberg_maintenance chạy SAU bronze_to_silver (bảng silver vừa được ghi xong)
    # nhưng SONG SONG với inference/dbt — nó chỉ gộp file và dọn snapshot cũ, không
    # đổi nội dung dữ liệu, nên không cần chặn các bước phía sau chờ nó.
    bronze_to_silver >> inference_task >> dbt_group >> check_drift >> trigger_retrain
    bronze_to_silver >> iceberg_maintenance
