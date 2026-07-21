import argparse
import logging
import os
import time

import pandas as pd
import pyspark.sql.functions as F
import torch
from pyspark.sql import SparkSession
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_pipeline.spark.session import create_spark_session

# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("InferenceJob")

# ==========================================
# 2. GLOBAL CONFIGURATION
# ==========================================
BATCH_SIZE = 32
MODEL_PATH = "artifacts/nlp_models/phobert/version_001"
MODEL = None
TOKENIZER = None
DEVICE = None
ONNX_SESSION = None


# ==========================================
# 3. MODEL LOADING FUNCTION
# ==========================================
def load_model():

    global MODEL, TOKENIZER, DEVICE, ONNX_SESSION

    if TOKENIZER is None:

        logger.info("Loading PhoBERT model...")

        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Dùng ONNX Runtime nếu có sẵn model.onnx — nhanh hơn PyTorch đáng kể trên
        # CPU (cùng model, cùng pattern đã dùng trong sentiment-api/phobert_api.py),
        # quan trọng vì job này chạy trên EC2 chỉ 2 vCPU, không có GPU (Phase 1).
        onnx_path = os.path.join(MODEL_PATH, "model.onnx")
        if os.path.exists(onnx_path):
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if DEVICE == "cuda"
                else ["CPUExecutionProvider"]
            )
            ONNX_SESSION = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(
                "✅ ONNX Runtime enabled (%s, provider=%s)",
                onnx_path,
                ONNX_SESSION.get_providers()[0],
            )
        else:
            logger.info("ONNX model không tồn tại tại %s — dùng PyTorch", onnx_path)
            MODEL = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, local_files_only=True
            )
            MODEL.to(DEVICE)
            MODEL.eval()

    return TOKENIZER, MODEL, DEVICE


def _predict_batch(tokenizer, model, device, texts):
    inputs = tokenizer(
        texts, truncation=True, padding=True, max_length=64, return_tensors="pt"
    )

    if ONNX_SESSION is not None:
        ort_out = ONNX_SESSION.run(
            None,
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
            },
        )[0]
        logits = torch.from_numpy(ort_out)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits

    return torch.argmax(logits, dim=1)


def predict_partition(rows):

    logger.info("Loading model in partition...")

    tokenizer, model, device = load_model()

    logger.info("Model loaded successfully")

    if model is not None:
        model.to(device)
        model.eval()

    label_map = {0: "NEG", 1: "POS", 2: "NEU"}

    batch = []
    processed = 0

    last_log = time.time()  # thời điểm log cuối
    LOG_INTERVAL = 30  # log mỗi 30 giây

    for row in rows:

        text = row.clean_comment

        # Empty comment -> default to neutral
        if not text or text.strip() == "":
            yield (row.review_id, "NEU", "phobert_v1")
            continue

        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            texts = [r.clean_comment if r.clean_comment else "" for r in batch]

            preds = _predict_batch(tokenizer, model, device, texts)

            for r, pred in zip(batch, preds):

                yield (r.review_id, label_map[pred.item()], "phobert_v1")
            processed += len(batch)

            now = time.time()
            if now - last_log >= LOG_INTERVAL:
                logger.info(f"Processed {processed} reviews in partition")
                last_log = now

            batch = []

    # Process remaining batch
    if batch:

        texts = [r.clean_comment if r.clean_comment else "" for r in batch]

        preds = _predict_batch(tokenizer, model, device, texts)

        for r, pred in zip(batch, preds):

            yield (r.review_id, label_map[pred.item()], "phobert_v1")


# ==========================================
# 4. MAIN PROCESSING FUNCTION
# ==========================================
def process_inference(spark):
    logger.info("Đọc dữ liệu từ bảng cleaned_comments...")

    df = spark.table("lakehouse.silver.cleaned_comment")

    # Only infer for reviews that do not already have sentiment
    if spark.catalog.tableExists("lakehouse.gold.review_sentiments"):

        existing = spark.table("lakehouse.gold.review_sentiments").select("review_id")

        df = df.join(existing, on="review_id", how="left_anti")

    # Select required columns only
    df = df.select("review_id", "clean_comment")

    # Reduce memory pressure
    df = df.repartition(2)

    logger.info(f"Total rows for inference: {df.count()}")

    # Spark RDD inference
    result_rdd = df.rdd.mapPartitions(predict_partition)

    df_result = spark.createDataFrame(
        result_rdd, schema=["review_id", "sentiment", "model_version"]
    )

    # Metadata
    df_result = df_result.withColumn("inference_time", F.current_timestamp())

    # Remove duplicate
    df_result = df_result.dropDuplicates(["review_id"])

    logger.info("Ghi kết quả dự đoán (Sentiment) vào lại Iceberg...")

    # Enable schema evolution (allow new columns in Iceberg table)
    spark.conf.set("spark.sql.iceberg.schema-evolution.enabled", "true")

    table_name = "lakehouse.gold.review_sentiments"

    if not spark.catalog.tableExists(table_name):

        df_result.writeTo(table_name).tableProperty("format-version", "2").create()

    else:

        logger.info("MERGE incremental sentiment results...")

        df_result.createOrReplaceTempView("new_sentiment")

        spark.sql(
            f"""
            MERGE INTO {table_name} t
            USING new_sentiment s
            ON t.review_id = s.review_id

            WHEN MATCHED THEN UPDATE SET
                t.sentiment = s.sentiment,
                t.model_version = s.model_version,
                t.inference_time = s.inference_time

            WHEN NOT MATCHED THEN INSERT *
        """
        )

    logger.info("✅ Đã hoàn thành quá trình ML Inference!")


# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=False, help="Execution date")
    args = parser.parse_args()

    spark = None
    try:
        spark = create_spark_session()
        process_inference(spark)
    except Exception as e:
        logger.error(f"❌ Job thất bại: {e}")
        raise e
    finally:
        if spark:
            spark.stop()
