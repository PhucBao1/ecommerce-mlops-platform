import logging
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# 1. SETUP LOGGING CHUẨN ENTERPRISE
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 2. TÁCH RIÊNG CLASS TRAINER RA NGOÀI (Tránh lỗi Multiprocessing)
class ImbalanceTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # Nhận trọng số từ bên ngoài

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Dùng model.config.num_labels an toàn hơn self.model
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ... (import các thư viện cần thiết khác)


class PhoBertSentimentTrainer:
    def __init__(self, config):
        """Khởi tạo các cấu hình từ bên ngoài truyền vào"""
        self.model_name = config.get(
            "model_name", "wonrax/phobert-base-vietnamese-sentiment"
        )
        self.data_path = config.get("data_path", "data/latest_comments.parquet")

        self.output_root = config.get("output_dir", "./models/phobert_retrained")
        self.batch_size = config.get("batch_size", 32)
        self.max_length = config.get("max_length", 64)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def load_and_prepare_data(self):
        """Bước 1: Đọc, chia split và tokenize data"""
        print(f"Loading data from {self.data_path}...")
        # ... Code đọc Parquet, train_test_split và map tokenize_function vào đây ...
        # return tokenized_datasets, class_weights

        label2id = {"Negative": 0, "Positive": 1, "Neutral": 2}

        df = pd.read_parquet(self.data_path)
        df["label"] = df["predicted_sentiment"].map(label2id).astype(int)

        # 3. Chia tập dữ liệu (80% Train, 20% Test/Validation)
        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )

        # 3. BƯỚC 2: Cưa đôi tập Temp (20%) -> 10% Validation và 10% Test
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
        )

        # 4. Gom tất cả vào định dạng Hugging Face Dataset
        raw_datasets = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(val_df),
                "test": Dataset.from_pandas(test_df),
            }
        )

        logger.info(
            f"Số lượng Train: {len(raw_datasets['train'])} | Val: {len(raw_datasets['validation'])} | Test: {len(raw_datasets['test'])}"
        )

        def tokenize_function(examples):
            return self.tokenizer(
                [str(x) for x in examples["segmented_comment"]],
                truncation=True,
                max_length=self.max_length,
            )

        # Map tokenization lên toàn bộ dataset
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        # --- A. TÍNH TRỌNG SỐ ---
        y_train = train_df["label"].values
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = torch.tensor(weights, dtype=torch.float32)

        return tokenized_datasets, class_weights

    def setup_trainer(self, tokenized_datasets, class_weights):
        """Bước 2: Cấu hình TrainingArguments và Custom Trainer"""
        # ... Setup training_args (FP16, batch_size, etc.)
        # ... Setup ImbalanceTrainer
        # return trainer
        # Tự động tạo Version
        os.makedirs(self.output_root, exist_ok=True)

        existing_versions = [
            d for d in os.listdir(self.output_root) if d.startswith("version_")
        ]
        new_version_id = (
            max([int(v.split("_")[1]) for v in existing_versions] + [0]) + 1
        )

        self.current_output_dir = os.path.join(
            self.output_root, f"version_{new_version_id}"
        )
        os.makedirs(self.current_output_dir, exist_ok=True)

        # 1. Load Model (Khởi tạo 3 nhãn)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=3
        )

        # (Tùy chọn) Đóng băng 8 lớp đầu tiên để train nhanh hơn & chống Overfit
        # for name, param in model.roberta.encoder.layer[:8].named_parameters():
        # param.requires_grad = False

        # 2. Cấu hình thông số huấn luyện
        training_args = TrainingArguments(
            output_dir=self.current_output_dir,
            learning_rate=2e-5,  # LR nhỏ an toàn
            per_device_train_batch_size=self.batch_size,  # Phù hợp GPU 15GB VRAM
            per_device_eval_batch_size=64,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",  # Đánh giá sau mỗi epoch
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",  # Ưu tiên model có F1-Macro cao nhất
            fp16=True,  # <--- VŨ KHÍ TỐI THƯỢNG (BẬT LÊN)
            dataloader_num_workers=2,  # Dùng 2 lõi CPU để đẩy data vào GPU nhanh hơn
            report_to="mlflow",
        )

        # 3. Chạy Custom Trainer
        trainer = ImbalanceTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets[
                "validation"
            ],  # Dùng Validation để check trong lúc học
            data_collator=self.data_collator,
            compute_metrics=lambda p: {
                "f1_macro": f1_score(
                    p.label_ids, np.argmax(p.predictions, axis=1), average="macro"
                ),
                "accuracy": accuracy_score(
                    p.label_ids, np.argmax(p.predictions, axis=1)
                ),
            },
        )

        return trainer

    def run_pipeline(self):
        """Bước 3: Nút bấm 'kích nổ' toàn bộ quy trình"""
        tokenized_datasets, class_weights = self.load_and_prepare_data()
        trainer = self.setup_trainer(tokenized_datasets, class_weights)

        logger.info("🚀 Bắt đầu Retrain tự động...")
        trainer.train()

        # BỔ SUNG BƯỚC QUAN TRỌNG: Đánh giá trên tập Test
        logger.info("📊 Đang đánh giá model trên tập Test...")
        test_results = trainer.evaluate(
            eval_dataset=tokenized_datasets["test"], metric_key_prefix="test"
        )
        logger.info(
            f"Kết quả tập Test: F1-Macro: {test_results['test_f1_macro']:.4f} | Accuracy: {test_results['test_accuracy']:.4f}"
        )

        logger.info("💾 Đang lưu model tốt nhất...")
        trainer.save_model(self.current_output_dir)
        self.tokenizer.save_pretrained(self.current_output_dir)
        logger.info(f"✅ Pipeline hoàn tất! Model lưu tại {self.current_output_dir}")


# ==========================================
# CÁCH HỆ THỐNG MLOPS GỌI CHẠY (Trong file main.py hoặc Airflow task)
# ==========================================
if __name__ == "__main__":
    # Đọc config từ file hoặc biến môi trường
    my_config = {
        "data_path": "data/comments_thang_5.parquet",
        "batch_size": 32,
        "max_length": 64,
        "output_dir": "./production_models",
    }

    pipeline = PhoBertSentimentTrainer(my_config)
    pipeline.run_pipeline()
