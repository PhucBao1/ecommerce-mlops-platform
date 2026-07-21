"""
Optuna hyperparameter tuning for Two-Tower RecSys model.

Loads data once outside the trial loop (expensive), then for each Optuna trial
trains a model with sampled params for N quick epochs and returns NDCG@10.
Each trial is logged as a nested MLflow run under the parent study run.

Usage:
    python -m src.ml_models.recsys.training.tune_optuna
    python -m src.ml_models.recsys.training.tune_optuna --n-trials 30 --epochs 5
"""

import argparse
import logging
import os
from dataclasses import dataclass

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.ml_models.recsys.data.build_dataset import build_dataset
from src.ml_models.recsys.datasets.recsys_dataset import EcommerceRecSysDataset
from src.ml_models.recsys.models.two_tower import TwoTowerModel
from src.ml_models.recsys.training.mlflow_logger import setup_mlflow
from src.ml_models.recsys.training.trainer import train_one_epoch_infonce, validate
from src.ml_models.recsys.utils.encoding import fit_encoders, safe_transform
from src.ml_models.recsys.utils.scaling import scale_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class DataBundle:
    train_df: object
    valid_df: object
    num_users: int
    num_items: int
    num_categories: int
    scaler: object


def load_data_once() -> DataBundle:
    """Load and preprocess data — called once before all trials."""
    logger.info("Loading dataset (done once for all trials)...")
    train_df, valid_df, _, _, _ = build_dataset()

    train_df, user_enc, item_enc, cat_enc = fit_encoders(train_df)

    for df in (valid_df,):
        df["customer_id_idx"] = safe_transform(user_enc, df["customer_id"])
        df["product_id_idx"] = safe_transform(item_enc, df["product_id"])
        df["category_id_idx"] = safe_transform(cat_enc, df["category_id"].astype(str))

    train_df, valid_df, _, scaler = scale_features(train_df, valid_df, valid_df)

    return DataBundle(
        train_df=train_df,
        valid_df=valid_df,
        num_users=len(user_enc.classes_) + 1,
        num_items=len(item_enc.classes_) + 1,
        num_categories=len(cat_enc.classes_) + 1,
        scaler=scaler,
    )


def train_with_params(params: dict, data: DataBundle, n_epochs: int) -> float:
    """Train Two-Tower with given params, return best NDCG@10 on validation set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pos_df = data.train_df[data.train_df["label"] > 0.0].copy()
    train_loader = DataLoader(
        EcommerceRecSysDataset(train_pos_df),
        batch_size=params["batch_size"],
        shuffle=True,
    )
    valid_loader = DataLoader(
        EcommerceRecSysDataset(data.valid_df),
        batch_size=params["batch_size"],
        shuffle=False,
    )

    model = TwoTowerModel(
        num_users=data.num_users,
        num_items=data.num_items,
        num_categories=data.num_categories,
        embedding_dim=params["embedding_dim"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    best_ndcg = 0.0
    for epoch in range(n_epochs):
        train_one_epoch_infonce(model, train_loader, optimizer, device)
        _, _, _, ndcg = validate(model, valid_loader, criterion, device)
        if ndcg > best_ndcg:
            best_ndcg = ndcg

    return best_ndcg


def make_objective(data: DataBundle, n_epochs: int, parent_run_id: str):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }

        with mlflow.start_run(
            run_name=f"trial_{trial.number:02d}",
            nested=True,
            tags={"optuna_trial": str(trial.number)},
        ):
            mlflow.log_params(params)
            ndcg = train_with_params(params, data, n_epochs)
            mlflow.log_metric("ndcg_10", ndcg)
            logger.info(
                "Trial %d | embedding_dim=%d lr=%.2e wd=%.2e bs=%d → NDCG@10=%.4f",
                trial.number,
                params["embedding_dim"],
                params["learning_rate"],
                params["weight_decay"],
                params["batch_size"],
                ndcg,
            )

        return ndcg

    return objective


def main(n_trials: int = 20, n_epochs: int = 5) -> None:
    setup_mlflow()
    mlflow.set_experiment("optuna_hpo")

    data = load_data_once()

    with mlflow.start_run(run_name="optuna_study") as parent_run:
        mlflow.log_params({"n_trials": n_trials, "epochs_per_trial": n_epochs})

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(
            make_objective(data, n_epochs, parent_run.info.run_id),
            n_trials=n_trials,
        )

        best = study.best_params
        best_ndcg = study.best_value
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_ndcg_10", best_ndcg)

    print(f"\n{'='*50}")
    print(f"Best NDCG@10 : {best_ndcg:.4f}")
    print(f"Best params  : {best}")
    print(f"{'='*50}")
    print("\nTo retrain with best params, set in train_model.py:")
    print(f"  EMBEDDING_DIM = {best['embedding_dim']}")
    print(f"  BATCH_SIZE    = {best['batch_size']}")
    print(f"  LR            = {best['learning_rate']:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs per trial (fewer = faster, noisier estimate)",
    )
    args = parser.parse_args()
    main(n_trials=args.n_trials, n_epochs=args.epochs)
