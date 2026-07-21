"""
Training script for SASRec (sequential next-item prediction).

Real data constraint (checked against artifacts/recsys_models/data_menu/user_history.parquet,
the same file loaders.py reads for serving): we only have REVIEW events (one row
per comment a customer left), not a clickstream with distinct purchase/click/ignore
actions, and no "shown but ignored" negatives. Two consequences vs. the original
version of this script:

  1. No DIN training. DIN needs real impression negatives (shown, not clicked) to
     learn a meaningful click model -- we don't have that signal, only positive
     reviews, so a DIN trained here would just be memorizing noise. Dropped
     entirely rather than shipping a model that looks trained but isn't valid.
  2. No 30-minute "session" splitting. That assumes clickstream-granularity
     timestamps; here `purchased_at` is a review timestamp and a given customer's
     reviews are typically weeks/months apart, so every pair would exceed the gap
     and every "session" would collapse to length 1. We instead treat each
     customer's full chronological review history as one sequence -- the standard
     setup for SASRec when only long-horizon interaction history is available.

Data reality check (run once, see if it changed):
    customers total: 78,076 | with >=2 reviews (usable for next-item pairs): 13,219 (~17%)
    median reviews/customer: 1 | n_items: 1,127
  Most customers have exactly one review, so SASRec only really learns from the
  ~17% with a real sequence. Expect modest NDCG@10 given this sparsity -- that's
  a data-scale limitation, not a bug.

Pipeline:
  1. Load user_history.parquet, drop the customer_id="0" placeholder (fillna
     artifact from preprocessing.py, not a real customer), keep only customers
     with >= 2 reviews.
  2. Train SASRec: next-item prediction on each customer's chronological sequence.
  3. Export model + item encoder to artifacts, log NDCG@10 to MLflow.

Usage:
    python -m src.ml_models.recsys.training.train_sasrec
    python -m src.ml_models.recsys.training.train_sasrec --epochs 20 --dim 64
"""

import argparse
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.ml_models.recsys.evaluation.metrics import ndcg_at_k
from src.ml_models.recsys.models.sasrec import SASRec
from src.ml_models.recsys.training.mlflow_logger import setup_mlflow

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)

_DATA_DIR = os.getenv("DATA_DIR", "/app/artifacts/recsys_models/data_menu")
_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/artifacts/recsys_models"))
_SEED = 42
_MAX_SEQ = 50
_MIN_SEQ_LEN = 2  # need >= 2 reviews to form one (input, target) next-item pair

torch.manual_seed(_SEED)
np.random.seed(_SEED)


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def _prepare_sequences(history: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the null-customer placeholder and customers with too few reviews to
    form a next-item pair. No session splitting -- see module docstring.
    """
    history = history[history["customer_id"] != "0"].copy()
    counts = history.groupby("customer_id")["customer_id"].transform("size")
    history = history[counts >= _MIN_SEQ_LEN]
    history["session_id"] = history["customer_id"]  # one sequence per customer
    return history.sort_values(["customer_id", "purchased_at"])


def _encode_items(history: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    item_ids = sorted(history["product_id"].unique())
    item_enc = {iid: i + 1 for i, iid in enumerate(item_ids)}  # 0 = padding
    history["item_idx"] = history["product_id"].map(item_enc).fillna(0).astype(int)
    return history, item_enc


# ---------------------------------------------------------------------------
# SASRec Dataset
# ---------------------------------------------------------------------------


class SASRecDataset(Dataset):
    """
    For each session, produce (input_seq, pos_seq, neg_seq) triples.
    input_seq[t] → pos_seq[t] = item at t+1 (next-item prediction).
    """

    def __init__(self, sessions: pd.DataFrame, n_items: int, max_len: int = _MAX_SEQ):
        self.samples = []
        self.n_items = n_items

        for _, group in sessions.groupby("session_id"):
            items = group.sort_values("purchased_at")["item_idx"].tolist()
            if len(items) < 2:
                continue

            # Input: all but last; target: all but first (shifted by 1)
            seq = items[:-1]
            pos = items[1:]

            # Truncate + pad to max_len
            seq = seq[-max_len:]
            pos = pos[-max_len:]
            pad_len = max_len - len(seq)
            seq = [0] * pad_len + seq
            pos = [0] * pad_len + pos

            # Random negative for each position
            neg = []
            pos_set = set(items)
            for _ in pos:
                while True:
                    n = np.random.randint(1, n_items + 1)
                    if n not in pos_set:
                        break
                neg.append(n)

            self.samples.append((seq, pos, neg))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, pos, neg = self.samples[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Evaluation — NDCG@10 for SASRec
# ---------------------------------------------------------------------------


def _eval_sasrec(
    model: SASRec,
    val_sessions: pd.DataFrame,
    n_items: int,
    device,
    eval_batch_size: int = 256,
) -> float:
    model.eval()
    all_items = torch.arange(1, n_items + 1, device=device)
    ndcg_scores = []

    seqs: list[list[int]] = []
    targets: list[int] = []
    for _, group in val_sessions.groupby("session_id"):
        items = group.sort_values("purchased_at")["item_idx"].tolist()
        if len(items) < 2:
            continue
        seq = items[:-1][-_MAX_SEQ:]
        seq = [0] * (_MAX_SEQ - len(seq)) + seq
        seqs.append(seq)
        targets.append(items[-1])

    if not seqs:
        return 0.0

    with torch.no_grad():
        for start in range(0, len(seqs), eval_batch_size):
            batch_seq = torch.tensor(
                seqs[start : start + eval_batch_size], dtype=torch.long, device=device
            )
            batch_targets = targets[start : start + eval_batch_size]
            scores = (
                model.predict_next(batch_seq, all_items).cpu().numpy()
            )  # [B, n_items]
            ranked = np.argsort(-scores, axis=1)
            for row_ranked, target in zip(ranked, batch_targets):
                ranked_ids = (
                    row_ranked + 1
                ).tolist()  # item_idx 1-indexed, khớp all_items
                ndcg_scores.append(ndcg_at_k([target], ranked_ids, k=10))

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    n_epochs: int = 20, embedding_dim: int = 64, lr: float = 1e-3, batch_size: int = 512
) -> dict:
    setup_mlflow()
    mlflow.set_experiment("sasrec_training")

    # Load data -- real review interactions, see module docstring for why
    # there's no action/label column and no session splitting.
    raw = pd.read_parquet(Path(_DATA_DIR) / "user_history.parquet")
    raw["purchased_at"] = pd.to_datetime(raw["purchased_at"])
    raw, item_enc = _encode_items(raw)
    n_items = len(item_enc)

    sessions = _prepare_sequences(raw)
    n_customers = sessions["customer_id"].nunique()
    logger.info(
        "Usable sequences: %d customers (>= %d reviews each), %d items",
        n_customers,
        _MIN_SEQ_LEN,
        n_items,
    )

    print(
        f"Usable sequences: {n_customers} customers (>= {_MIN_SEQ_LEN} reviews "
        f"each), {n_items} items",
        flush=True,
    )

    cutoff = raw["purchased_at"].quantile(0.8)
    train_sess = sessions[sessions["purchased_at"] < cutoff]
    val_sess = sessions[sessions["purchased_at"] >= cutoff]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with mlflow.start_run(run_name="sasrec"):
        mlflow.log_params(
            {
                "n_items": n_items,
                "n_customers_used": n_customers,
                "min_seq_len": _MIN_SEQ_LEN,
                "embedding_dim": embedding_dim,
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
            }
        )

        sasrec = SASRec(n_items=n_items, d_model=embedding_dim).to(device)
        opt_sr = optim.Adam(sasrec.parameters(), lr=lr)
        ds_sr = SASRecDataset(train_sess, n_items)
        loader_sr = DataLoader(ds_sr, batch_size=batch_size, shuffle=True)

        best_sasrec_ndcg = 0.0
        for epoch in range(1, n_epochs + 1):
            sasrec.train()
            total_loss = 0.0
            for seq, pos, neg in loader_sr:
                seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
                loss = sasrec(seq, pos, neg)
                opt_sr.zero_grad()
                loss.backward()
                opt_sr.step()
                total_loss += loss.item()

            ndcg = _eval_sasrec(sasrec, val_sess, n_items, device)
            if ndcg > best_sasrec_ndcg:
                best_sasrec_ndcg = ndcg
            mlflow.log_metrics(
                {"sasrec_loss": total_loss / len(loader_sr), "sasrec_ndcg10": ndcg},
                step=epoch,
            )
            logger.info(
                "[SASRec] epoch=%d loss=%.4f ndcg@10=%.4f",
                epoch,
                total_loss / len(loader_sr),
                ndcg,
            )
            print(
                f"[SASRec] epoch={epoch} loss={total_loss / len(loader_sr):.4f} "
                f"ndcg@10={ndcg:.4f}",
                flush=True,
            )

        # ── Save model ───────────────────────────────────────────────────
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        sasrec_path = _MODEL_DIR / "sasrec.pt"
        torch.save(sasrec.state_dict(), sasrec_path)

        import json

        enc_path = _MODEL_DIR / "sasrec_item_enc.json"
        enc_path.write_text(json.dumps(item_enc))

        mlflow.log_artifact(str(sasrec_path), "sasrec_model")
        mlflow.log_artifact(str(enc_path), "encoders")
        mlflow.log_metrics({"best_sasrec_ndcg10": best_sasrec_ndcg})
        logger.info("Model saved. SASRec best NDCG@10=%.4f", best_sasrec_ndcg)

    return {"sasrec_ndcg10": best_sasrec_ndcg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=512)
    args = parser.parse_args()
    results = train(
        n_epochs=args.epochs, embedding_dim=args.dim, lr=args.lr, batch_size=args.batch
    )
    print(f"\nSASRec NDCG@10: {results['sasrec_ndcg10']:.4f}")
