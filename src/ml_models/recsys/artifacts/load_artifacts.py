import joblib
import torch


def load_artifacts(base_path):

    user_encoder = joblib.load(f"{base_path}/encoders/user_encoder.pkl")

    item_encoder = joblib.load(f"{base_path}/encoders/item_encoder.pkl")

    cat_encoder = joblib.load(f"{base_path}/encoders/cat_encoder.pkl")

    scaler = joblib.load(f"{base_path}/scalers/scaler.pkl")

    return (user_encoder, item_encoder, cat_encoder, scaler)
