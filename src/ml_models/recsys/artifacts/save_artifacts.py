import joblib
import torch


def save_artifacts(model, scaler, user_encoder, item_encoder, cat_encoder):

    torch.save(model.state_dict(), "artifacts/recsys_models/model/best_two_tower.pt")

    joblib.dump(user_encoder, "artifacts/recsys_models/encoders/user_encoder.pkl")

    joblib.dump(item_encoder, "artifacts/recsys_models/encoders/item_encoder.pkl")

    joblib.dump(cat_encoder, "artifacts/recsys_models/encoders/cat_encoder.pkl")

    joblib.dump(scaler, "artifacts/recsys_models/scalers/scaler.pkl")
