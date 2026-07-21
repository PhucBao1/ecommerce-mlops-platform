import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None
DEVICE = None

LABEL_MAP = {0: -1.0, 1: 1.0, 2: 0.0}


def load_model():

    global MODEL, TOKENIZER, DEVICE

    if MODEL is None:

        TOKENIZER = AutoTokenizer.from_pretrained(
            "/app/artifacts/nlp_models/phobert/version_001", local_files_only=True
        )

        MODEL = AutoModelForSequenceClassification.from_pretrained(
            "/app/artifacts/nlp_models/phobert/version_001", local_files_only=True
        )

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        MODEL.to(DEVICE)
        MODEL.eval()

    # 🔥 warm-up dummy forward (QUAN TRỌNG)
    dummy = TOKENIZER(
        ["warmup"], return_tensors="pt", padding=True, truncation=True, max_length=64
    )

    dummy = {k: v.to(DEVICE) for k, v in dummy.items()}

    with torch.no_grad():
        MODEL(**dummy)

    print("✅ Model ready (warmed)")


def predict_sentiment(text):

    inputs = TOKENIZER(
        [text], truncation=True, padding=True, max_length=64, return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = MODEL(**inputs)
        logger.info(f"SENTIMENT DEBUG: {text} -> {outputs.logits}")

        pred = torch.argmax(outputs.logits, dim=1).item()

    return LABEL_MAP[pred]

    """with torch.no_grad():
        outputs = MODEL(**inputs)
        logger.info(f"SENTIMENT DEBUG: {text} -> {outputs.logits}")

        # Softmax ra xác suất
        probs = F.softmax(outputs.logits, dim=1)
        # Weighted score: positive - negative
        score = probs[0, 0]*(-1.0) + probs[0, 1]*0.0 + probs[0, 2]*1.0

    return score.item()"""
