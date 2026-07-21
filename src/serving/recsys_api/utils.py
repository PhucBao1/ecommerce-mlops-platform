# =========================================================
# FILE: utils.py
# =========================================================

import pandas as pd


def safe_transform(mapping, values):

    values = pd.Series(values)

    unknown_idx = len(mapping)

    return values.map(mapping).fillna(unknown_idx).astype(int)
