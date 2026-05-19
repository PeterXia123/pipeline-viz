"""Feature engineering used by notebook 02."""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_log"] = np.log1p(out["amount"].astype(float))
    out["high_value"] = (out["amount"] > 100).astype(int)
    return out
