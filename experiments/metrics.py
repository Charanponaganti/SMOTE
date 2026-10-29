"""AUC, AUPR, and how we count wins.

Paper counts ties as wins for everyone involved (Table 2).
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    # a fold with only one class can't make an ROC
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        return 0.5

    try:
        score = roc_auc_score(y_true_arr, y_score_arr)
        return float(score)
    except Exception:
        return 0.5


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # minority = 1, same as the paper
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        return float(np.mean(y_true_arr == 1))

    try:
        score = average_precision_score(y_true_arr, y_score_arr, pos_label=1)
        return float(score)
    except Exception:
        return float(np.mean(y_true_arr == 1))


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    return {
        "AUC": compute_roc_auc(y_true, y_score),
        "AUPR": compute_aupr(y_true, y_score),
    }


def compute_win_counts(
    summary_df: pd.DataFrame,
    tol: float = 1e-4,
) -> pd.Series:
    """Count how many datasets each method wins. ties count for all of them."""
    methods = summary_df.columns
    wins = pd.Series(0, index=methods, dtype=int)

    for _, row in summary_df.iterrows():
        max_val = row.max()
        winners = [m for m in methods if abs(row[m] - max_val) <= tol]
        for winner in winners:
            wins[winner] += 1

    return wins

