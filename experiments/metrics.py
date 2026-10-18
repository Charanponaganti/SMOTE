"""
Evaluation Metrics & Win Count Analysis.

Implements the exact evaluation criteria reported in the SMOTEFUNA paper:
- Area under the ROC curve (ROC-AUC)
- Area under the Precision-Recall curve (AUPR)
- Win count matrix across datasets with tie handling (matching Table 2)
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC-AUC)."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    # Check for single-class edge cases
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        return 0.5

    try:
        score = roc_auc_score(y_true_arr, y_score_arr)
        return float(score)
    except Exception:
        return 0.5


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes Area Under the Precision-Recall Curve (AUPR).
    Uses average_precision_score with the minority class (1) as the positive class.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        # Default baseline prior
        return float(np.mean(y_true_arr == 1))

    try:
        score = average_precision_score(y_true_arr, y_score_arr, pos_label=1)
        return float(score)
    except Exception:
        return float(np.mean(y_true_arr == 1))


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """Computes both ROC-AUC and AUPR metrics."""
    return {
        "AUC": compute_roc_auc(y_true, y_score),
        "AUPR": compute_aupr(y_true, y_score),
    }


def compute_win_counts(
    summary_df: pd.DataFrame,
    tol: float = 1e-4,
) -> pd.Series:
    """
    Computes the number of wins for each algorithm across datasets.
    In case of equal performance within tolerance `tol`, all tied methods
    are counted as winners (matching Table 2 of Tarawneh et al., 2020).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Rows are datasets, columns are resampling methods, values are mean scores.
    tol : float, default=1e-4
        Numerical tolerance for considering performances equal.

    Returns
    -------
    pd.Series
        Number of wins for each method.
    """
    methods = summary_df.columns
    wins = pd.Series(0, index=methods, dtype=int)

    for _, row in summary_df.iterrows():
        max_val = row.max()
        winners = [m for m in methods if abs(row[m] - max_val) <= tol]
        for winner in winners:
            wins[winner] += 1

    return wins
