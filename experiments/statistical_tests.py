"""
Statistical Significance Analysis: Wilcoxon Signed-Rank Test.

Implements the Wilcoxon signed-rank test and effect size calculation
according to Tarawneh et al., Section IV-D:
- p-value of paired tests (Table 3)
- Sum of positive ranks (W+)
- Sum of negative ranks (W-)
- Wilcoxon Z statistic
- Effect size R = |Z| / sqrt(N) (Table 4)
- Effect categorization: Small (R <= 0.1), Medium (0.1 < R < 0.5), Large (R >= 0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm


@dataclass
class WilcoxonComparisonResult:
    method_a: str
    method_b: str
    n_total: int
    n_nonzero: int
    w_plus: float
    w_minus: float
    z_stat: float
    p_value: float
    effect_size_r: float
    effect_magnitude: str
    significant_at_05: bool

    def summary_string(self) -> str:
        return f"{self.effect_size_r:.2f} / {int(round(self.w_plus))} / {int(round(self.w_minus))}"


def compute_wilcoxon_signed_rank(
    scores_a: Union[np.ndarray, List[float]],
    scores_b: Union[np.ndarray, List[float]],
    name_a: str = "SMOTEFUNA",
    name_b: str = "Baseline",
    zero_method: str = "wilcox",
) -> WilcoxonComparisonResult:
    """
    Computes the paired Wilcoxon signed-rank test and effect size R.

    Parameters
    ----------
    scores_a : array-like of shape (N,)
        Scores for proposed method (e.g. SMOTEFUNA).
    scores_b : array-like of shape (N,)
        Scores for baseline method (e.g. SMOTE, ADASYN, SWIM, OR).
    name_a : str
        Name of method A.
    name_b : str
        Name of method B.
    zero_method : str
        Method to handle zero differences ('wilcox' discards zeros).

    Returns
    -------
    WilcoxonComparisonResult
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have the same length.")

    n_total = len(a)
    diff = a - b

    # Filter non-zero differences
    nonzero_mask = np.abs(diff) > 1e-12
    diff_nz = diff[nonzero_mask]
    n_nonzero = len(diff_nz)

    if n_nonzero == 0:
        # All differences are zero
        return WilcoxonComparisonResult(
            method_a=name_a,
            method_b=name_b,
            n_total=n_total,
            n_nonzero=0,
            w_plus=0.0,
            w_minus=0.0,
            z_stat=0.0,
            p_value=1.0,
            effect_size_r=0.0,
            effect_magnitude="Small (R <= 0.1)",
            significant_at_05=False,
        )

    # Rank absolute differences
    abs_diff = np.abs(diff_nz)
    ranks = rankdata(abs_diff, method="average")

    w_plus = float(np.sum(ranks[diff_nz > 0]))
    w_minus = float(np.sum(ranks[diff_nz < 0]))

    # Mean and standard deviation under H0
    # Tie correction for rank variance
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_corr = np.sum(tie_counts**3 - tie_counts) / 48.0

    mean_w = n_nonzero * (n_nonzero + 1) / 4.0
    var_w = (n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1)) / 24.0 - tie_corr
    std_w = np.sqrt(max(var_w, 1e-9))

    # Continuity correction
    z_stat = (w_plus - mean_w - 0.5 * np.sign(w_plus - mean_w)) / std_w

    # Two-sided p-value
    p_value = float(2.0 * norm.sf(abs(z_stat)))

    # Effect size R = |Z| / sqrt(N)
    effect_r = abs(z_stat) / np.sqrt(n_total)

    if effect_r >= 0.5:
        magnitude = "Large (R >= 0.5)"
    elif effect_r >= 0.1:
        magnitude = "Medium (0.1 < R < 0.5)"
    else:
        magnitude = "Small (R <= 0.1)"

    significant = p_value < 0.05

    return WilcoxonComparisonResult(
        method_a=name_a,
        method_b=name_b,
        n_total=n_total,
        n_nonzero=n_nonzero,
        w_plus=w_plus,
        w_minus=w_minus,
        z_stat=float(z_stat),
        p_value=p_value,
        effect_size_r=float(effect_r),
        effect_magnitude=magnitude,
        significant_at_05=significant,
    )


def compute_statistical_suite(
    results_matrix: Dict[str, Dict[str, List[float]]],
    proposed_method: str = "SMOTEFUNA",
    baseline_methods: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes Table 3 (p-values) and Table 4 (Wilcoxon R, W+, W-) across all
    configurations:
    - (AUC, SVM)
    - (AUC, NB)
    - (AUPR, SVM)
    - (AUPR, NB)

    Parameters
    ----------
    results_matrix : dict
        Nested dictionary:
        results_matrix[classifier][metric][method] -> list of scores across datasets.
    proposed_method : str, default='SMOTEFUNA'
    baseline_methods : list, default=['SMOTE', 'ADASYN', 'SWIM', 'Original']

    Returns
    -------
    table3_df : pd.DataFrame (p-values)
    table4_df : pd.DataFrame (R, W+, W-)
    """
    if baseline_methods is None:
        baseline_methods = ["SMOTE", "ADASYN", "SWIM", "Original"]

    configurations = [
        ("AUC", "SVM"),
        ("AUC", "NB"),
        ("AUPR", "SVM"),
        ("AUPR", "NB"),
    ]

    p_val_records = []
    r_w_records = []

    for metric, clf in configurations:
        p_row = {"Measure": metric, "Classifier": clf}
        r_row = {"Measure": metric, "Classifier": clf}

        if clf in results_matrix and metric in results_matrix[clf]:
            method_dict = results_matrix[clf][metric]
            scores_prop = method_dict.get(proposed_method, [])

            for base in baseline_methods:
                scores_base = method_dict.get(base, [])
                if len(scores_prop) > 0 and len(scores_base) == len(scores_prop):
                    res = compute_wilcoxon_signed_rank(
                        scores_prop,
                        scores_base,
                        name_a=proposed_method,
                        name_b=base,
                    )
                    p_row[base] = f"{res.p_value:.4e}" if res.p_value < 0.001 else f"{res.p_value:.4f}"
                    r_row[base] = res.summary_string()
                else:
                    p_row[base] = "N/A"
                    r_row[base] = "N/A"

        p_val_records.append(p_row)
        r_w_records.append(r_row)

    table3_df = pd.DataFrame(p_val_records)
    table4_df = pd.DataFrame(r_w_records)

    return table3_df, table4_df
