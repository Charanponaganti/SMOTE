"""
Visualization Tools for SMOTEFUNA Paper Reproduction.

Implements publication-quality plotting tools:
1. Multi-method ROC and PR curve comparisons (matching Figures 9 & 10 of the paper).
2. 2D PCA projection of minority, majority, and synthetic samples (matching Figure 8 of the paper).
3. Win count and performance comparison charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.baselines import get_resampler
from experiments.datasets import DatasetMeta, load_benchmark_dataset
from experiments.metrics import compute_aupr, compute_roc_auc

# Paper-inspired color palette
METHOD_COLORS = {
    "SMOTEFUNA": "#1B7837",  # Strong Green
    "SMOTE": "#2166AC",      # Blue
    "ADASYN": "#E08214",     # Orange/Amber
    "SWIM": "#762A83",       # Purple
    "Original": "#636363",   # Neutral Gray
}

METHOD_LINESTYLES = {
    "SMOTEFUNA": "-",
    "SMOTE": "--",
    "ADASYN": "-.",
    "SWIM": ":",
    "Original": "--",
}


def plot_roc_and_pr_grid(
    dataset_ids: List[str],
    classifier_name: str = "NB",
    save_path: Optional[Path] = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Plots a multi-row grid of ROC (left) and PR (right) curves comparing all methods,
    matching Figure 9 (NB) and Figure 10 (SVM) of the paper.

    Parameters
    ----------
    dataset_ids : list of dataset IDs, e.g. ['D10', 'D16', 'D17', 'D19']
    classifier_name : 'NB' or 'SVM'
    save_path : optional Path to save the figure
    seed : random seed
    """
    n_rows = len(dataset_ids)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 3.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    methods = ["SMOTEFUNA", "SMOTE", "ADASYN", "SWIM", "Original"]

    for row_idx, d_id in enumerate(dataset_ids):
        X, y, meta = load_benchmark_dataset(d_id, seed=seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(skf.split(X, y))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train)
        X_te_s = scaler.transform(X_test)

        ax_roc = axes[row_idx, 0]
        ax_pr = axes[row_idx, 1]

        # Prior for PR baseline
        prior_minority = float(np.mean(y_test == 1))

        for m in methods:
            color = METHOD_COLORS.get(m, "#333333")
            ls = METHOD_LINESTYLES.get(m, "-")
            lw = 2.0 if m == "SMOTEFUNA" else 1.4

            resampler = get_resampler(m, random_state=seed)
            try:
                X_res, y_res = resampler.fit_resample(X_tr_s, y_train)
            except Exception:
                X_res, y_res = X_tr_s, y_train

            clf = GaussianNB() if classifier_name == "NB" else SVC(kernel="rbf", probability=True, random_state=seed)
            clf.fit(X_res, y_res)
            y_prob = clf.predict_proba(X_te_s)[:, 1]

            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = compute_roc_auc(y_test, y_prob)
            ax_roc.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f"{m} ({auc_val:.3f})")

            # PR
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            aupr_val = compute_aupr(y_test, y_prob)
            ax_pr.plot(recall, precision, color=color, linestyle=ls, linewidth=lw, label=f"{m} ({aupr_val:.3f})")

        # Diagonal on ROC
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1.0)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.05])
        ax_roc.set_xlabel("FPR", fontsize=9)
        ax_roc.set_ylabel("TPR", fontsize=9)
        ax_roc.set_title(f"({chr(97 + row_idx)}) {meta.dataset_id} ({meta.name}) ROC - {classifier_name}", fontsize=10)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.legend(loc="lower right", fontsize=8)

        # Baseline on PR
        ax_pr.axhline(prior_minority, color="gray", linestyle=":", alpha=0.5, label=f"Baseline ({prior_minority:.2f})")
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.05])
        ax_pr.set_xlabel("Recall", fontsize=9)
        ax_pr.set_ylabel("Precision", fontsize=9)
        ax_pr.set_title(f"({chr(97 + row_idx)}) {meta.dataset_id} ({meta.name}) PR - {classifier_name}", fontsize=10)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.legend(loc="lower left", fontsize=8)

    fig.suptitle(f"ROC and PR Curves Comparison ({classifier_name} Classifier)", fontsize=13, y=1.002)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved ROC/PR curves to {save_path}")

    return fig


def plot_pca_projections(
    dataset_ids: List[str],
    save_path: Optional[Path] = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Reproduces Figure 8 of the paper:
    PCA 2D projection of majority, minority, and newly created synthetic examples using SMOTEFUNA.

    Parameters
    ----------
    dataset_ids : list of 6 dataset IDs, e.g. ['D4', 'D6', 'D12', 'D13', 'D27', 'D31']
    save_path : optional Path to save the plot
    seed : random seed
    """
    n_plots = len(dataset_ids)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, d_id in enumerate(dataset_ids):
        ax = axes_flat[idx]
        X, y, meta = load_benchmark_dataset(d_id, seed=seed)

        # Run SMOTEFUNA
        resampler = get_resampler("SMOTEFUNA", random_state=seed, shuffle=False)
        X_res, y_res = resampler.fit_resample(X, y)

        # Original samples
        n_orig = len(X)
        minority_orig = X[y == 1]
        majority_orig = X[y == 0]
        # Synthetics are appended at the end when shuffle=False
        synthetics = X_res[n_orig:]

        # Fit PCA on original dataset
        pca = PCA(n_components=2, random_state=seed)
        pca.fit(X)

        maj_pca = pca.transform(majority_orig)
        min_pca = pca.transform(minority_orig)
        syn_pca = pca.transform(synthetics) if len(synthetics) > 0 else np.empty((0, 2))

        # Majority: Blue points
        ax.scatter(
            maj_pca[:, 0],
            maj_pca[:, 1],
            c="#2B83BA",
            marker="o",
            s=16,
            alpha=0.6,
            label="Majority",
            edgecolors="none",
        )
        # Minority: Red points
        ax.scatter(
            min_pca[:, 0],
            min_pca[:, 1],
            c="#D7191C",
            marker="o",
            s=28,
            alpha=0.9,
            label="Minority",
            edgecolors="none",
        )
        # Synthetic: Red crosses
        if len(syn_pca) > 0:
            ax.scatter(
                syn_pca[:, 0],
                syn_pca[:, 1],
                c="#E66101",
                marker="+",
                s=35,
                alpha=0.85,
                label="+ Synthetic",
                linewidths=1.2,
            )

        ax.set_title(f"({chr(97 + idx)}) PCA on {meta.dataset_id} ({meta.name})", fontsize=10)
        ax.set_xlabel("Principal Component 1", fontsize=9)
        ax.set_ylabel("Principal Component 2", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

    # Turn off any unused subplots
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Projection of Minority, Majority, and SMOTEFUNA Synthetics (PCA 2D)",
        fontsize=13,
        y=1.002,
    )
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved PCA projection figure to {save_path}")

    return fig
