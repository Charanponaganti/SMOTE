"""
2D demo of SMOTEFUNA vs original imbalance.

Run from repo root:
    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smotefuna import SMOTEFUNA  # noqa: E402


def main() -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        class_sep=1.2,
        random_state=42,
    )

    sampler = SMOTEFUNA(random_state=42)
    X_bal, y_bal = sampler.fit_resample(X, y)

    print(f"Before: majority={np.sum(y == 0)}, minority={np.sum(y == 1)}")
    print(f"After:  majority={np.sum(y_bal == 0)}, minority={np.sum(y_bal == 1)}")

    # Quick classifier check (train on resampled, test on held-out original split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train_bal, y_train_bal = SMOTEFUNA(random_state=0).fit_resample(X_train, y_train)

    for name, clf in [
        ("Naive Bayes", GaussianNB()),
        ("SVM (RBF)", SVC(probability=True, random_state=0)),
    ]:
        clf_raw = clf.__class__(**clf.get_params())
        clf_bal = clf.__class__(**clf.get_params())
        clf_raw.fit(X_train, y_train)
        clf_bal.fit(X_train_bal, y_train_bal)
        auc_raw = roc_auc_score(y_test, clf_raw.predict_proba(X_test)[:, 1])
        auc_bal = roc_auc_score(y_test, clf_bal.predict_proba(X_test)[:, 1])
        print(f"{name:12s}  AUC original={auc_raw:.3f}  AUC +SMOTEFUNA={auc_bal:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, Xx, yy, title in [
        (axes[0], X, y, "Original (imbalanced)"),
        (axes[1], X_bal, y_bal, "After SMOTEFUNA"),
    ]:
        ax.scatter(
            Xx[yy == 0, 0],
            Xx[yy == 0, 1],
            c="#4A6FA5",
            s=22,
            alpha=0.7,
            label="majority",
            edgecolors="none",
        )
        ax.scatter(
            Xx[yy == 1, 0],
            Xx[yy == 1, 1],
            c="#C45C26",
            s=36,
            alpha=0.85,
            label="minority",
            edgecolors="none",
        )
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
        ax.grid(True, alpha=0.25)

    fig.suptitle("SMOTEFUNA demo", fontsize=13)
    fig.tight_layout()
    out = ROOT / "examples" / "smotefuna_demo.png"
    fig.savefig(out, dpi=150)
    print(f"Saved plot: {out}")
    if "--show" in sys.argv:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
