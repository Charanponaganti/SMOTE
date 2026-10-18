"""Unit and integration tests for the benchmark experiment suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines import (
    ADASYNBaseline,
    OriginalDataBaseline,
    SMOTEBaseline,
    SWIMBaseline,
    get_resampler,
)
from experiments.datasets import (
    TABLE_1_DATASETS,
    get_table1_metadata,
    load_benchmark_dataset,
)
from experiments.metrics import (
    compute_all_metrics,
    compute_aupr,
    compute_roc_auc,
    compute_win_counts,
)
from experiments.pipeline import BenchmarkPipeline
from experiments.statistical_tests import (
    compute_wilcoxon_signed_rank,
)


def _toy_imbalanced_data(n_maj=60, n_min=10, seed=42):
    rng = np.random.default_rng(seed)
    maj = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(n_maj, 2))
    mi = rng.normal(loc=[2.5, 2.5], scale=0.8, size=(n_min, 2))
    X = np.vstack([maj, mi])
    y = np.array([0] * n_maj + [1] * n_min)
    return X, y


# --- Baselines Tests ---


def test_smote_baseline():
    X, y = _toy_imbalanced_data()
    smote = SMOTEBaseline(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    assert np.sum(y_res == 0) == np.sum(y_res == 1)
    assert len(X_res) == 2 * np.sum(y == 0)


def test_adasyn_baseline():
    X, y = _toy_imbalanced_data()
    adasyn = ADASYNBaseline(n_neighbors=5, random_state=42)
    X_res, y_res = adasyn.fit_resample(X, y)
    assert np.sum(y_res == 1) >= np.sum(y == 1)
    assert len(X_res) > len(X)


def test_swim_baseline():
    X, y = _toy_imbalanced_data()
    swim = SWIMBaseline(random_state=42)
    X_res, y_res = swim.fit_resample(X, y)
    assert np.sum(y_res == 0) == np.sum(y_res == 1)
    assert len(X_res) == 2 * np.sum(y == 0)


def test_original_data_baseline():
    X, y = _toy_imbalanced_data()
    orig = OriginalDataBaseline()
    X_res, y_res = orig.fit_resample(X, y)
    assert len(X_res) == len(X)
    np.testing.assert_array_equal(y_res, y)


def test_get_resampler_factory():
    res = get_resampler("SMOTEFUNA", random_state=1)
    assert res.__class__.__name__ == "SMOTEFUNA"

    with pytest.raises(ValueError, match="Unknown resampler"):
        get_resampler("UNKNOWN_ALGO")


# --- Datasets Tests ---


def test_table1_metadata():
    df = get_table1_metadata()
    assert len(df) == 33
    assert "IR" in df.columns
    assert "nInes" in df.columns
    assert "nAt" in df.columns


def test_load_benchmark_dataset():
    # Test loading real Wisconsin
    X_wisc, y_wisc, meta_wisc = load_benchmark_dataset("D27")
    assert meta_wisc.name == "Wisconsin"
    assert len(X_wisc) > 500
    assert len(np.unique(y_wisc)) == 2

    # Test loading Bank dataset
    X_bank, y_bank, meta_bank = load_benchmark_dataset("D12")
    assert meta_bank.name == "Bank"
    assert X_bank.shape[1] == 4


# --- Metrics Tests ---


def test_metrics_computation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    auc = compute_roc_auc(y_true, y_score)
    aupr = compute_aupr(y_true, y_score)
    assert auc == 1.0
    assert aupr == 1.0

    all_m = compute_all_metrics(y_true, y_score)
    assert all_m["AUC"] == 1.0
    assert all_m["AUPR"] == 1.0


def test_win_counts_with_ties():
    import pandas as pd

    data = {
        "SMOTEFUNA": [0.95, 0.80, 0.90],
        "SMOTE": [0.90, 0.80, 0.85],
        "ADASYN": [0.85, 0.70, 0.88],
    }
    df = pd.DataFrame(data)
    wins = compute_win_counts(df)
    assert wins["SMOTEFUNA"] == 3  # Wins row 0, ties row 1, wins row 2
    assert wins["SMOTE"] == 1       # Ties row 1


# --- Statistical Tests ---


def test_wilcoxon_signed_rank():
    # If SMOTEFUNA strictly dominates baseline
    scores_a = [0.95, 0.90, 0.88, 0.92, 0.96, 0.91, 0.89, 0.94]
    scores_b = [0.80, 0.81, 0.75, 0.78, 0.82, 0.79, 0.80, 0.81]
    res = compute_wilcoxon_signed_rank(scores_a, scores_b, name_a="SMOTEFUNA", name_b="SMOTE")
    assert res.w_plus > res.w_minus
    assert res.p_value < 0.05
    assert res.significant_at_05 is True
    assert res.effect_size_r > 0.5


# --- Pipeline Integration Test ---


def test_pipeline_integration():
    pipeline = BenchmarkPipeline(
        methods=["SMOTEFUNA", "SMOTE", "Original"],
        n_splits=2,
        n_trials=1,
        random_state=42,
        verbose=False,
    )
    X, y, meta = load_benchmark_dataset("D4")
    res = pipeline.evaluate_dataset(X, y, meta)

    assert "NB" in res
    assert "SVM" in res
    assert "SMOTEFUNA" in res["NB"]
    assert len(res["NB"]["SMOTEFUNA"].auc_scores) == 2  # 2 folds
    assert res["NB"]["SMOTEFUNA"].auc_mean > 0.0
