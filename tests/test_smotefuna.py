"""Unit tests for SMOTEFUNA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smotefuna import SMOTEFUNA


def _toy_data(n_maj=40, n_min=8, seed=0):
    rng = np.random.default_rng(seed)
    maj = rng.normal(loc=[0.0, 0.0], scale=0.6, size=(n_maj, 2))
    mi = rng.normal(loc=[3.0, 3.0], scale=0.4, size=(n_min, 2))
    X = np.vstack([maj, mi])
    y = np.array([0] * n_maj + [1] * n_min)
    return X, y


def test_balances_to_majority_count():
    X, y = _toy_data()
    Xs, ys = SMOTEFUNA(random_state=1).fit_resample(X, y)
    assert np.sum(ys == 0) == np.sum(ys == 1)
    assert len(Xs) == 2 * np.sum(y == 0)


def test_reproducible_with_seed():
    X, y = _toy_data()
    a = SMOTEFUNA(random_state=7).fit_resample(X, y)
    b = SMOTEFUNA(random_state=7).fit_resample(X, y)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_sampling_strategy_ratio():
    X, y = _toy_data(n_maj=50, n_min=10)
    Xs, ys = SMOTEFUNA(sampling_strategy=0.5, random_state=2).fit_resample(X, y)
    n_min = int(np.sum(ys == 1))
    n_maj = int(np.sum(ys == 0))
    assert n_min == 25
    assert n_maj == 50


def test_rejects_single_minority():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])
    y = np.array([0, 0, 0, 1])
    with pytest.raises(ValueError, match="at least 2 minority"):
        SMOTEFUNA(random_state=0).fit_resample(X, y)


def test_already_balanced_returns_same_size():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.array([0] * 10 + [1] * 10)
    Xs, ys = SMOTEFUNA(random_state=0).fit_resample(X, y)
    assert len(Xs) == len(X)


def test_accepted_points_nearer_to_minority():
    """Spot-check: every synthetic should satisfy theta <= beta on original sets."""
    X, y = _toy_data(n_maj=60, n_min=12, seed=3)
    minority = X[y == 1]
    majority = X[y == 0]
    Xs, ys = SMOTEFUNA(shuffle=False, random_state=3).fit_resample(X, y)
    synthetics = Xs[len(X) :]  # appended when shuffle=False
    assert len(synthetics) == np.sum(y == 0) - np.sum(y == 1)
    for s in synthetics:
        theta = np.min(np.sum(np.abs(minority - s), axis=1))
        beta = np.min(np.sum(np.abs(majority - s), axis=1))
        assert theta <= beta + 1e-9
