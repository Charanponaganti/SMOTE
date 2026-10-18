"""
Baseline Resampling Algorithms.

Provides a unified interface for all resampling techniques evaluated in
Tarawneh et al., "SMOTEFUNA: Synthetic Minority Over-Sampling Technique
Based on Furthest Neighbour Algorithm", IEEE Access, 2020:
- SMOTEFUNA (Proposed method)
- SMOTE (Chawla et al., 2002)
- ADASYN (He et al., 2008)
- SWIM (Bellinger et al., 2019 - Sampling With the Majority class)
- Original Data (No-resampling baseline)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import pdist

from smotefuna.sampler import SMOTEFUNA

ArrayLike = Union[np.ndarray, list]


class OriginalDataBaseline:
    """Identity baseline (no resampling), evaluating models on original imbalanced data."""

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        return X_arr.copy(), y_arr.copy()


class SMOTEBaseline:
    """
    SMOTE baseline oversampler.
    Uses imbalanced-learn when available, with a robust fallback for small minority counts.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError(f"Binary classification required; got {len(classes)} classes.")

        min_idx = int(np.argmin(counts))
        n_min = counts[min_idx]
        n_maj = counts[int(np.argmax(counts))]

        if n_min >= n_maj or n_min < 2:
            return X_arr.copy(), y_arr.copy()

        # Adapt k_neighbors to available minority samples
        effective_k = min(self.k_neighbors, n_min - 1)
        if effective_k < 1:
            effective_k = 1

        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(k_neighbors=effective_k, random_state=self.random_state)
            return smote.fit_resample(X_arr, y_arr)
        except Exception:
            # Robust native SMOTE fallback
            return self._native_smote(X_arr, y_arr, effective_k)

    def _native_smote(
        self, X: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        min_label = classes[int(np.argmin(counts))]
        maj_label = classes[int(np.argmax(counts))]

        minority = X[y == min_label]
        n_to_add = len(X[y == maj_label]) - len(minority)
        if n_to_add <= 0:
            return X.copy(), y.copy()

        # Pairwise Euclidean distances within minority
        diffs = minority[:, np.newaxis, :] - minority[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dist_matrix, np.inf)

        k = min(k, len(minority) - 1)
        nn_indices = np.argsort(dist_matrix, axis=1)[:, :k]

        synthetics = np.empty((n_to_add, X.shape[1]), dtype=float)
        for i in range(n_to_add):
            idx = int(rng.integers(0, len(minority)))
            nn_idx = int(rng.choice(nn_indices[idx]))
            diff = minority[nn_idx] - minority[idx]
            gap = rng.uniform(0.0, 1.0)
            synthetics[i] = minority[idx] + gap * diff

        X_res = np.vstack([X, synthetics])
        y_res = np.concatenate([y, np.full(n_to_add, min_label, dtype=y.dtype)])
        return X_res, y_res


class ADASYNBaseline:
    """
    ADASYN baseline oversampler.
    Uses imbalanced-learn when available, with a robust fallback.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError(f"Binary classification required; got {len(classes)} classes.")

        min_idx = int(np.argmin(counts))
        n_min = counts[min_idx]
        n_maj = counts[int(np.argmax(counts))]

        if n_min >= n_maj or n_min < 2:
            return X_arr.copy(), y_arr.copy()

        effective_k = min(self.n_neighbors, n_min - 1)
        if effective_k < 1:
            effective_k = 1

        try:
            from imblearn.over_sampling import ADASYN

            adasyn = ADASYN(n_neighbors=effective_k, random_state=self.random_state)
            return adasyn.fit_resample(X_arr, y_arr)
        except Exception:
            # In cases where ADASYN cannot allocate ratios (e.g., zero boundary points),
            # fall back gracefully to SMOTE with effective_k
            return SMOTEBaseline(
                k_neighbors=effective_k, random_state=self.random_state
            ).fit_resample(X_arr, y_arr)


class SWIMBaseline:
    """
    SWIM (Sampling With the Majority class) oversampling algorithm.
    Based on Colin Bellinger et al., 'Framework for extreme imbalance classification:
    SWIM—sampling with the majority class', Knowledge and Information Systems, 2019.
    Implements the RBF mutual density sampling strategy with full scikit-learn API compatibility.
    """

    def __init__(
        self,
        epsilon: Optional[float] = None,
        steps: int = 5,
        tau: float = 0.25,
        random_state: Optional[int] = None,
    ) -> None:
        self.epsilon = epsilon
        self.steps = steps
        self.tau = tau
        self.random_state = random_state

    def _score(self, point: np.ndarray, majority: np.ndarray, eps: float) -> float:
        # Vectorized Manhattan distance and RBF density calculation
        dists = np.sum(np.abs(majority - point), axis=1)
        return float(np.sum(np.exp(-((dists * eps) ** 2))))

    def _estimate_epsilon(self, majority: np.ndarray) -> float:
        if len(majority) < 2:
            return 1.0
        d = pdist(majority)
        std_d = np.std(d)
        mean_d = np.mean(d)
        eps = 0.5 * std_d * mean_d
        if np.isnan(eps) or eps <= 1e-8:
            eps = 1.0 / (np.var(majority) + 1e-4)
        return float(eps)

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            raise ValueError(f"Binary classification required; got {len(classes)} classes.")

        min_label = classes[int(np.argmin(counts))]
        maj_label = classes[int(np.argmax(counts))]

        minority = X_arr[y_arr == min_label]
        majority = X_arr[y_arr == maj_label]

        n_to_add = len(majority) - len(minority)
        if n_to_add <= 0 or len(minority) == 0:
            return X_arr.copy(), y_arr.copy()

        rng = np.random.default_rng(self.random_state)
        eps = self.epsilon if self.epsilon is not None else self._estimate_epsilon(majority)

        min_std = np.std(minority, axis=0)
        # Avoid zero standard deviation
        min_std = np.where(min_std == 0, 1e-3, min_std)
        stds = self.tau * min_std

        synthetics = np.empty((n_to_add, X_arr.shape[1]), dtype=float)
        accepted = 0
        max_trials = n_to_add * 25
        trials = 0

        while accepted < n_to_add and trials < max_trials:
            trials += 1
            j = int(rng.integers(0, len(minority)))
            seed_point = minority[j]
            score_cur = self._score(seed_point, majority, eps)

            for _ in range(self.steps):
                step = seed_point + rng.normal(0, stds, size=X_arr.shape[1])
                step_score = self._score(step, majority, eps)
                if step_score <= score_cur:
                    synthetics[accepted] = step
                    accepted += 1
                    break

        if accepted < n_to_add:
            # If not enough candidates met the strict RBF majority threshold,
            # fill remainder with mild Gaussian perturbation of minority
            while accepted < n_to_add:
                j = int(rng.integers(0, len(minority)))
                synthetics[accepted] = minority[j] + rng.normal(0, stds * 0.5, size=X_arr.shape[1])
                accepted += 1

        X_res = np.vstack([X_arr, synthetics])
        y_res = np.concatenate([y_arr, np.full(n_to_add, min_label, dtype=y_arr.dtype)])
        return X_res, y_res


AVAILABLE_RESAMPLERS = {
    "SMOTEFUNA": SMOTEFUNA,
    "SMOTE": SMOTEBaseline,
    "ADASYN": ADASYNBaseline,
    "SWIM": SWIMBaseline,
    "Original": OriginalDataBaseline,
}


def get_resampler(
    name: str,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Factory helper to instantiate a resampler by name with standard parameters."""
    normalized = name.strip()
    if normalized not in AVAILABLE_RESAMPLERS:
        valid = list(AVAILABLE_RESAMPLERS.keys())
        raise ValueError(f"Unknown resampler '{name}'. Choose from {valid}")

    cls = AVAILABLE_RESAMPLERS[normalized]
    if normalized == "SMOTEFUNA":
        return cls(random_state=random_state, strict=False, **kwargs)
    return cls(random_state=random_state, **kwargs)
