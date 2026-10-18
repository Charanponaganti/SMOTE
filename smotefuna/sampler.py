"""
SMOTEFUNA oversampler.

Based on Tarawneh et al., "SMOTEFUNA: Synthetic Minority Over-Sampling
Technique Based on Furthest Neighbour Algorithm", IEEE Access, 2020.

Algorithm summary
-----------------
1. Split minority / majority.
2. Need Rate = |majority| - |minority| new minority samples.
3. Pick random seed S1; find furthest minority neighbour S2 (Manhattan).
4. Draw each feature of S_new uniformly in [min(S1_j, S2_j), max(...)].
5. Accept only if nearer to minority than majority (theta <= beta); else retry.
6. Concatenate and optionally shuffle.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, list]


class SMOTEFUNA:
    """
    Parameter-free furthest-neighbour synthetic oversampler.

    Parameters
    ----------
    sampling_strategy : float or str, default="auto"
        - "auto": balance minority up to majority count.
        - float in (0, 1]: desired minority/majority ratio after sampling.
    shuffle : bool, default=True
        Shuffle the balanced dataset before returning.
    max_attempts_factor : int, default=50
        Max candidate draws per accepted sample = factor * Rate.
        Prevents infinite loops on pathological layouts.
    random_state : int or None, default=None
        Seed for reproducibility.
    """

    def __init__(
        self,
        sampling_strategy: Union[str, float] = "auto",
        shuffle: bool = True,
        max_attempts_factor: int = 50,
        random_state: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.shuffle = shuffle
        self.max_attempts_factor = max_attempts_factor
        self.random_state = random_state
        self.strict = strict

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample the minority class and return a (more) balanced dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary labels. The minority class is the less frequent label.

        Returns
        -------
        X_resampled, y_resampled
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError(
                f"SMOTEFUNA supports binary classification; got {len(classes)} classes."
            )

        # Minority = smaller class; tie-break: first unique in sorted order
        minority_label = classes[int(np.argmin(counts))]
        majority_label = classes[int(np.argmax(counts))]
        if counts[0] == counts[1]:
            # Already balanced — return copy
            X_out, y_out = X.copy(), y.copy()
            if self.shuffle:
                idx = rng.permutation(len(X_out))
                return X_out[idx], y_out[idx]
            return X_out, y_out

        minority_mask = y == minority_label
        majority_mask = y == majority_label
        minority = X[minority_mask]
        majority = X[majority_mask]

        n_min = len(minority)
        n_maj = len(majority)
        if n_min < 2:
            raise ValueError(
                "SMOTEFUNA needs at least 2 minority samples to form a furthest pair."
            )

        target_min = self._target_minority_count(n_min, n_maj)
        rate = target_min - n_min
        if rate <= 0:
            X_out, y_out = X.copy(), y.copy()
            if self.shuffle:
                idx = rng.permutation(len(X_out))
                return X_out[idx], y_out[idx]
            return X_out, y_out

        synthetics = self._generate_synthetics(
            minority=minority,
            majority=majority,
            rate=rate,
            rng=rng,
        )

        X_resampled = np.vstack([X, synthetics])
        y_resampled = np.concatenate(
            [y, np.full(len(synthetics), minority_label, dtype=y.dtype)]
        )

        if self.shuffle:
            idx = rng.permutation(len(X_resampled))
            X_resampled = X_resampled[idx]
            y_resampled = y_resampled[idx]

        return X_resampled, y_resampled

    def _target_minority_count(self, n_min: int, n_maj: int) -> int:
        strategy = self.sampling_strategy
        if strategy == "auto":
            return n_maj
        if isinstance(strategy, (int, float)):
            ratio = float(strategy)
            if not 0.0 < ratio <= 1.0:
                raise ValueError("sampling_strategy float must be in (0, 1].")
            return max(n_min, int(np.ceil(ratio * n_maj)))
        raise ValueError(
            "sampling_strategy must be 'auto' or a float in (0, 1]."
        )

    def _generate_synthetics(
        self,
        minority: np.ndarray,
        majority: np.ndarray,
        rate: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_features = minority.shape[1]
        synthetics = np.empty((rate, n_features), dtype=float)
        accepted = 0
        attempts = 0
        max_attempts = max(rate * self.max_attempts_factor, rate)

        while accepted < rate and attempts < max_attempts:
            attempts += 1
            # Random minority seed
            i1 = int(rng.integers(0, len(minority)))
            s1 = minority[i1]

            # Furthest minority neighbour (exclude self)
            dists = np.sum(np.abs(minority - s1), axis=1)
            dists[i1] = -1.0  # ensure self is never chosen as furthest
            i2 = int(np.argmax(dists))
            s2 = minority[i2]

            # Cuboidal random sample between s1 and s2
            lo = np.minimum(s1, s2)
            hi = np.maximum(s1, s2)
            # Where lo == hi, Random(lo, hi) is a single point
            s_new = rng.uniform(lo, hi)

            # Safety check: nearer to minority than majority?
            theta = np.min(np.sum(np.abs(minority - s_new), axis=1))
            beta = np.min(np.sum(np.abs(majority - s_new), axis=1))
            if theta <= beta:
                synthetics[accepted] = s_new
                accepted += 1

        if accepted < rate:
            if self.strict:
                raise RuntimeError(
                    f"Could only accept {accepted}/{rate} synthetic samples "
                    f"after {attempts} attempts. The minority/majority geometry "
                    f"may be heavily overlapping; try more minority seeds or "
                    f"increase max_attempts_factor."
                )
            import warnings

            warnings.warn(
                f"SMOTEFUNA generated {accepted}/{rate} synthetic samples after "
                f"{attempts} attempts due to high class overlap. Returning available samples.",
                UserWarning,
            )
            return synthetics[:accepted]
        return synthetics
