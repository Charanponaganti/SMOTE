"""SMOTEFUNA — oversample the minority class using furthest neighbours."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, list]


class SMOTEFUNA:
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
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        rng = np.random.default_rng(self.random_state)
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError("SMOTEFUNA only handles two classes.")

        # smaller class = minority
        min_label = labels[int(np.argmin(counts))]
        maj_label = labels[int(np.argmax(counts))]
        if counts[0] == counts[1]:
            return self._maybe_shuffle(X.copy(), y.copy(), rng)

        minority = X[y == min_label]
        majority = X[y == maj_label]
        if len(minority) < 2:
            raise ValueError(
                "SMOTEFUNA needs at least 2 minority samples to form a furthest pair."
            )

        n_new = self._how_many_to_add(len(minority), len(majority))
        if n_new <= 0:
            return self._maybe_shuffle(X.copy(), y.copy(), rng)

        extra = self._make_new_points(minority, majority, n_new, rng)
        X_out = np.vstack([X, extra])
        y_out = np.concatenate(
            [y, np.full(len(extra), min_label, dtype=y.dtype)]
        )
        return self._maybe_shuffle(X_out, y_out, rng)

    def _how_many_to_add(self, n_min: int, n_maj: int) -> int:
        # "auto" = add until minority matches majority
        if self.sampling_strategy == "auto":
            return n_maj - n_min
        if isinstance(self.sampling_strategy, (int, float)):
            ratio = float(self.sampling_strategy)
            if not 0.0 < ratio <= 1.0:
                raise ValueError("sampling_strategy float must be in (0, 1].")
            return max(0, int(np.ceil(ratio * n_maj)) - n_min)
        raise ValueError("sampling_strategy must be 'auto' or a float in (0, 1].")

    def _maybe_shuffle(self, X, y, rng):
        if not self.shuffle:
            return X, y
        order = rng.permutation(len(X))
        return X[order], y[order]

    def _make_new_points(self, minority, majority, n_new, rng):
        """This is the actual SMOTEFUNA step"""
        extra = np.empty((n_new, minority.shape[1]))
        got = 0
        tries = 0
        max_tries = max(n_new * self.max_attempts_factor, n_new)

        while got < n_new and tries < max_tries:
            tries += 1

            # 1. pick a random minority point S1
            i = int(rng.integers(0, len(minority)))
            s1 = minority[i]

            # 2. find its furthest minority neighbour S2 (manhattan distance)
            dist = np.abs(minority - s1).sum(axis=1)
            dist[i] = -1  # ignore itself
            s2 = minority[int(np.argmax(dist))]

            # 3. draw a new point inside the box between S1 and S2
            low = np.minimum(s1, s2)
            high = np.maximum(s1, s2)
            s_new = rng.uniform(low, high)

            # 4. keep it only if it is closer to minority than to majority
            theta = np.abs(minority - s_new).sum(axis=1).min()
            beta = np.abs(majority - s_new).sum(axis=1).min()
            if theta <= beta:
                extra[got] = s_new
                got += 1

        if got < n_new:
            if self.strict:
                raise RuntimeError(
                    f"Could only accept {got}/{n_new} synthetic samples "
                    f"after {tries} attempts."
                )
            import warnings

            warnings.warn(
                f"SMOTEFUNA generated {got}/{n_new} synthetic samples after "
                f"{tries} attempts due to high class overlap. Returning available samples.",
                UserWarning,
            )
            return extra[:got]
        return extra

