"""The 33 datasets from Table 1.

--real pulls OpenML copies (cached under data_cache/).
Without --real we fake them with make_classification so it still runs offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"


@dataclass(frozen=True)
class DatasetMeta:
    dataset_id: str
    name: str
    n_instances: int
    n_classes: int
    ir: float
    n_features: int
    openml_name: Optional[str] = None
    openml_id: Optional[int] = None


# paper Table 1. IR=605 for Kc1 is what they printed, looks off but we kept it.
TABLE_1_DATASETS: Dict[str, DatasetMeta] = {
    "D1": DatasetMeta("D1", "Ar1", 121, 2, 13.4, 30, openml_name="ar1", openml_id=1240),
    "D2": DatasetMeta("D2", "Ar3", 63, 2, 7.875, 30, openml_name="ar3", openml_id=1239),
    "D3": DatasetMeta("D3", "Ar4", 107, 2, 5.45, 30, openml_name="ar4", openml_id=1241),
    "D4": DatasetMeta("D4", "Ar5", 36, 2, 4.5, 30, openml_name="ar5", openml_id=1242),
    "D5": DatasetMeta("D5", "Ar6", 101, 2, 6.7, 30, openml_name="ar6", openml_id=1243),
    "D6": DatasetMeta("D6", "Kc1", 2109, 2, 605.0, 22, openml_name="kc1", openml_id=1067),
    "D7": DatasetMeta("D7", "Kc2", 522, 2, 4.9, 22, openml_name="kc2", openml_id=1065),
    "D8": DatasetMeta("D8", "Pc1", 1109, 2, 14.4, 22, openml_name="pc1", openml_id=1068),
    "D9": DatasetMeta("D9", "Pc3", 1563, 2, 9.8, 38, openml_name="pc3", openml_id=1050),
    "D10": DatasetMeta("D10", "Pc4", 1458, 2, 8.2, 38, openml_name="pc4", openml_id=1049),
    "D11": DatasetMeta("D11", "Australian", 690, 2, 1.25, 42, openml_name="australian", openml_id=40981),
    "D12": DatasetMeta("D12", "Bank", 1372, 2, 1.25, 4, openml_name="banknote-authentication", openml_id=1462),
    "D13": DatasetMeta("D13", "Heart", 270, 2, 1.25, 25, openml_name="heart-statlog", openml_id=53),
    "D14": DatasetMeta("D14", "Oil-Spill", 937, 2, 21.85, 49, openml_name="oil_spill", openml_id=313),
    "D15": DatasetMeta("D15", "Phoneme", 5404, 2, 2.41, 5, openml_name="phoneme", openml_id=1489),
    "D16": DatasetMeta("D16", "Abalone19", 4174, 2, 129.44, 8, openml_name="abalone", openml_id=183),
    "D17": DatasetMeta("D17", "Abalone9-18", 731, 2, 16.4, 8, openml_name="abalone", openml_id=183),
    "D18": DatasetMeta("D18", "Page-blocks0", 5472, 2, 8.79, 11, openml_name="page-blocks", openml_id=30),
    "D19": DatasetMeta("D19", "Pima", 768, 2, 1.87, 9, openml_name="diabetes", openml_id=37),
    "D20": DatasetMeta("D20", "Segment0", 2308, 2, 6.2, 20, openml_name="segment", openml_id=36),
    "D21": DatasetMeta("D21", "Shuttle-c0", 1829, 2, 13.87, 10, openml_name="shuttle", openml_id=40685),
    "D22": DatasetMeta("D22", "Vehicle0", 846, 2, 3.25, 19, openml_name="vehicle", openml_id=54),
    "D23": DatasetMeta("D23", "Vehicle1", 846, 2, 2.9, 19, openml_name="vehicle", openml_id=54),
    "D24": DatasetMeta("D24", "Vehicle2", 846, 2, 2.88, 19, openml_name="vehicle", openml_id=54),
    "D25": DatasetMeta("D25", "Vehicle3", 846, 2, 2.99, 19, openml_name="vehicle", openml_id=54),
    "D26": DatasetMeta("D26", "Vowel0", 988, 2, 9.98, 14, openml_name="vowel", openml_id=307),
    "D27": DatasetMeta("D27", "Wisconsin", 683, 2, 1.86, 10, openml_name="breast-w", openml_id=15),
    "D28": DatasetMeta("D28", "Yeast-1-2-8-9", 947, 2, 30.57, 9, openml_name="yeast", openml_id=181),
    "D29": DatasetMeta("D29", "Yeast-1-4-5-8", 693, 2, 22.1, 9, openml_name="yeast", openml_id=181),
    "D30": DatasetMeta("D30", "Yeast1", 1484, 2, 2.46, 9, openml_name="yeast", openml_id=181),
    "D31": DatasetMeta("D31", "Yeast3", 1484, 2, 8.1, 9, openml_name="yeast", openml_id=181),
    "D32": DatasetMeta("D32", "Yeast4", 1484, 2, 28.1, 9, openml_name="yeast", openml_id=181),
    "D33": DatasetMeta("D33", "Yeast5", 1484, 2, 32.73, 9, openml_name="yeast", openml_id=181),
}

# some KEEL sets aren't binary — these are the recipes we used
_EXPLICIT_BINARIZE = {
    "D16": {"mode": "vs_rest", "minority": {"19", "19.0"}},
    "D17": {"mode": "ovo", "minority": {"9", "9.0"}, "majority": {"18", "18.0"}},
    "D28": {
        "mode": "subset",
        "minority": {"VAC", "vac"},
        "majority": {"NUC", "CYT", "POX", "ERL", "nuc", "cyt", "pox", "erl"},
    },
    "D29": {
        "mode": "subset",
        "minority": {"VAC", "vac"},
        "majority": {"NUC", "ME3", "ME2", "POX", "nuc", "me3", "me2", "pox"},
    },
}

NAME_TO_ID: Dict[str, str] = {
    meta.name.lower(): meta.dataset_id for meta in TABLE_1_DATASETS.values()
}
for d_id in TABLE_1_DATASETS:
    NAME_TO_ID[d_id.lower()] = d_id


def get_table1_metadata() -> pd.DataFrame:
    records = []
    for d_id, m in TABLE_1_DATASETS.items():
        records.append({
            "#": d_id,
            "Name": m.name,
            "nInes": m.n_instances,
            "nCl": m.n_classes,
            "IR": m.ir,
            "nAt": m.n_features,
        })
    return pd.DataFrame(records)


def _norm_label(val) -> str:
    text = str(val).strip()
    if text.endswith(".0"):
        try:
            if float(text) == int(float(text)):
                text = str(int(float(text)))
        except ValueError:
            pass
    return text


def _labels_as_str(y: np.ndarray) -> np.ndarray:
    return np.array([_norm_label(v) for v in np.asarray(y)], dtype=object)


def _ensure_minority_is_one(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    if y.sum() > len(y) - y.sum():
        y = 1 - y
    return y


def _ir(y: np.ndarray) -> float:
    n_min = int(np.sum(y == 1))
    n_maj = int(np.sum(y == 0))
    if n_min == 0:
        return float("inf")
    return n_maj / n_min


def _binarize_vs_rest(y_str: np.ndarray, minority_labels: Optional[set] = None, target_ir: Optional[float] = None) -> np.ndarray:
    if minority_labels:
        wanted = {_norm_label(v) for v in minority_labels}
        return _ensure_minority_is_one(np.array([lab in wanted for lab in y_str], dtype=int))

    unique = np.unique(y_str)
    if len(unique) == 2:
        return _ensure_minority_is_one((y_str == unique[np.argmin([np.sum(y_str == u) for u in unique])]).astype(int))

    best_y = None
    best_err = float("inf")
    target = 1.0 if target_ir is None else float(target_ir)
    for lab in unique:
        cand = (y_str == lab).astype(int)
        if cand.sum() < 2 or cand.sum() >= len(cand) - 1:
            continue
        cand = _ensure_minority_is_one(cand)
        err = abs(_ir(cand) - target)
        if err < best_err:
            best_err = err
            best_y = cand
    if best_y is None:
        raise RuntimeError("Could not binarize labels with vs-rest.")
    return best_y


def _binarize_ovo(y_str: np.ndarray, target_ir: float, target_n: int, minority_labels=None, majority_labels=None) -> Tuple[np.ndarray, np.ndarray]:
    """Keep two classes only (one-vs-one)."""
    if minority_labels and majority_labels:
        min_set = {_norm_label(v) for v in minority_labels}
        maj_set = {_norm_label(v) for v in majority_labels}
        mask = np.array([lab in min_set or lab in maj_set for lab in y_str])
        y_bin = np.array([1 if lab in min_set else 0 for lab in y_str[mask]], dtype=int)
        return mask, _ensure_minority_is_one(y_bin)

    unique = np.unique(y_str)
    best = None
    best_score = float("inf")
    for a in unique:
        for b in unique:
            if a == b:
                continue
            mask = (y_str == a) | (y_str == b)
            y_bin = _ensure_minority_is_one((y_str[mask] == a).astype(int))
            n = int(mask.sum())
            if np.sum(y_bin == 1) < 2:
                continue
            score = abs(_ir(y_bin) - target_ir) + 0.001 * abs(n - target_n)
            if score < best_score:
                best_score = score
                best = (mask, y_bin)
    if best is None:
        raise RuntimeError("Could not binarize labels with one-vs-one.")
    return best


def _binarize_subset(y_str: np.ndarray, minority_labels: set, majority_labels: set) -> Tuple[np.ndarray, np.ndarray]:
    min_set = {_norm_label(v) for v in minority_labels}
    maj_set = {_norm_label(v) for v in majority_labels}
    mask = np.array([lab in min_set or lab in maj_set for lab in y_str])
    y_bin = np.array([1 if lab in min_set else 0 for lab in y_str[mask]], dtype=int)
    return mask, _ensure_minority_is_one(y_bin)


def _frame_to_xy(fetched) -> Tuple[np.ndarray, np.ndarray]:
    # OpenML sometimes gives strings / nans, just make it numeric
    X_df = pd.DataFrame(fetched.data).copy()
    y_raw = np.asarray(fetched.target)

    for col in list(X_df.columns):
        if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
            codes, _ = pd.factorize(X_df[col], sort=True)
            X_df[col] = codes
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X = np.nan_to_num(X_df.to_numpy(dtype=float), nan=0.0)
    return X, y_raw


def _apply_keel_binarize(meta: DatasetMeta, X: np.ndarray, y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_str = _labels_as_str(y_raw)
    spec = _EXPLICIT_BINARIZE.get(meta.dataset_id)

    if spec and spec["mode"] == "vs_rest":
        y = _binarize_vs_rest(y_str, minority_labels=spec.get("minority"), target_ir=meta.ir)
        return X, y
    if spec and spec["mode"] == "ovo":
        mask, y = _binarize_ovo(
            y_str,
            target_ir=meta.ir,
            target_n=meta.n_instances,
            minority_labels=spec.get("minority"),
            majority_labels=spec.get("majority"),
        )
        return X[mask], y
    if spec and spec["mode"] == "subset":
        mask, y = _binarize_subset(y_str, spec["minority"], spec["majority"])
        return X[mask], y

    unique = np.unique(y_str)
    if len(unique) <= 2:
        y = _binarize_vs_rest(y_str, target_ir=meta.ir)
        return X, y

    # shuttle etc: pick the split whose IR is closest to Table 1
    if "shuttle" in meta.name.lower() or meta.dataset_id == "D21":
        mask, y = _binarize_ovo(y_str, target_ir=meta.ir, target_n=meta.n_instances)
        return X[mask], y

    y = _binarize_vs_rest(y_str, target_ir=meta.ir)
    return X, y


def _fetch_openml_xy(meta: DatasetMeta) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    last_err = None
    fetched = None
    if meta.openml_id is not None:
        try:
            fetched = fetch_openml(data_id=int(meta.openml_id), as_frame=True, parser="auto")
        except Exception as exc:
            last_err = exc
    if fetched is None and meta.openml_name:
        try:
            fetched = fetch_openml(name=meta.openml_name, as_frame=True, parser="auto")
        except Exception as exc:
            last_err = exc
    if fetched is None:
        raise RuntimeError(f"OpenML fetch failed for {meta.dataset_id} ({meta.name}): {last_err}")

    X, y_raw = _frame_to_xy(fetched)
    return _apply_keel_binarize(meta, X, y_raw)


def _real_cache_path(meta: DatasetMeta) -> Path:
    return CACHE_DIR / f"real_{meta.dataset_id}_{meta.name.lower()}.npz"


def _load_real_wisconsin() -> Tuple[np.ndarray, np.ndarray]:
    # sklearn version, not the KEEL 683-row one. only used if OpenML is off.
    data = load_breast_cancer()
    X = data.data
    y = data.target
    classes, counts = np.unique(y, return_counts=True)
    min_cls = classes[np.argmin(counts)]
    y_bin = (y == min_cls).astype(int)
    return X, y_bin


def _load_real_banknote() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # leftover cache from earlier, ignore if missing
    cache_file = CACHE_DIR / "banknote.npz"
    if cache_file.exists():
        loaded = np.load(cache_file)
        return loaded["X"], loaded["y"]
    return None


def generate_benchmark_from_meta(
    meta: DatasetMeta, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Fake data with the same size / IR as Table 1. not the real KEEL tables."""
    n_samples = meta.n_instances
    ir = meta.ir
    n_features = meta.n_features

    n_min = max(2, int(round(n_samples / (ir + 1.0))))
    n_maj = n_samples - n_min
    weights = [n_maj / n_samples, n_min / n_samples]

    n_informative = min(n_features, max(2, int(n_features * 0.7)))
    n_redundant = min(n_features - n_informative, max(0, int(n_features * 0.15)))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1 if n_samples < 500 else 2,
        weights=weights,
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )

    classes, counts = np.unique(y, return_counts=True)
    min_label = classes[np.argmin(counts)]
    y_binary = np.where(y == min_label, 1, 0)
    return X, y_binary


def load_benchmark_dataset(
    name_or_id: str,
    allow_online: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, DatasetMeta]:
    """Load by id ('D19') or name ('Pima'). minority is always 1."""
    key = name_or_id.strip().lower()
    if key not in NAME_TO_ID:
        raise ValueError(
            f"Dataset '{name_or_id}' not found in Table 1 benchmarks. "
            f"Valid options include: {list(TABLE_1_DATASETS.keys())}"
        )

    meta = TABLE_1_DATASETS[NAME_TO_ID[key]]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if allow_online:
        real_cache = _real_cache_path(meta)
        if real_cache.exists():
            data = np.load(real_cache, allow_pickle=False)
            if verbose:
                print(f"  {meta.dataset_id} {meta.name}: REAL (cache) n={len(data['y'])} feats={data['X'].shape[1]}")
            return data["X"], data["y"], meta
        try:
            X, y = _fetch_openml_xy(meta)
            np.savez_compressed(real_cache, X=X, y=y)
            if verbose:
                n_min = int(np.sum(y == 1))
                n_maj = int(np.sum(y == 0))
                print(
                    f"  {meta.dataset_id} {meta.name}: REAL (OpenML) "
                    f"n={len(y)} feats={X.shape[1]} IR={n_maj / max(n_min, 1):.2f} "
                    f"(paper n={meta.n_instances}, IR={meta.ir})"
                )
            return X, y, meta
        except Exception as exc:
            if verbose:
                print(f"  {meta.dataset_id} {meta.name}: OpenML failed ({exc}); using GENERATED stand-in")

    if not allow_online and meta.name.lower() == "wisconsin":
        X, y = _load_real_wisconsin()
        return X, y, meta

    cache_path = CACHE_DIR / f"{meta.dataset_id}_{meta.name.lower()}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["X"], data["y"], meta

    X, y = generate_benchmark_from_meta(meta, seed=seed)
    if verbose:
        print(f"  {meta.dataset_id} {meta.name}: GENERATED stand-in n={len(y)} feats={X.shape[1]}")
    return X, y, meta


def load_representative_suite(
    seed: int = 42,
    allow_online: bool = False,
    verbose: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, DatasetMeta]]:
    """Small mix of IRs so --quick doesn't take forever."""
    selected_ids = ["D10", "D12", "D16", "D17", "D19", "D27"]
    suite = {}
    for d_id in selected_ids:
        X, y, meta = load_benchmark_dataset(
            d_id, seed=seed, allow_online=allow_online, verbose=verbose
        )
        suite[f"{meta.dataset_id}_{meta.name}"] = (X, y, meta)
    return suite


def load_all_benchmark_datasets(
    dataset_ids: Optional[List[str]] = None,
    seed: int = 42,
    allow_online: bool = False,
    verbose: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, DatasetMeta]]:
    """All 33, or whatever ids you pass."""
    target_ids = dataset_ids if dataset_ids is not None else list(TABLE_1_DATASETS.keys())
    all_data = {}
    for d_id in target_ids:
        X, y, meta = load_benchmark_dataset(
            d_id, seed=seed, allow_online=allow_online, verbose=verbose
        )
        all_data[f"{meta.dataset_id}_{meta.name}"] = (X, y, meta)
    return all_data
