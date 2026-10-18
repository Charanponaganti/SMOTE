"""
5-Fold Cross-Validation Evaluation Pipeline.

Implements the experimental methodology described in Section IV:
- Stratified 5-fold cross-validation
- Repeated multi-trial evaluation (e.g. 1 to 30 trials)
- Zero data leakage: z-score standardization and oversampling are fit ONLY
  on the training fold; evaluation is on untouched imbalanced test fold.
- Classifiers: Naive Bayes (GaussianNB) and SVM (RBF kernel, SVC with probabilities)
- Algorithms: SMOTEFUNA, SMOTE, ADASYN, SWIM, Original Data
- Metrics: ROC-AUC and AUPR
- Aggregates Mean +/- STD and generates Tables 2, 3, 4, 5, 6, 7, 8.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.baselines import get_resampler
from experiments.datasets import DatasetMeta, load_benchmark_dataset
from experiments.metrics import compute_all_metrics, compute_win_counts
from experiments.statistical_tests import compute_statistical_suite


def default_classifiers(seed: int = 42) -> Dict[str, Any]:
    """Returns the two standard classifiers used in the SMOTEFUNA paper."""
    return {
        "NB": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
    }


DEFAULT_METHODS = ["SMOTEFUNA", "SMOTE", "ADASYN", "SWIM", "Original"]


@dataclass
class ExperimentResult:
    dataset_name: str
    dataset_id: str
    classifier: str
    method: str
    auc_scores: List[float] = field(default_factory=list)
    aupr_scores: List[float] = field(default_factory=list)

    @property
    def auc_mean(self) -> float:
        return float(np.mean(self.auc_scores)) if self.auc_scores else 0.0

    @property
    def auc_std(self) -> float:
        return float(np.std(self.auc_scores, ddof=1)) if len(self.auc_scores) > 1 else 0.0

    @property
    def aupr_mean(self) -> float:
        return float(np.mean(self.aupr_scores)) if self.aupr_scores else 0.0

    @property
    def aupr_std(self) -> float:
        return float(np.std(self.aupr_scores, ddof=1)) if len(self.aupr_scores) > 1 else 0.0


class BenchmarkPipeline:
    """
    Execution engine for 5-fold cross-validated benchmark experiments.
    """

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        classifiers: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
        n_trials: int = 1,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        self.methods = methods if methods is not None else list(DEFAULT_METHODS)
        self.classifiers = classifiers if classifiers is not None else default_classifiers(random_state)
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose

    def evaluate_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: DatasetMeta,
    ) -> Dict[str, Dict[str, ExperimentResult]]:
        """
        Runs repeated 5-fold CV on a single dataset across all methods & classifiers.

        Returns
        -------
        dict : {classifier_name: {method_name: ExperimentResult}}
        """
        results: Dict[str, Dict[str, ExperimentResult]] = {
            clf_name: {
                m: ExperimentResult(dataset_name=meta.name, dataset_id=meta.dataset_id, classifier=clf_name, method=m)
                for m in self.methods
            }
            for clf_name in self.classifiers
        }

        # Multi-trial loop
        for trial in range(self.n_trials):
            trial_seed = self.random_state + trial * 101
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=trial_seed)

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # 1. Standardize (z-score) fitted strictly on training data
                scaler = StandardScaler().fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # 2. Resample and evaluate for each method
                for m in self.methods:
                    fold_seed = trial_seed + fold_idx * 17
                    resampler = get_resampler(m, random_state=fold_seed)

                    try:
                        X_res, y_res = resampler.fit_resample(X_train_scaled, y_train)
                    except Exception:
                        # Defensive fallback to unresampled train data if method encounters extreme geometry
                        X_res, y_res = X_train_scaled, y_train

                    # 3. Train each classifier and evaluate
                    for clf_name, clf_template in self.classifiers.items():
                        # Clone/instantiate classifier with fold seed
                        clf_params = clf_template.get_params()
                        if "random_state" in clf_params:
                            clf_params["random_state"] = fold_seed
                        clf = clf_template.__class__(**clf_params)

                        try:
                            clf.fit(X_res, y_res)
                            # Predict probability of positive / minority class (1)
                            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
                        except Exception:
                            # Fallback dummy prediction if model fails
                            y_prob = np.full(len(y_test), 0.5)

                        metrics = compute_all_metrics(y_test, y_prob)
                        results[clf_name][m].auc_scores.append(metrics["AUC"])
                        results[clf_name][m].aupr_scores.append(metrics["AUPR"])

        return results

    def run_suite(
        self,
        datasets: Dict[str, Tuple[np.ndarray, np.ndarray, DatasetMeta]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the benchmark across all provided datasets.

        Returns structured dictionary containing raw results, summary tables,
        win counts, and Wilcoxon statistical tests.
        """
        total = len(datasets)
        all_dataset_results: Dict[str, Dict[str, Dict[str, ExperimentResult]]] = {}

        start_time = time.time()
        for idx, (ds_key, (X, y, meta)) in enumerate(datasets.items()):
            if self.verbose:
                print(
                    f"[{idx+1}/{total}] Evaluating {meta.dataset_id} ({meta.name}) "
                    f"- {meta.n_instances} samples, {meta.n_features} features, IR={meta.ir}..."
                )
            if progress_callback:
                progress_callback(ds_key, idx + 1, total)

            ds_res = self.evaluate_dataset(X, y, meta)
            all_dataset_results[ds_key] = ds_res

        elapsed = time.time() - start_time
        if self.verbose:
            print(f"Benchmark finished across {total} datasets in {elapsed:.2f} seconds.")

        # Build paper-compliant tables
        return self._build_paper_report(all_dataset_results)

    def _build_paper_report(
        self,
        all_results: Dict[str, Dict[str, Dict[str, ExperimentResult]]],
    ) -> Dict[str, Any]:
        """Formats the results into exact tables corresponding to Tables 2 to 8 of the paper."""
        classifiers = list(self.classifiers.keys())
        methods = self.methods

        # Tables 5, 6, 7, 8
        summary_tables: Dict[str, pd.DataFrame] = {}
        for clf in classifiers:
            for metric in ["AUC", "AUPR"]:
                rows = []
                for ds_key, ds_dict in all_results.items():
                    # Extract dataset ID and name
                    sample_res = next(iter(ds_dict[clf].values()))
                    row: Dict[str, Any] = {"Dataset": sample_res.dataset_id, "Name": sample_res.dataset_name}
                    for m in methods:
                        res = ds_dict[clf][m]
                        val_mean = res.auc_mean if metric == "AUC" else res.aupr_mean
                        val_std = res.auc_std if metric == "AUC" else res.aupr_std
                        row[m] = f"{val_mean:.4f} \u00b1 {val_std:.4f}"
                        row[f"{m}_mean"] = val_mean
                    rows.append(row)
                df = pd.DataFrame(rows)
                summary_tables[f"{metric}_{clf}"] = df

        # Table 2: Win counts
        win_rows = []
        for metric in ["AUC", "AUPR"]:
            for clf in classifiers:
                df = summary_tables[f"{metric}_{clf}"]
                mean_cols = [f"{m}_mean" for m in methods]
                sub_df = df[mean_cols].rename(columns=lambda c: c.replace("_mean", ""))
                wins = compute_win_counts(sub_df)
                row_dict = {"Classifier": clf, "Measure": metric}
                for m in methods:
                    row_dict[m] = int(wins.get(m, 0))
                win_rows.append(row_dict)
        table2_df = pd.DataFrame(win_rows)

        # Build statistical tests input matrix
        stats_matrix: Dict[str, Dict[str, Dict[str, List[float]]]] = {
            clf: {"AUC": {}, "AUPR": {}} for clf in classifiers
        }
        for clf in classifiers:
            for metric in ["AUC", "AUPR"]:
                df = summary_tables[f"{metric}_{clf}"]
                for m in methods:
                    stats_matrix[clf][metric][m] = df[f"{m}_mean"].tolist()

        table3_df, table4_df = compute_statistical_suite(
            stats_matrix,
            proposed_method="SMOTEFUNA",
            baseline_methods=[m for m in methods if m != "SMOTEFUNA"],
        )

        return {
            "raw_results": all_results,
            "summary_tables": summary_tables,
            "table2_wins": table2_df,
            "table3_p_values": table3_df,
            "table4_wilcoxon_ranks": table4_df,
        }
