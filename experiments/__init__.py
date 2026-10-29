"""Experiment code: datasets, baselines, 5-fold CV, Wilcoxon."""

from experiments.baselines import (
    AVAILABLE_RESAMPLERS,
    ADASYNBaseline,
    OriginalDataBaseline,
    SMOTEBaseline,
    SWIMBaseline,
    get_resampler,
)
from experiments.datasets import (
    TABLE_1_DATASETS,
    DatasetMeta,
    get_table1_metadata,
    load_all_benchmark_datasets,
    load_benchmark_dataset,
    load_representative_suite,
)
from experiments.metrics import (
    compute_all_metrics,
    compute_aupr,
    compute_roc_auc,
    compute_win_counts,
)
from experiments.pipeline import BenchmarkPipeline, ExperimentResult
from experiments.statistical_tests import (
    WilcoxonComparisonResult,
    compute_statistical_suite,
    compute_wilcoxon_signed_rank,
)
from experiments.visualize import (
    plot_pca_projections,
    plot_roc_and_pr_grid,
)

__all__ = [
    "AVAILABLE_RESAMPLERS",
    "ADASYNBaseline",
    "OriginalDataBaseline",
    "SMOTEBaseline",
    "SWIMBaseline",
    "get_resampler",
    "TABLE_1_DATASETS",
    "DatasetMeta",
    "get_table1_metadata",
    "load_all_benchmark_datasets",
    "load_benchmark_dataset",
    "load_representative_suite",
    "compute_all_metrics",
    "compute_aupr",
    "compute_roc_auc",
    "compute_win_counts",
    "BenchmarkPipeline",
    "ExperimentResult",
    "WilcoxonComparisonResult",
    "compute_statistical_suite",
    "compute_wilcoxon_signed_rank",
    "plot_pca_projections",
    "plot_roc_and_pr_grid",
]

