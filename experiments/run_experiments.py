"""
Main Benchmark Runner for SMOTEFUNA Experimental Replication.

CLI entry-point to execute the experimental evaluations:
- 5-fold cross-validation on real and Table 1 benchmark datasets
- Compares SMOTEFUNA vs SMOTE, ADASYN, SWIM, and Original Data
- Evaluates on Naive Bayes and Support Vector Machine (SVM)
- Computes ROC-AUC, AUPR, win counts (Table 2), and Wilcoxon signed-rank tests (Tables 3 & 4)
- Generates publication plots (Figures 8, 9, 10) and markdown/CSV tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.datasets import (
    load_all_benchmark_datasets,
    load_benchmark_dataset,
    load_representative_suite,
    TABLE_1_DATASETS,
)
from experiments.pipeline import BenchmarkPipeline
from experiments.visualize import plot_pca_projections, plot_roc_and_pr_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SMOTEFUNA research paper benchmark evaluations."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quickly on a representative subset of benchmark datasets.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark across all 33 datasets from Table 1.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset IDs or names (e.g. 'D10,D12,D16,D17,D19,D27').",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of repeated cross-validation trials (paper uses 30). Default=1.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds. Default=5.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to save output tables and figures.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate and save publication figures (ROC/PR curves and PCA projections).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default=42.",
    )
    return parser.parse_args()


def df_to_markdown(df: pd.DataFrame) -> str:
    """Renders a pandas DataFrame as a GitHub-flavored markdown table."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
        return "\n".join([header, sep] + rows)


def save_tables(results: dict, out_dir: Path) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 2: Wins
    results["table2_wins"].to_csv(tables_dir / "table2_wins.csv", index=False)
    with open(tables_dir / "table2_wins.md", "w", encoding="utf-8") as f:
        f.write(df_to_markdown(results["table2_wins"]))

    # Table 3: Wilcoxon p-values
    results["table3_p_values"].to_csv(tables_dir / "table3_wilcoxon_pvalues.csv", index=False)
    with open(tables_dir / "table3_wilcoxon_pvalues.md", "w", encoding="utf-8") as f:
        f.write(df_to_markdown(results["table3_p_values"]))

    # Table 4: Wilcoxon ranks & effect sizes
    results["table4_wilcoxon_ranks"].to_csv(tables_dir / "table4_wilcoxon_ranks.csv", index=False)
    with open(tables_dir / "table4_wilcoxon_ranks.md", "w", encoding="utf-8") as f:
        f.write(df_to_markdown(results["table4_wilcoxon_ranks"]))

    # Tables 5-8: Mean +/- STD per configuration
    mapping = {
        "AUC_SVM": "table5_auc_svm",
        "AUPR_SVM": "table6_aupr_svm",
        "AUC_NB": "table7_auc_nb",
        "AUPR_NB": "table8_aupr_nb",
    }
    for key, filename in mapping.items():
        if key in results["summary_tables"]:
            df = results["summary_tables"][key]
            # Select display columns (Dataset, Name, and formatted methods)
            display_cols = ["Dataset", "Name"] + [c for c in df.columns if not c.endswith("_mean") and c not in ["Dataset", "Name"]]
            df[display_cols].to_csv(tables_dir / f"{filename}.csv", index=False)
            with open(tables_dir / f"{filename}.md", "w", encoding="utf-8") as f:
                f.write(df_to_markdown(df[display_cols]))

    # Master Markdown Report
    report_path = out_dir / "benchmark_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SMOTEFUNA Empirical Benchmark Evaluation Report\n\n")
        f.write("Based on Tarawneh et al., *IEEE Access*, 2020: **SMOTEFUNA: Synthetic Minority Over-Sampling Technique Based on Furthest Neighbour Algorithm**.\n\n")

        f.write("## 1. Table 2: Number of Wins for Each Algorithm\n")
        f.write(df_to_markdown(results["table2_wins"]))
        f.write("\n\n")

        f.write("## 2. Table 3: Wilcoxon Signed-Rank Test (p-values)\n")
        f.write(df_to_markdown(results["table3_p_values"]))
        f.write("\n\n")

        f.write("## 3. Table 4: Wilcoxon Effect Size R, W+, and W-\n")
        f.write(df_to_markdown(results["table4_wilcoxon_ranks"]))
        f.write("\n\n")

        f.write("## 4. Tables 5-8: Detailed Mean \u00b1 STD Performance\n\n")
        for key, filename in mapping.items():
            if key in results["summary_tables"]:
                title = key.replace("_", " (") + " Classifier)"
                f.write(f"### {title}\n\n")
                df = results["summary_tables"][key]
                display_cols = ["Dataset", "Name"] + [c for c in df.columns if not c.endswith("_mean") and c not in ["Dataset", "Name"]]
                f.write(df_to_markdown(df[display_cols]))
                f.write("\n\n")

    print(f"Saved all tables and markdown report to {out_dir}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" SMOTEFUNA RESEARCH BENCHMARK EXPERIMENT PIPELINE")
    print("=================================================================")

    # Select datasets
    if args.datasets:
        d_ids = [d.strip() for d in args.datasets.split(",") if d.strip()]
        datasets = {f"{meta.dataset_id}_{meta.name}": load_benchmark_dataset(meta.dataset_id, seed=args.seed) for meta in [TABLE_1_DATASETS[d] if d in TABLE_1_DATASETS else load_benchmark_dataset(d, seed=args.seed)[2] for d in d_ids]}
    elif args.full:
        print("Loading all 33 benchmark datasets from Table 1...")
        datasets = load_all_benchmark_datasets(seed=args.seed)
    else:
        # Default or quick: representative suite
        print("Loading representative suite of benchmark datasets...")
        datasets = load_representative_suite(seed=args.seed)

    print(f"Selected {len(datasets)} datasets for evaluation.")
    print(f"Configuration: 5-fold CV, {args.n_trials} trial(s), seed={args.seed}")

    pipeline = BenchmarkPipeline(
        n_splits=args.n_splits,
        n_trials=args.n_trials,
        random_state=args.seed,
        verbose=True,
    )

    results = pipeline.run_suite(datasets)

    # Save CSV and Markdown tables
    save_tables(results, out_dir)

    # Print Win Table in console
    print("\n-----------------------------------------------------------------")
    print("TABLE 2: WIN COUNTS SUMMARY")
    print("-----------------------------------------------------------------")
    print(results["table2_wins"].to_string(index=False))
    print("\n-----------------------------------------------------------------")
    print("TABLE 3: WILCOXON SIGNED-RANK TEST P-VALUES")
    print("-----------------------------------------------------------------")
    print(results["table3_p_values"].to_string(index=False))
    print("\n-----------------------------------------------------------------")
    print("TABLE 4: WILCOXON EFFECT SIZE R, W+, W-")
    print("-----------------------------------------------------------------")
    print(results["table4_wilcoxon_ranks"].to_string(index=False))
    print("-----------------------------------------------------------------")

    # Generate Figures
    if args.plots:
        figures_dir = out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        print("\nGenerating publication figures...")

        # Figures 9 & 10: ROC and PR Curves for D10, D16, D17, D19 (or available datasets)
        plot_d_ids = [d for d in ["D10", "D16", "D17", "D19"] if any(k.startswith(f"{d}_") for k in datasets)]
        if not plot_d_ids:
            plot_d_ids = [k.split("_")[0] for k in list(datasets.keys())[:4]]

        plot_roc_and_pr_grid(
            dataset_ids=plot_d_ids,
            classifier_name="NB",
            save_path=figures_dir / "fig9_nb_roc_pr_curves.png",
            seed=args.seed,
        )

        plot_roc_and_pr_grid(
            dataset_ids=plot_d_ids,
            classifier_name="SVM",
            save_path=figures_dir / "fig10_svm_roc_pr_curves.png",
            seed=args.seed,
        )

        # Figure 8: PCA 2D projections
        pca_d_ids = [d for d in ["D4", "D6", "D12", "D13", "D27", "D31"] if any(k.startswith(f"{d}_") for k in datasets)]
        if len(pca_d_ids) < 3:
            pca_d_ids = [k.split("_")[0] for k in list(datasets.keys())[:min(6, len(datasets))]]

        plot_pca_projections(
            dataset_ids=pca_d_ids,
            save_path=figures_dir / "fig8_pca_projections.png",
            seed=args.seed,
        )
        print(f"Publication figures successfully saved to {figures_dir}")

    print("\nBenchmark execution completed successfully!")


if __name__ == "__main__":
    main()
