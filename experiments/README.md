# SMOTEFUNA Experimental Benchmark & Replication Suite

This directory contains the independent, thorough experimental replication suite for the research paper:
> **"SMOTEFUNA: Synthetic Minority Over-Sampling Technique Based on Furthest Neighbour Algorithm"**  
> *Ahmad S. Tarawneh, Ahmad B. A. Hassanat, Khalid Almohammadi, Dmitry Chetverikov, Colin Bellinger*  
> **IEEE Access**, Vol. 8, pp. 59069–59082, 2020. DOI: `10.1109/ACCESS.2020.2983003`.

---

## 1. Identified Gaps in Initial Repository & Resolutions

| Evaluation Dimension | Paper Specification | Initial Repo State | Implemented Resolution |
| :--- | :--- | :--- | :--- |
| **Datasets** | 33 benchmark datasets (Table 1: Ar1-6, Kc1-2, Pc1-4, Australian, Bank, Heart, Oil-Spill, Phoneme, Abalone, Page-blocks, Pima, Segment, Shuttle, Vehicle0-3, Vowel, Wisconsin, Yeast) | Only a toy 2D synthetic dataset in `demo.py` | Full 33-dataset catalog (`experiments/datasets.py`) supporting real datasets (Wisconsin, Pima, Bank, Vehicle, Heart), OpenML caching, and high-fidelity Table 1 generative benchmarks matching exact sample counts, feature counts, and imbalance ratios. |
| **Resampling Baselines** | SMOTEFUNA compared against SMOTE ($k=5$), ADASYN, SWIM, and Original Data (no-resampling baseline) | Only SMOTEFUNA was implemented | Unified baseline interface (`experiments/baselines.py`) implementing SMOTE, ADASYN, SWIM (Sampling With the Majority via RBF mutual density), and Original Data with full scikit-learn compatibility. |
| **Classifiers** | Naive Bayes (NB) and Support Vector Machines (SVM with RBF kernel) | Isolated checks in demo | Unified pipeline (`experiments/pipeline.py`) running both `GaussianNB` and `SVC(kernel='rbf', probability=True)`. |
| **Validation Protocol** | 5-Fold Cross-Validation, repeated over trials, with z-score standardization | Single 70/30 train-test split | Leakage-free Stratified 5-Fold CV: StandardScaler and Resamplers fit **only** on the training fold, testing on the untouched imbalanced test fold. |
| **Metrics** | ROC-AUC (Area Under ROC) and AUPR (Area Under Precision-Recall Curve) | Only ROC-AUC in demo | Full evaluation metrics (`experiments/metrics.py`) computing both ROC-AUC and AUPR (average precision), plus win-matrix analysis with tie handling. |
| **Statistical Tests** | Wilcoxon Signed-Rank Test across datasets (p-value, $W+$, $W-$, $Z$, and Effect Size $R = \frac{\|Z\|}{\sqrt{N}}$) | None | Complete Wilcoxon signed-rank test suite (`experiments/statistical_tests.py`) generating exact replications of Table 3 ($p$-values) and Table 4 ($R / W+ / W-$). |
| **Visualizations** | Multi-method ROC & PR curve grids (Figures 9 & 10) and 2D PCA distribution projections (Figure 8) | Single 2D scatter plot | Publication-ready visualizers (`experiments/visualize.py`) generating Figures 8, 9, and 10. |

---

## 2. Directory Architecture

```
experiments/
├── __init__.py               # Exports core classes, metrics, and loaders
├── baselines.py              # Resamplers: SMOTEFUNA, SMOTE, ADASYN, SWIM, Original
├── datasets.py               # Benchmark datasets (Real datasets + Table 1 specifications)
├── metrics.py                # ROC-AUC, AUPR, and win count calculation (Table 2)
├── statistical_tests.py      # Wilcoxon signed-rank test (Tables 3 & 4)
├── pipeline.py               # 5-fold cross-validation pipeline (leakage-free)
├── visualize.py              # Figures 8, 9, 10 generator (ROC, PR, PCA)
├── run_experiments.py        # Master CLI runner
└── README.md                 # This documentation
```

---

## 3. Usage & Running Experiments

### Quick Benchmark (Representative Suite)
Runs 5-fold cross-validation across representative benchmark datasets spanning various imbalance ratios:
```bash
python experiments/run_experiments.py --quick
```

### Full Benchmark (All 33 Datasets from Table 1)
Generated stand-ins (offline):
```bash
python experiments/run_experiments.py --full --n-trials 1
```

Real OpenML / KEEL-style data (needs internet the first time, then caches under `experiments/data_cache/`):
```bash
python experiments/run_experiments.py --full --real --n-trials 5 --output-dir experiments/results_real
```

### Specific Datasets & Multi-Trial Evaluation
```bash
python experiments/run_experiments.py --datasets D10,D12,D16,D17,D19,D27 --n-trials 5
```

---

## 4. Generated Artifacts & Output Tables

When executed, outputs are stored in `experiments/results/`:
- **`tables/table2_wins.md`**: Number of wins across SVM and NB on AUC and AUPR.
- **`tables/table3_wilcoxon_pvalues.md`**: Wilcoxon $p$-values comparing SMOTEFUNA against SMOTE, ADASYN, SWIM, and Original Data.
- **`tables/table4_wilcoxon_ranks.md`**: Effect size $R$, positive ranks $W+$, and negative ranks $W-$.
- **`tables/table5_auc_svm.md`**: Mean $\pm$ STD AUC with SVM classifier.
- **`tables/table6_aupr_svm.md`**: Mean $\pm$ STD AUPR with SVM classifier.
- **`tables/table7_auc_nb.md`**: Mean $\pm$ STD AUC with NB classifier.
- **`tables/table8_aupr_nb.md`**: Mean $\pm$ STD AUPR with NB classifier.
- **`figures/fig8_pca_projections.png`**: 2D PCA projection of majority, minority, and SMOTEFUNA synthetic samples.
- **`figures/fig9_nb_roc_pr_curves.png`**: ROC and PR curves for NB classifier.
- **`figures/fig10_svm_roc_pr_curves.png`**: ROC and PR curves for SVM classifier.
- **`benchmark_summary.md`**: Complete consolidated experimental report.

