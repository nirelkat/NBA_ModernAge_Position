# Homework Mapping — Course Concepts → Project Pipeline

This document traces how each step of the **Positionless NBA Player Archetypes** project corresponds to the exercises
and concepts practised in the *Introduction to AI and Data Science* course homework. The notebooks themselves are
written as a standalone project; this file is where the link to the coursework is kept.

## Source homework

| ID | Notebook | Topics | Dataset(s) |
|---|---|---|---|
| **HW1** | `HW1_EDA_Preprocessing_PCA` | EDA, missing values, imputation, scaling, PCA | Melbourne Housing |
| **HW2** | `group_05` (Homework 2) | Linear regression, multicollinearity, overfitting, classification metrics | Advertising, California Housing |
| **HW3** | `HW3_KNN_and_DT` | Distances, KNN from scratch, scaling sanity check, decision trees, CV | Pima Diabetes |
| **HW4** | `group_05` (Homework 4) | k-means, inertia & silhouette, DBSCAN, similarity measures | StreamFlix, Northwind, TuneIn, synthetic blobs |
| **HW5** | `HW5_ANN_CNN` | Backpropagation, ANN, CNN | make_moons, images |

Legend: ✅ direct reuse of the technique · 🔁 technique adapted to the NBA domain · 🆕 new step with no homework
equivalent · 🔜 planned for a later notebook

---

## 1. `notebooks/01_EDA.ipynb` — section-by-section mapping

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| Setup | Imports, constants (`MIN_MP`, `POS_ORDER`) | HW1 setup cell | ✅ pandas / seaborn / sklearn preprocessing stack | — |
| **§1 Load & Inspect** — `shape`, `info()`, `head()` | Integrity check: row count, dtypes, non-null counts | **HW1 1.1** Load and inspect | ✅ First-look inventory of the data | 127 columns from 6 merged tables → explicit one-row-per-player check |
| §1 Column-group table | Groups 127 columns by meaning (style / volume / quality) | HW1 2.1 (choosing which columns are usable) | 🔁 Feature relevance reasoning before modelling | Grouping by *style vs. quality* is specific to archetype discovery |
| §1 `pos` as validation-only | Listed position kept out of the feature matrix | **HW1 3.2** (`Type` used only to colour the PCA scatter) | ✅ Category label used for interpretation, not as an input | Becomes the ARI ground truth at the end of the pipeline |
| **§2 500-minute filter** | Removes 203 low-minute players (582 → 379) | HW1 2.1 (drop rows without `Price`) | 🔁 Remove rows that cannot be trusted | 🆕 Reliability filter on *sample size* (rate stats from tiny samples); no direct homework equivalent |
| §2 BLK% vs. minutes "funnel" scatter | Visual justification of the cut-off | HW2 2.1 (scatter plots to inspect relationships) | 🔁 Scatter plot as a diagnostic | Used to show variance shrinking with sample size |
| **§3 Missing-value report** — count, %, bar chart | Before/after filter missingness table and plot | **HW1 1.2** Missing values report + bar chart | ✅ Same report format | Shown both before and after the minutes filter |
| §3 Missingness mechanism check | Proves every NaN has a zero denominator | **HW1 2.3** Imputation (Iterative / KNN / median by member) | 🔁 Choosing an imputation strategy | Key difference: values are *undefined*, not *unobserved*, so imputation is avoided by design |
| §3 Median-imputation demo | Shows imputation would make non-shooting centres 35.4% shooters | HW1 2.3 Member C (SimpleImputer, median) | 🔁 Critique of the median imputer | Domain argument replaces "which imputer scores best" |
| §3 Handling-decisions table | Exclude make-% columns, fill corner-3 share with 0 | HW1 2.1 (justify each dropped column) | ✅ Written justification per column | — |
| **§4 Histograms of USG%, 3PA rate, AST%, BLK%** | Shape, skew, mean vs. median, bimodality | **HW1 1.3** Histograms of 3 numeric features + shape description | ✅ Describe each distribution's shape | Four role features instead of three; skew value printed in each title |
| §4 Skewness table (23 features, raw vs. Yeo-Johnson) | Quantifies how many features need reshaping | **HW1 2.4** Scaling (each member a different method) | 🔁 Choosing a scaler from the data's shape | Decision made *before* modelling, from skewness, rather than by trying all three |
| §4 StandardScaler vs. PowerTransformer side-by-side | Visual comparison of the two transforms | HW1 2.4 Member A (PowerTransformer) vs. Member C (StandardScaler) | ✅ Same two scalers compared | Recommendation also covers which version anomaly detection should use |
| §4 Distance-sensitivity rationale | Why skew matters for Euclidean methods | **HW3 1.1 / 2.3** (raw vs. scaled distances, scaling sanity check); **HW4 Q4** k-means | 🔁 Distance-based models are sensitive to feature scale and tails | Extended from *scale* to *shape* (long tails) |
| **§5a Per-game vs. per-100 correlation table** | Shows per-game stats track minutes, per-100 do not | HW1 1.4 (correlation) | 🔁 Correlation used to diagnose a confounder | 🆕 Normalisation-by-opportunity is domain-specific (pace & minutes) |
| **§5b Correlation heatmap** | 23 × 23 annotated heatmap | **HW1 1.4** Correlation matrix heatmap | ✅ Same plot | Restricted to candidate style features for readability |
| §5b Sorted list of pairs with \|r\| ≥ 0.8 | Explicit redundancy list | HW1 1.4 (top 3 correlated pairs + explanation) | ✅ Identify and explain top pairs | Threshold-based list instead of "top 3" |
| §5 Redundant-groups table + zone-share constraint | One representative per concept | **HW2 2.3 – 2.4** (Radio ↔ Newspaper multicollinearity) | 🔁 Correlated features distort models | In clustering, redundancy = implicit double weighting in the distance; compositional (sum-to-1) collinearity |
| **§6 Boxplots by listed position** | Six style stats × five positions | **HW1 1.5** Boxplot of `Price` by `Type` | ✅ Same plot type | Inverted question: does the category *fail* to explain the variable? |
| §6 η² (eta-squared) table | Share of variance explained by position | **HW2 2.2** (R² of simple regressions) | 🔁 Variance explained | η² = R² of a regression on position dummies |
| §6 Listed vs. played position crosstab | 77% agreement, disagreements only between neighbouring positions | HW2 4.1 (confusion-matrix reading) | 🔁 Reading agreement / disagreement in a cross-tabulation | Play-by-play data used as a second ground truth |
| §6 Implication → soft membership | Motivates GMM | HW4 Q1 (silhouette ≈ "how well a point fits its group", border points) | 🔁 Points on the border between groups | 🔜 GMM soft membership in the clustering notebook |
| **§7 Summary & export** | Decision table, saves `data/processed/nba_2025_26_mp500.csv` | HW1 3.3 / HW3 4.2 (group comparison & recommendation) | ✅ Decisions backed by numbers | — |

---

## 2. Concept index — where each course concept appears

| Course concept | Homework | Project location |
|---|---|---|
| Data loading & inspection | HW1 1.1 | 01_EDA §1 |
| Missing-value report | HW1 1.2 | 01_EDA §3 |
| Imputation strategies (Iterative, KNN, median) | HW1 2.3 | 01_EDA §3 (evaluated and rejected for structural NaNs) |
| Histograms & distribution shape | HW1 1.3 | 01_EDA §4 |
| Correlation heatmap | HW1 1.4 | 01_EDA §5 |
| Boxplot by category | HW1 1.5 | 01_EDA §6 |
| Dropping unusable columns / rows | HW1 2.1 | 01_EDA §2, §3 |
| Categorical encoding | HW1 2.2 | Not needed (`pos` is never an input) |
| Scaling: StandardScaler / PowerTransformer / QuantileTransformer | HW1 2.4 | 01_EDA §4 · 🔜 02 Feature selection & scaling (3-scaler comparison) |
| Multicollinearity | HW2 2.3 – 2.4 | 01_EDA §5 · 🔜 02 (re-check on the final feature set) |
| R² / variance explained | HW2 2.2 | 01_EDA §6 (η²) |
| Distances & effect of scaling | HW3 1.1, 2.3 | 01_EDA §4 rationale · 🔜 03 / 04 |
| PCA: scree plot | HW1 3.1 | 🔜 03 Dimensionality reduction |
| PCA: 2-component scatter coloured by category | HW1 3.2 | 🔜 03 (coloured by listed position) |
| PCA comparison across preprocessing choices | HW1 3.3 | 🔜 03 (PowerTransformer vs. StandardScaler inputs) |
| k-means (implementation & behaviour) | HW4 Q4a | 🔜 04 Clustering |
| Choosing k: inertia vs. silhouette, business constraints | HW4 Q1 | 🔜 04 (number of archetypes) |
| Per-point silhouette (border points) | HW4 Q1a | 🔜 04 (players between archetypes) |
| DBSCAN, core / border / noise, effect of ε | HW4 Q2, Q4b | 🔜 05 Anomaly detection (comparison with Isolation Forest / LOF) |
| k-means forces outliers into clusters vs. DBSCAN noise | HW4 Q2c, Q4c | 🔜 05 (motivation for anomaly detection) |
| Cosine vs. Euclidean similarity, mean-centring | HW4 Q3 | 🔜 player-similarity / "closest comparable player" lookup |
| Confusion matrix / agreement metrics | HW2 Part 4 | 🔜 06 Validation (ARI, cluster × position crosstab) |
| Cross-validation & stability | HW3 4.1 | 🔜 06 (bootstrap stability of clusters) |
| Neural networks (ANN / CNN) | HW5 | Not used; outside the scope of this unsupervised project |

---

## 3. Planned notebooks — expected homework links

| Notebook | Main steps | Primary homework links |
|---|---|---|
| `02_Features_Scaling.ipynb` | Final ~14-feature matrix, redundancy re-check, StandardScaler vs. PowerTransformer vs. QuantileTransformer | HW1 2.1, 2.4 · HW2 2.4 |
| `03_PCA.ipynb` | Scree plot, loadings, 2-D projection coloured by position | HW1 3.1 – 3.3 |
| `04_Clustering.ipynb` | k-means (elbow + silhouette), hierarchical clustering, GMM soft membership | HW4 Q1, Q4a |
| `05_Anomalies.ipynb` | Isolation Forest, LOF, DBSCAN noise comparison | HW4 Q2, Q4b – c |
| `06_Validation.ipynb` | ARI vs. listed and played positions, cluster profiles | HW2 Part 4 · HW3 4.1 |
