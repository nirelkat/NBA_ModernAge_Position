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

## 1b. `notebooks/02_Features_Scaling.ipynb` — section-by-section mapping

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| §1 Load filtered population | Loads the 379-player EDA export | HW1 1.1 | ✅ Load & verify | Loads a saved intermediate instead of re-filtering |
| **§2 Feature selection** (13 features, 4 role dimensions) | Chooses rate-based style features; excludes volume, efficiency and impact | **HW1 2.1** (drop columns with written justification) | 🔁 Justified column selection | Selection criterion is *style vs. quality*, specific to archetype discovery |
| §2 Engineered features (`rim_share`, `mid_share`, `corner3_rate`) | Merges zones; builds corner-3 share of all FGA | HW1 2.2 (transforming raw columns into model-ready ones) | 🔁 Feature construction | 🆕 Domain-driven construction that removes undefined values instead of imputing them |
| §2 Raw vs. engineered corner-3 ranking | Shows the raw share is topped by 1-attempt centres | HW1 2.3 (consequences of how missing / sparse data is handled) | 🔁 Sanity-checking a preprocessing choice | Small-sample noise handled by weighting with 3PA rate |
| **§3 VIF check** | VIF of naive zone set (≈ 88,000) vs. selected set (max 8.8) | **HW2 2.3 – 2.4** (multicollinearity: Radio ↔ Newspaper) | 🔁 Multicollinearity diagnosis | 🆕 VIF and compositional (sum-to-1) collinearity go beyond the pairwise correlation used in HW2 |
| §3 Correlation heatmap (13 features) | Confirms no near-duplicates remain | HW1 1.4 · HW2 2.4 | ✅ Correlation heatmap | Distinguishes duplicate *measurements* from related *skills* |
| **§4 Three-scaler comparison** | StandardScaler vs. PowerTransformer vs. QuantileTransformer: skew, kurtosis, max \|z\|, spread share | **HW1 2.4** (Member A Power, Member B Quantile-normal, Member C Standard) | ✅ The exact same three scalers | Compared on one shared feature set with quantitative criteria instead of one scaler per member |
| §4 3 × 4 histogram grid | Visual effect of each scaler on four features | HW1 1.3 / 2.4 (histogram shape before/after) | ✅ Distribution-shape reading | Exposes the QuantileTransformer tie artefact (−5.2 block) |
| **§5a Distance & neighbour agreement** | Spearman of 71,631 pairwise distances; 10-NN overlap | **HW3 1.1, 2.3** (distances change with scaling) · HW3 1.2 (KNN) | 🔁 Effect of preprocessing on distance-based methods | Measured on the whole geometry rather than on one model's accuracy |
| §5b Nearest neighbours of well-known players | Face-validity check of the feature space | HW3 1.2 (nearest-neighbour lookup) · HW4 Q3 (similar users) | 🔁 "Most similar" lookup | Neighbours judged by basketball role |
| §5 Scaling decision table | Power for PCA/clustering, Standard for anomalies | HW1 3.3 (group comparison & conclusion) · HW4 Q2c, Q4c (outliers vs. clusters) | 🔁 Choosing preprocessing by downstream use | Different scaler per downstream task |
| §6 Export (raw / power / standard) | Saves aligned feature tables with metadata | — | — | Metadata (`pos`, `pos_played`) kept for validation only |

---

## 1c. `notebooks/03_PCA.ipynb` — section-by-section mapping

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| §1 Load scaled features | Power-transformed matrix, mean 0 / std 1 check | HW1 3.1 (PCA on each member's scaled data) | ✅ PCA requires standardised inputs | Input is the scaler chosen in notebook 02 |
| **§2 Scree plot + cumulative variance** | Eigenvalues, explained and cumulative variance | **HW1 3.1** Scree plot | ✅ Same plot | — |
| §2 Kaiser rule & parallel analysis | Keeps 3 components (both rules agree) | HW1 3.1 (choosing the number of components) | 🔁 Component retention | 🆕 Parallel analysis (permutation baseline) goes beyond the elbow / variance-threshold rule |
| **§3 Loadings heatmap + sign convention** | Names PC1–PC3 from feature weights | HW1 3.2 (interpreting the components) | 🔁 Reading loadings | Axes named in basketball terms; anchor-feature sign convention |
| §3 Extreme players per component | Top / bottom 5 players on PC1–PC3 | HW4 Q1a (reading one point's position relative to groups) | 🔁 Interpreting individual observations | Face validity with real players |
| **§4 PC1 × PC2 scatter coloured by position** | Player map, listed vs. played position, group medians | **HW1 3.2** 2-component scatter coloured by `Type` | ✅ Same plot | Two colourings (roster label vs. play-by-play usage) and annotated stars |
| §4 η² of components by position | PC1 η² ≈ 0.6, PC2–PC3 ≤ 0.16 | HW2 2.2 (R² / variance explained) | 🔁 Variance explained by a grouping | Quantifies which style axes positions miss |
| §4 Biplot | Loadings drawn as arrows over the scores | HW1 3.2 (component interpretation) | 🔁 Linking components back to features | 🆕 Biplot not used in the homework |
| **§5 Robustness: Power vs. Standard PCA** | \|cos\| of loadings, score correlation | **HW1 3.3** (group comparison of PCA results across preprocessing choices) | ✅ Compare PCA across scalers | Quantitative comparison; explains PC4–PC5 instability (near-equal eigenvalues) |
| §6 Clustering-input decision & export | Full 13 features primary, 3 PCs as sensitivity check | HW4 Q1 (choosing inputs and k with evidence) | 🔁 Evidence-based modelling choice | Saves scores, loadings, explained variance |

---

## 1d. `notebooks/04_Clustering.ipynb` — section-by-section mapping

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| **§2 k-means for k = 2…12: inertia & silhouette** | Elbow plot and average silhouette per k | **HW4 Q1b–c** (inertia always falls; silhouette rises then falls) | ✅ Same two criteria, same reasoning | Applied to real data where no k is clearly best |
| §2 Calinski-Harabasz, Davies-Bouldin | Two more internal validity indices | HW4 Q1 | 🔁 Choosing k with several criteria | 🆕 Additional indices beyond the homework |
| §2 Bootstrap stability (ARI) | Refit on resampled players | HW3 4.1 (cross-validation, stability / variance of a model) | 🔁 Stability under resampling | 🆕 Bootstrap ARI for an unsupervised model |
| (implementation) k-means | `sklearn.cluster.KMeans` | **HW4 Q4a** (k-means from scratch, checked against sklearn) | ✅ Same algorithm | Library version, `n_init=50` restarts |
| **§3 Ward dendrogram + cophenetic correlation** | Hierarchical clustering, merge-height gaps | — | — | 🆕 Hierarchical clustering not covered in the homework |
| **§4 GMM with BIC (full / diag / tied)** | Model selection for mixture models | HW2 Part 3 (complexity vs. overfitting) | 🔁 Penalising model complexity | 🆕 GMM and BIC are new; `full` covariance is penalised just as high polynomial degrees overfit in HW2 |
| **§5 Choosing k: statistics vs. usefulness** | k = 3 families (statistical optimum) and k = 8 archetypes (useful granularity) | **HW4 Q1d** (StreamFlix: best-silhouette k vs. a business constraint of at most 4 tracks) | ✅ Combining the metric with a practical constraint | Two-level taxonomy instead of one k |
| §6 Archetype fingerprints & profiles | Cluster means (scaled) and medians (raw units), typical members | HW1 1.5 / 3.2 (describing groups) | 🔁 Group profiling | Names attached through anchor players |
| §6c Archetypes on the PCA map | PC1 × PC2 and PC1 × PC3 coloured by archetype | HW1 3.2 (PCA scatter coloured by category) | ✅ Same plot | Categories are discovered clusters, not given labels |
| **§7a Silhouette plot (per player)** | Silhouette of every player by archetype | **HW4 Q1a** (silhouette of one viewer: a, b, s) | ✅ Per-point silhouette | Aggregated per archetype; % negative |
| §7b Per-archetype bootstrap Jaccard | Stability of each cluster | HW3 4.1 (stability) | 🔁 Resampling-based reliability | 🆕 Cluster-wise Jaccard |
| **§8 GMM soft membership & hybrids** | Probability per archetype; hybrids = max probability < 0.6 | HW4 Q1a (silhouette near 0 = on the border between groups) | 🔁 Border points between groups | 🆕 Soft clustering quantifies "how much" a player belongs |
| §9 Agreement across methods & inputs (ARI) | k-means vs. Ward, GMM, Standard scaling, PC scores | HW4 Q4 (`same_partition` check against sklearn) · HW1 3.3 (comparison across preprocessing) | 🔁 Comparing partitions | ARI instead of an exact-match check |
| §10 Archetype × position crosstab, ARI preview | Archetypes vs. listed positions | HW2 Part 4 (confusion-matrix reading) | 🔁 Cross-tabulation | Full validation in notebook 06 |

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
| Dropping unusable columns / rows | HW1 2.1 | 01_EDA §2, §3 · 02 §2 |
| Categorical encoding | HW1 2.2 | Not needed (`pos` is never an input) |
| Feature construction | HW1 2.2 | 02 §2 (`mid_share`, `corner3_rate`) |
| Scaling: StandardScaler / PowerTransformer / QuantileTransformer | HW1 2.4 | 01_EDA §4 · 02 §4 (3-scaler comparison) |
| Multicollinearity | HW2 2.3 – 2.4 | 01_EDA §5 · 02 §3 (VIF) |
| R² / variance explained | HW2 2.2 | 01_EDA §6 (η²) · 02 §3 (VIF = 1 / (1 − R²)) |
| Distances & effect of scaling | HW3 1.1, 2.3 | 01_EDA §4 rationale · 02 §5 (distance & neighbour agreement) |
| Nearest-neighbour lookup | HW3 1.2 · HW4 Q3 | 02 §5b (similar players) |
| PCA: scree plot | HW1 3.1 | 03_PCA §2 (+ Kaiser & parallel analysis) |
| PCA: 2-component scatter coloured by category | HW1 3.2 | 03_PCA §4 (listed & played position) · §3 loadings |
| PCA comparison across preprocessing choices | HW1 3.3 | 03_PCA §5 (PowerTransformer vs. StandardScaler) |
| k-means (implementation & behaviour) | HW4 Q4a | 04_Clustering §2, §6 |
| Choosing k: inertia vs. silhouette, business constraints | HW4 Q1 | 04_Clustering §2, §5 (3 families / 8 archetypes) |
| Per-point silhouette (border points) | HW4 Q1a | 04_Clustering §7a · §8 (GMM hybrids) |
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
| `02_Features_Scaling.ipynb` ✅ done | 13-feature matrix, VIF check, StandardScaler vs. PowerTransformer vs. QuantileTransformer | HW1 2.1, 2.2, 2.4 · HW2 2.3 – 2.4 · HW3 1.1 – 1.2, 2.3 |
| `03_PCA.ipynb` ✅ done | Scree plot, parallel analysis, loadings, player map, biplot, robustness | HW1 3.1 – 3.3 · HW2 2.2 |
| `04_Clustering.ipynb` ✅ done | k-means (5 criteria + stability), Ward dendrogram, GMM + BIC, 8 archetypes, hybrids | HW4 Q1, Q4a · HW3 4.1 · HW2 Part 3 |
| `05_Anomalies.ipynb` | Isolation Forest, LOF, DBSCAN noise comparison | HW4 Q2, Q4b – c |
| `06_Validation.ipynb` | ARI vs. listed and played positions, cluster profiles | HW2 Part 4 · HW3 4.1 |
