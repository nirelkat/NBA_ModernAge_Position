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
| §6 Implication → read labels with fit quality | Hard labels plus silhouette | HW4 Q1a (silhouette ≈ "how well a point fits its group", border points) | 🔁 Points on the border between groups | Border players identified with silhouette in notebook 04 §6 |
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

Methods are restricted to those practised in the course: **k-means** (primary model) and **DBSCAN** (density-based
perspective), both from HW4.

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| (implementation) k-means | `sklearn.cluster.KMeans`, 50 restarts | **HW4 Q4a** (k-means from scratch: assign to nearest centroid, move centroid to the mean; checked against sklearn) | ✅ Same algorithm | Library version with multiple initialisations |
| **§2 k-means for k = 2…12: inertia & silhouette** | Elbow plot and average silhouette per k | **HW4 Q1b–c** (inertia always falls with k, so "lowest inertia" can never choose k; silhouette rises then falls) | ✅ Same two criteria, same reasoning | Applied to real data where no k is clearly best |
| §2 Bootstrap stability (ARI) | Refit on resampled players | HW3 4.1 (cross-validation: stability / variance of a model) | 🔁 Stability under resampling | 🆕 Bootstrap ARI for an unsupervised model |
| **§3 Choosing k: statistics vs. usefulness** | k = 3 families (statistical optimum) and k = 8 archetypes (useful granularity) | **HW4 Q1c–d** (StreamFlix: best-silhouette k combined with a business constraint of at most 4 tracks) | ✅ Combining the metric with a practical constraint | Two-level taxonomy instead of a single k |
| §4 Centroid fingerprints & profiles | Centroids in scaled units, medians in raw units, most typical members | HW4 Q4a (centroids as cluster means) · HW1 1.5 (describing groups) | 🔁 Interpreting centroids | Archetype names attached through anchor players |
| §4c Archetypes on the PCA map | PC1 × PC2 and PC1 × PC3 coloured by archetype, centroids marked | HW1 3.2 (PCA scatter coloured by category) | ✅ Same plot | Categories are discovered clusters, not given labels |
| **§5a Silhouette plot (per player)** | Silhouette of every player by archetype | **HW4 Q1a** (silhouette of one viewer: a, b, s) | ✅ Per-point silhouette, same formula | Aggregated per archetype; % negative |
| §5b Per-archetype bootstrap Jaccard | Stability of each cluster | HW3 4.1 (stability) | 🔁 Resampling-based reliability | 🆕 Cluster-wise Jaccard |
| **§6 Border players** | Silhouette < 0.05; second archetype = the cluster that defines *b* | **HW4 Q1a** (s near 0 = "sits on the border between groups"; *b* = nearest other group) | ✅ Direct use of a, b and s | Applied to all 379 players to find versatile players |
| §7 Robustness to preprocessing (ARI) | k-means on StandardScaler features and on PC scores | HW4 Q4 (`same_partition` check against sklearn) · HW1 3.3 (comparison across preprocessing) | 🔁 Comparing partitions | ARI instead of an exact-match check |
| **§8 DBSCAN: core / border / noise** | Point types, clusters as chains of core neighbourhoods | **HW4 Q2a–b** (Northwind: label each depot core / border / noise, list hubs and noise) | ✅ Same definitions | 13-dimensional data instead of a 10-depot distance matrix |
| §8a k-distance plot | Distance to the `min_samples`-th neighbour, sorted | HW4 Q2d-1 (the single distance that decides when a point stops being noise) | 🔁 Reasoning about ε from neighbour distances | 🆕 k-distance plot as a tool to pick ε |
| **§8b ε × `min_samples` sweep** | Number of clusters, sizeable clusters and noise % over a grid | **HW4 Q2d** (smallest ε at which noise joins a hub / hubs merge) | ✅ Effect of ε on clusters and noise | Full grid instead of single thresholds |
| §8b Monotonicity check | Noise never increases with ε | **HW4 Q2d-3** ("growing ε only adds neighbours, so it can only merge clusters or absorb noise, never split") | ✅ Same property, verified empirically | — |
| **§8c Dense cores vs. k-means archetypes** | ε = 2.25: interior core, perimeter core, noise vs. archetypes | **HW4 Q2c, Q4c** (k-means forces outliers into clusters; DBSCAN leaves them as noise) | ✅ Same comparison | Shows only rim-running bigs are density-separated |
| §8d Outlier setting (ε = 2.75) | Players still noise when the league is connected | HW4 Q2 (D9, D10 as remote one-offs) · Q4c | 🔁 Noise points as outliers | Bridge to notebook 05 (anomaly detection) |
| §9 Archetype × position crosstab, ARI preview | Archetypes vs. listed positions and vs. families | HW2 Part 4 (confusion-matrix reading) | 🔁 Cross-tabulation | Full validation in notebook 06 |

---

## 1e. `notebooks/05_Anomalies.ipynb` — section-by-section mapping

Anomaly detection is built only from techniques practised in the course: DBSCAN noise, k-means centroids, z-scores
from standardisation, and the silhouette.

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| §1 Standardised feature space | Anomalies measured on `StandardScaler` features (values = z-scores) | **HW1 2.4** (Member C: StandardScaler) | ✅ Standardisation | Chosen over the power transform because it keeps extremes extreme |
| **§2 DBSCAN noise persistence (ε-exit)** | Smallest ε at which each player stops being noise | **HW4 Q2d-1** (smallest ε at which D9 stops being noise; the single controlling distance; joins as core or border) | ✅ Same reasoning, applied to every player | 🆕 Exact formula ε-exit = min(k-dist(p), min_q max(d(p,q), k-dist(q))), verified against sklearn |
| §2 Controlling neighbour | The player whose link ends a player's noise status | **HW4 Q2d-1** ("name the single controlling distance") | ✅ Controlling distance | Read as "closest comparable player" |
| §2 Noise survival curve | Players still noise vs. ε | **HW4 Q2d-3** (growing ε only adds neighbours: noise is absorbed, never created) | ✅ Monotonicity of DBSCAN in ε | — |
| **§3 Distance to the archetype centroid** | ‖xᵢ − μ_c(i)‖ for each player | **HW4 Q1** (inertia = Σ‖xᵢ − μ_c(i)‖²) · **HW4 Q2c / Q4c** (k-means forces outliers into a cluster) | ✅ Centroid distance / inertia | Per-player contribution to inertia used as an outlier measure |
| **§4 Extreme z-scores** | Max \|z\|, its feature, count of \|z\| ≥ 3 | **HW1 2.4** (StandardScaler) · HW1 1.3 (distribution tails) | 🔁 Standardised values as distances from the mean | Distinguishes extreme specialists from rare combinations |
| **§5a Measure correlations** | Spearman correlation of the three measures and the silhouette | HW1 1.4 (correlation matrix) | 🔁 Correlation analysis | Shows unusual ≠ in-between |
| **§5b–c Unicorns vs. border players** | Anomaly score (mean percentile rank) × silhouette quadrant | **HW4 Q1a** (silhouette s = (b − a)/max(a, b); near 0 = on the border) | ✅ Per-point silhouette | Four-way classification: unicorn / unique hybrid / border / typical |
| §5d Signatures & z-score fingerprints | Top-5% players with raw values and z-scores | HW1 1.3 (interpreting extremes) | 🔁 Explaining a value relative to the distribution | — |
| §6 Robustness | Other `min_samples`; power-transformed space | HW1 3.3 (comparison across preprocessing choices) · HW4 Q2d (effect of DBSCAN parameters) | 🔁 Sensitivity to preprocessing and parameters | Spearman ρ and top-k overlap |
| §7 Unicorns on the PCA map | PC1 × PC2 with categories highlighted | HW1 3.2 (PCA scatter coloured by category) | ✅ Same plot | — |

---

## 1f. `notebooks/06_Validation.ipynb` — section-by-section mapping

Validation is built from tools practised in the course: confusion-matrix metrics, KNN and decision trees with
stratified cross-validation, R² of a linear regression on one-hot dummies, the silhouette, PCA scatter plots and group
boxplots. The Adjusted Rand Index and its permutation baseline are supporting checks beyond the homework.

| Notebook section | What it does | Homework exercise | Concept reused | Adaptation for the NBA domain |
|---|---|---|---|---|
| §1 Load and align | Joins archetypes, power and raw features, anomalies and the full season table; asserts `player_id` alignment | HW1 1.1 (load & inspect) | ✅ Integrity check before analysis | Five tables from earlier notebooks, checked row by row |
| **§2 ARI vs. listed and played position** | Families / archetypes vs. `pos` and `pos_played`; listed vs. played position as a reference | *Supporting check beyond the homework* · related to HW4 Q4 (`same_partition`: comparing two partitions of the same points) | 🆕 Chance-corrected partition comparison | Two versions of ground truth |
| §2 Permutation baseline | 1,000 label shuffles (fixed seed, group sizes kept) → null distribution of ARI, permutation p-value | *Supporting check beyond the homework* | 🆕 Is the score better than chance? | Shows ARI is low but far above chance |
| **§3 Cross-tabulations** (counts, row %) | Archetype × listed / played position, family × position; purity per archetype | **HW2 Part 4** (reading a confusion matrix) | 🔁 Agreement / disagreement in a cross-tabulation | Rows read as roles, columns as positions |
| **§3 Confusion matrix of the majority mapping** | Each archetype "predicts" its most common position; accuracy, precision / recall / F1 per position, `classification_report`, `ConfusionMatrixDisplay` | **HW2 4.1 – 4.3** (confusion matrix, accuracy, precision, recall, F1) · **HW3 2.2** (`classification_report`) | ✅ Same metrics | Low score read as "positions hide several roles", not as a failed model |
| **§4 R² on one-hot dummies** | `LinearRegression` on `pd.get_dummies(drop_first=True)` of position / archetype / family for each of 13 style features; `r2_score`; grouped bar chart and mean R² | **HW2 2.2** (R² of a regression) · **HW1 2.2** (one-hot encoding) | ✅ Linear regression + R²; ✅ dummy encoding | R² = share of a feature's variance explained by group membership |
| §4 Held-out features | Same R² on 7 style descriptors never used for clustering | HW2 2.2 · HW3 4.1 (evaluate on data the model was not built from) | 🔁 Avoiding an optimistic, in-sample comparison | Held-out *features* rather than held-out rows |
| **§5 KNN and decision tree with stratified CV** | Predict listed position, archetype and family from the 13 power-transformed features; K = 1–30, `max_depth` = 1–15; `StratifiedKFold(5, shuffle=True, random_state=42)` + `cross_validate` (accuracy, macro F1) | **HW3 2.1 – 2.2** (KNN, evaluation) · **HW3 3.1** (decision tree, depth) · **HW3 4.1** (cross-validation) | ✅ Same models, parameter sweeps and CV | Archetype / family scores act as a ceiling (circular by construction); the position score is the finding |
| §5 Majority-class baseline | Always predict the most common class, scored with the same folds | HW3 2.2 / 4.1 (baseline to beat) | ✅ Baseline comparison | One baseline per target |
| **§6 Silhouette by labelling** | `silhouette_score` on the 13 features for listed position, played position, archetypes, families | **HW4 Q1** (silhouette) | ✅ Average silhouette | Applied to given labels (positions) as well as to clusters |
| §6 PCA maps side by side | PC1 × PC2 coloured by listed position and by archetype, group medians marked | **HW1 3.2** (PCA scatter coloured by category) | ✅ Same plot | Two colourings compared directly |
| **§7 Archetype profiles + value stats** | Median style features in real units; TS%, BPM, WS/48, VORP, minutes; BPM boxplots by archetype and by position; R² of value stats | **HW1 1.5** (boxplot of a variable by category) · HW2 2.2 (R²) | ✅ Group comparison with boxplots | Tests whether archetypes describe style or quality |
| §8 Unicorns and border players vs. positions | Anomaly category × position (rates) and × archetype | HW2 Part 4 (cross-tabulation) · HW4 Q1a (silhouette) | 🔁 Cross-tabulating two labelings | Links the anomaly notebook back to positions |
| §9 Export | `validation_summary.csv` (ARI, R², confusion-matrix scores, CV results, silhouettes) | — | — | — |
| §10 Conclusions | What positions capture and miss; continuum; taxonomy; limitations incl. circularity | HW1 3.3 / HW3 4.2 (comparison & recommendation) | ✅ Conclusions backed by numbers | Final summary of the whole project |

---

## 2. Concept index — where each course concept appears

| Course concept | Homework | Project location |
|---|---|---|
| Data loading & inspection | HW1 1.1 | 01_EDA §1 |
| Missing-value report | HW1 1.2 | 01_EDA §3 |
| Imputation strategies (Iterative, KNN, median) | HW1 2.3 | 01_EDA §3 (evaluated and rejected for structural NaNs) |
| Histograms & distribution shape | HW1 1.3 | 01_EDA §4 |
| Correlation heatmap | HW1 1.4 | 01_EDA §5 |
| Boxplot by category | HW1 1.5 | 01_EDA §6 · 06_Validation §7 (BPM by archetype and by position) |
| Dropping unusable columns / rows | HW1 2.1 | 01_EDA §2, §3 · 02 §2 |
| Categorical encoding (one-hot) | HW1 2.2 | Never a clustering input · 06_Validation §4 (one-hot dummies of position / archetype / family for R²) |
| Feature construction | HW1 2.2 | 02 §2 (`mid_share`, `corner3_rate`) |
| Scaling: StandardScaler / PowerTransformer / QuantileTransformer | HW1 2.4 | 01_EDA §4 · 02 §4 (3-scaler comparison) |
| Multicollinearity | HW2 2.3 – 2.4 | 01_EDA §5 · 02 §3 (VIF) |
| R² / variance explained | HW2 2.2 | 01_EDA §6 (η²) · 02 §3 (VIF = 1 / (1 − R²)) · 03_PCA §4 (η²) · 06_Validation §4 (`LinearRegression` + `r2_score` on group dummies, 13 style features) and §7 (value stats) |
| Distances & effect of scaling | HW3 1.1, 2.3 | 01_EDA §4 rationale · 02 §5 (distance & neighbour agreement) |
| Nearest-neighbour lookup | HW3 1.2 · HW4 Q3 | 02 §5b (similar players) |
| PCA: scree plot | HW1 3.1 | 03_PCA §2 (+ Kaiser & parallel analysis) |
| PCA: 2-component scatter coloured by category | HW1 3.2 | 03_PCA §4 (listed & played position) · §3 loadings · 06_Validation §6 (position vs. archetype, side by side) |
| PCA comparison across preprocessing choices | HW1 3.3 | 03_PCA §5 (PowerTransformer vs. StandardScaler) |
| k-means (implementation & behaviour) | HW4 Q4a | 04_Clustering §2, §4 |
| Choosing k: inertia vs. silhouette, business constraints | HW4 Q1 | 04_Clustering §2, §3 (3 families / 8 archetypes) |
| Per-point silhouette (border points) | HW4 Q1a | 04_Clustering §5a, §6 (border players) · 05_Anomalies §5 (unicorns vs. border players) |
| Average silhouette of a labelling | HW4 Q1 | 04_Clustering §2 (choosing k) · 06_Validation §6 (positions vs. archetypes vs. families) |
| Inertia as a sum of point-to-centroid distances | HW4 Q1 | 05_Anomalies §3 (distance to centroid as an outlier measure) |
| Standardised values as z-scores | HW1 2.4 | 05_Anomalies §4 (extreme z-scores, specialist vs. rare combination) |
| DBSCAN, core / border / noise, effect of ε | HW4 Q2, Q4b | 04_Clustering §8 (k-distance plot, ε × min_samples sweep, point types) · 05_Anomalies §2 (noise persistence, controlling distance) |
| k-means forces outliers into clusters vs. DBSCAN noise | HW4 Q2c, Q4c | 04_Clustering §8c–d · 05_Anomalies §3 (distance to centroid) |
| Cosine vs. Euclidean similarity, mean-centring | HW4 Q3 | 🔜 player-similarity / "closest comparable player" lookup |
| Confusion matrix, accuracy, precision, recall, F1, `classification_report` | HW2 Part 4 (4.1 – 4.3) · HW3 2.2 | 06_Validation §3 (archetype → majority position) · §5 (CV accuracy and macro F1) · §8 (anomaly category × position) |
| KNN and decision tree classifiers, parameter sweeps, majority baseline | HW3 2.1 – 2.2, 3.1 | 06_Validation §5 (predicting position vs. archetype vs. family from style) |
| Comparing two partitions (`same_partition` → ARI; ARI itself is a supporting check beyond the homework) | HW4 Q4 | 04_Clustering §7 · 06_Validation §2 (ARI vs. listed and played position, with permutation baseline) |
| Cross-validation & stability | HW3 4.1 | 04_Clustering §2, §5b (bootstrap stability) · 06_Validation §5 (`StratifiedKFold` + `cross_validate`) |
| Neural networks (ANN / CNN) | HW5 | Not used; outside the scope of this unsupervised project |

---

## 3. Notebook roadmap — homework links

| Notebook | Main steps | Primary homework links |
|---|---|---|
| `02_Features_Scaling.ipynb` ✅ done | 13-feature matrix, VIF check, StandardScaler vs. PowerTransformer vs. QuantileTransformer | HW1 2.1, 2.2, 2.4 · HW2 2.3 – 2.4 · HW3 1.1 – 1.2, 2.3 |
| `03_PCA.ipynb` ✅ done | Scree plot, parallel analysis, loadings, player map, biplot, robustness | HW1 3.1 – 3.3 · HW2 2.2 |
| `04_Clustering.ipynb` ✅ done | k-means (inertia, silhouette, bootstrap stability), 3 families / 8 archetypes, silhouette border players, DBSCAN sweep and noise | HW4 Q1, Q2, Q4 · HW3 4.1 |
| `05_Anomalies.ipynb` ✅ done | DBSCAN noise persistence, distance to centroid, extreme z-scores, silhouette quadrant | HW4 Q1, Q2, Q4 · HW1 2.4 |
| `06_Validation.ipynb` ✅ done | ARI + permutation baseline, crosstabs read as a confusion matrix, R² on group dummies, KNN / decision tree with stratified CV, silhouette by labelling, PCA maps, profiles and value stats, unicorns vs. positions, conclusions | HW2 2.2, Part 4 · HW3 2.1 – 2.2, 3.1, 4.1 · HW4 Q1 · HW1 1.5, 2.2, 3.2 |
