# Positionless NBA Player Archetypes — Project Overview

## 1. Purpose

The NBA still describes players with five positions (PG, SG, SF, PF, C), but the modern game no longer works that
way: centres run the offense, forwards shoot half their shots from three, and "point guard" describes a skill set more
than a spot on the floor.

**The question:** if we ignore the position labels and group players only by *what they actually do on the court*,
which groups emerge, how well do they match the official positions, and which players fit no group at all?

**The goal:** an unsupervised-learning pipeline that turns box-score and shooting data into data-driven **player
archetypes**, identifies **one-of-a-kind players**, and measures how much the five-position system misses.

**Constraint:** the project deliberately uses **only techniques practised in the course homework** (EDA, scaling,
PCA, k-means, DBSCAN, silhouette, z-scores). How each step maps to the homework is documented in
[`docs/HOMEWORK_MAPPING.md`](docs/HOMEWORK_MAPPING.md); the notebooks themselves are written as a standalone project.

---

## 2. Data

| Item | Detail |
|---|---|
| Source | Basketball-Reference season tables, via the public mirror `sumitrodatta/bball-reference-datasets` (built by `scripts/fetch_data.py`) |
| Season | **2025-26 regular season** (the most recent complete season) |
| Tables | per-game, per-100 possessions, advanced, season totals, shot-zone shooting, play-by-play position estimates |
| Raw size | 582 players × 127 columns, one row per player (traded players use their season-total row) |
| Analysed population | **379 players with ≥ 500 minutes** (low-minute players have unreliable rate stats) |
| Ground truth for validation | listed position `pos`, and `pos_played` = position with the most minutes (play-by-play) |

Details: [`data/README.md`](data/README.md).

---

## 3. The pipeline

| # | Notebook | Purpose | Main output |
|---|---|---|---|
| 01 | `01_EDA.ipynb` | Understand the data and make the preprocessing decisions | `nba_2025_26_mp500.csv` |
| 02 | `02_Features_Scaling.ipynb` | Build a clean, non-redundant, scaled feature matrix | `features_raw / _power / _standard.csv` |
| 03 | `03_PCA.ipynb` | Find and name the main axes of playing style | `pca_scores.csv`, `pca_loadings.csv` |
| 04 | `04_Clustering.ipynb` | Group players into families and archetypes | `archetypes.csv`, `archetype_profiles.csv` |
| 05 | `05_Anomalies.ipynb` | Find statistical unicorns and separate them from border players | `anomalies.csv` |
| 06 | `06_Validation.ipynb` | Validate the groups against listed and played positions; what archetypes add | `validation_summary.csv` |

### 01 — Exploratory Data Analysis
- Inspection (`info`, `head`), **500-minute filter** (582 → 379 players).
- **Missing values are structural**: shooting percentages are undefined when a player took zero attempts, so they are
  excluded rather than imputed (median imputation would make non-shooting centres "35% three-point shooters").
- **Skewness** of the role stats (13 of 23 candidates have |skew| > 1), which motivates a power transform.
- **Correlation heatmap**: removes redundant groups; per-100 stats preferred over per-game (per-game stats track minutes).
- **Boxplots by position**: only centres are clearly distinct; usage is almost unrelated to position (η² = 0.05).
  Listed position matches the most-played position for 77% of players.

### 02 — Feature Selection & Scaling
- **13 style features in four dimensions** (no missing values, no imputation):

  | Dimension | Features |
  |---|---|
  | Shot diet | 3PA rate, corner-3 rate, mid-range share, rim share, free-throw rate |
  | Creation | usage, AST%, TOV%, % of made 2s assisted |
  | Rebounding | ORB%, DRB% |
  | Defense | BLK%, STL% |

- Engineered features: `mid_share` (merged mid-range zones), `corner3_rate` (corner threes as a share of all shots).
- Redundancy check: the naive zone set is perfectly collinear (VIF ≈ 88,000); the selected set has max VIF 8.8.
- **Scaler comparison** (StandardScaler, PowerTransformer, QuantileTransformer): **PowerTransformer** for PCA and
  clustering, **StandardScaler** for anomaly detection; QuantileTransformer rejected (ties create artificial outliers).

### 03 — PCA
- **3 components carry real structure** (69% of variance):

  | Axis | Share | Meaning |
  |---|---|---|
  | PC1 | 32% | Interior ↔ Perimeter |
  | PC2 | 26% | On-ball creator ↔ Off-ball finisher |
  | PC3 | 11% | Disruptor / connector ↔ Self-sufficient scorer |

- Position explains PC1 well (η² ≈ 0.6) but **PC2 and PC3 hardly at all** (η² ≤ 0.16): the five positions are
  essentially a one-dimensional (size) description of a three-dimensional reality.
- The axes are identical under both scalers (|cos| ≈ 0.996).

### 04 — Clustering
- **k-means** (primary model), k chosen with **inertia (elbow), silhouette and bootstrap stability**.
- Silhouette is low at every k (≤ 0.21): roles form a **continuum**, not islands.
- **Two-level taxonomy**:
  - **3 families** (statistical optimum, stability ≈ 0.84): On-Ball Creator · Off-Ball Perimeter · Big.
  - **8 archetypes** (useful granularity): Primary Creator · Shot-Making Guard · Secondary Scoring Wing · Spot-Up Floor
    Spacer · Defensive Disruptor · Stretch Forward · Hub Big · Rim-Running Big.
- Per-archetype reliability (bootstrap Jaccard): Rim-Running Big highly stable (0.86), Secondary Scoring Wing unstable
  (0.46, the "versatile / no specialty" middle of the map).
- **Border players**: 78 players with silhouette < 0.05, each with a "second archetype".
- **DBSCAN** (ε × `min_samples` sweep): never more than two sizeable dense regions (rim-running bigs vs. everyone else);
  at a larger ε only 15 players remain noise.

### 05 — Anomaly Detection
- Three measures on the standardised features:
  1. **DBSCAN noise persistence**: the exact smallest ε at which each player stops being noise.
  2. **Distance to the archetype centroid**.
  3. **Extreme z-scores** (which skill is unusual).
- Combined with the **silhouette** into four groups:

  | Group | n | Examples |
  |---|---|---|
  | Statistical unicorns (top 5%, clearly in one archetype) | 17 | Dwight Powell, DeRozan, Wembanyama, McConnell, Draymond Green, Jokić, Giannis, Dončić |
  | Unique hybrids (top 5%, between archetypes) | 2 | Paul Reed, Josh Green |
  | Border players (in-between, ordinary profile) | 76 | Brunson, Curry, Bam Adebayo |
  | Typical | 284 | |

- **Being unusual and being in-between are different things**: the silhouette is essentially uncorrelated with the
  anomaly measures.

### 06 — Validation
- **Adjusted Rand Index** against positions is low but far above chance:

  | Comparison | ARI | Shuffled-label baseline (1,000 shuffles) |
  |---|---|---|
  | 8 archetypes vs. listed / played position | 0.14 / 0.16 | mean ≈ 0.000, max ≈ 0.02 (z ≈ 41–44) |
  | 3 families vs. listed / played position | 0.17 / 0.18 | mean ≈ 0.000, max ≈ 0.02 (z ≈ 44–52) |
  | Listed vs. played position (reference) | 0.54 | — |

  Archetypes are *related to* positions but are not a relabelling of them.
- **Cross-tabulations**: only Rim-Running Big is essentially one position (90% listed C); size-weighted archetype
  purity is 50%. Centres split into Rim-Running Big / Hub Big / Stretch Forward; SG, SF and PF each spread over five
  archetypes; half of the Primary Creators are not point guards.
- **η² (variance explained) across the 13 style features**: listed position **0.27**, 3 families 0.39,
  8 archetypes **0.54** (chance level ≈ 0.01–0.02). Position explains the size features (ORB% 0.55, DRB% 0.53,
  BLK% 0.46) and is slightly better than archetypes on DRB%, but almost none of usage (0.05 vs. 0.57), steals,
  mid-range share or turnovers. The gain holds on 7 held-out style descriptors never used for clustering
  (0.28 → 0.53).
- **Style, not quality**: archetype explains only about 21% of the variance in BPM (position 7%); every archetype
  contains players with a negative BPM.
- **Unicorns vs. positions**: 10 of the 17 statistical unicorns are centres; border players are most common among
  small forwards (29%) and shooting guards (23%).

---

## 4. Tools

### Language and libraries

| Tool | Used for |
|---|---|
| **Python 3** + **Jupyter** | All analysis, one notebook per stage |
| **pandas**, **NumPy** | Loading, merging, filtering, feature engineering |
| **matplotlib**, **seaborn** | Histograms, boxplots, heatmaps, scatter maps, scree plots |
| **SciPy** | Skewness / kurtosis, Spearman correlation, pairwise distances |
| **scikit-learn** | Scaling, PCA, k-means, DBSCAN, nearest neighbours, silhouette, Adjusted Rand Index |

Dependencies: [`requirements.txt`](requirements.txt).

### Methods used

| Area | Methods |
|---|---|
| EDA | `info` / `head`, missing-value report, histograms, correlation heatmap, boxplots |
| Preprocessing | minutes filter, structural-missingness handling, feature engineering, `StandardScaler`, `PowerTransformer` (Yeo-Johnson), `QuantileTransformer` (compared) |
| Dimensionality reduction | PCA: scree plot, loadings, 2-D projection, biplot |
| Clustering | **k-means** (inertia, silhouette, centroids), **DBSCAN** (ε, `min_samples`, core / border / noise) |
| Cluster quality | silhouette (average and per player), bootstrap stability |
| Anomaly detection | DBSCAN noise persistence, distance to centroid, z-scores |
| Comparison / validation | Adjusted Rand Index (ARI) with a permutation baseline, cross-tabulations (counts, row and column %), η² per feature, group medians and boxplots |

**Supporting checks beyond the homework** (used to verify decisions, not as models): variance inflation factor (VIF),
parallel analysis for PCA, η² (variance explained by position), bootstrap ARI / Jaccard stability, k-distance plot.

**Deliberately not used** (outside the course): Gaussian Mixture Models, hierarchical / Ward clustering, Isolation
Forest, Local Outlier Factor, Calinski-Harabasz and Davies-Bouldin indices.

---

## 5. Repository structure

```
NBA_ModernAge_Position/
├── PROJECT_OVERVIEW.md          ← this file
├── requirements.txt
├── scripts/
│   └── fetch_data.py            builds data/ from the Basketball-Reference mirror
├── data/
│   ├── README.md                data dictionary and notes
│   ├── nba_2025_26_players.csv  582 players, all tables merged
│   ├── raw/                     season-filtered source tables (every team stint)
│   └── processed/               outputs of each notebook (filtered data, features, PCA, archetypes, anomalies, validation)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Features_Scaling.ipynb
│   ├── 03_PCA.ipynb
│   ├── 04_Clustering.ipynb
│   ├── 05_Anomalies.ipynb
│   └── 06_Validation.ipynb
└── docs/
    └── HOMEWORK_MAPPING.md      notebook steps ↔ course homework exercises
```

## 6. How to run

```bash
pip install -r requirements.txt
```

```bash
python scripts/fetch_data.py
```

Then run the notebooks in order (01 → 06); each one reads the previous one's output from `data/processed/`. The data
is already included in the repository, so the fetch step is only needed to rebuild it.

---

## 7. Key takeaways

1. **Roles are continuous.** There are no empty gaps between most archetypes (low silhouette, a single connected DBSCAN
   cloud). Clusters partition a continuum.
2. **Positions capture one dimension, size.** On-ball creation, the second-largest source of difference between
   players, is almost invisible to position labels: position explains 5% of the variance in usage, archetypes 57%.
   Over all 13 style features, archetypes explain about twice as much variance as positions (η² 0.54 vs. 0.27).
3. **"Centre" and "guard" each hide several jobs.** For example, Rim-Running Big, Hub Big and Stretch Forward for
   centres.
4. **Only the rim-running big is a sharply separated role.** It is the most stable archetype and the only one DBSCAN
   isolates as its own dense region.
5. **Unusual ≠ in-between.** Unicorns are extreme versions of a role, mostly centres; border players are ordinary
   profiles that mix two neighbouring roles, mostly wings.
6. **Archetypes are related to positions, not a relabelling.** ARI 0.14–0.18, far above the chance baseline but far
   below the 0.54 agreement between the two position labels themselves.
7. **Archetypes describe style, not quality.** Every archetype contains both strong and weak players.

**Limitations:** box-score data under-describes defence; one season only; a 500-minute cut-off; k = 8 was chosen
partly for interpretability (k = 3 is the statistical optimum); the η² comparison on the clustering features favours
the archetypes (held-out features reduce but do not remove this).

**Next step:** the core pipeline (01 → 06) is complete. Possible extensions: a "closest comparable player" lookup
(cosine vs. Euclidean similarity), repeating the analysis on earlier seasons to track how roles change, and a short
written report or slide summary of the findings.
