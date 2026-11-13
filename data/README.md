# Data — NBA 2025-26 regular season (Basketball-Reference)

**Source:** Basketball-Reference season tables, via the public mirror
[sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets)
(snapshot of 2026-11-02). `season = 2026` means the 2025-26 season, the most recent
complete one. Rebuild with `python scripts/fetch_data.py`.

## Files
| File | Rows | What it is |
|---|---|---|
| `nba_2025_26_players.csv` | 582 players × 127 cols | **Main file.** All tables merged, one row per player |
| `raw/per_game_2026.csv` | 733 | Per-game box score |
| `raw/per_100_2026.csv` | 733 | Per-100-possessions box score + ORtg/DRtg |
| `raw/advanced_2026.csv` | 733 | TS%, 3PAr, FTr, ORB/DRB/TRB%, AST%, STL%, BLK%, TOV%, USG%, WS, BPM, VORP |
| `raw/totals_2026.csv` | 733 | Season totals |
| `raw/shooting_2026.csv` | 733 | Shot distance & zone mix (0-3 ft, 3-10, 10-16, 16-3P, 3P), % assisted, dunks, corner 3s |
| `raw/play_by_play_2026.csv` | 733 | % of minutes at PG/SG/SF/PF/C, on/off +/-, turnover & foul types |

`raw/` files keep every team stint; a player traded mid-season has one row per team **plus**
a `2TM`/`3TM`/`4TM` season-total row. The merged file keeps only the season-total row for those players.

## Notes for modelling
- `pos` = listed position (PG 97, SG 159, SF 112, PF 106, C 107) → validation target #1 (never a model input).
- `pg_percent` … `c_percent` = actual minutes played at each slot; notebook 01 turns them into `pos_played` (the slot
  with the most minutes) → validation target #2.
- Missing values are **structural**, not random: e.g. `x3p_percent` is NaN when a player took 0 threes,
  `fg_percent_from_x16_3p_range` when the player took no long twos. With a minutes filter (≥500 MP → 379 players)
  only shooting-% columns have any NaNs.
- Many low-minute players have extreme rate stats (small samples) — filter on `mp` before clustering.

## Processed files (`processed/`, written by the notebooks)
Each file holds only what a later notebook reads (or, for the last two, the final results). All are one row per
player, in the same order, keyed by `player_id`.

| File | Written by | Read by | Contents |
|---|---|---|---|
| `nba_2025_26_mp500.csv` | 01 | 02, 06 | The 379 players with ≥ 500 minutes, all original columns + `pos_played` |
| `features_raw.csv` | 02 | 04, 05, 06 | Identifiers, `pos`, `pos_played` + the 13 style features in real units |
| `features_power.csv` | 02 | 03, 04, 05, 06 | Same, Yeo-Johnson power-transformed (clustering space) |
| `features_standard.csv` | 02 | 03, 04, 05 | Same, standardised (anomaly space, robustness checks) |
| `pca_scores.csv` | 03 | 04 | Identifiers + PC1–PC5 scores |
| `archetypes.csv` | 04 | 05, 06 | Identifiers, PC1–PC3, `family` (k = 3), `archetype` (k = 8), `silhouette`, `border_player` |
| `anomalies.csv` | 05 | 06 | `player_id`, `player` + anomaly measures, `anomaly_score`, `category`, `unicorn_type`, `signature` |
| `validation_summary.csv` | 06 | — | ARI with permutation baseline, R² tables, confusion-matrix scores, CV results, silhouettes |
