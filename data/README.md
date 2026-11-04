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
- `pos` = listed position (PG 97, SG 159, SF 112, PF 106, C 107) → ground truth #1 for ARI.
- `pg_percent` … `c_percent` = actual minutes played at each slot → richer ground truth #2.
- Missing values are **structural**, not random: e.g. `x3p_percent` is NaN when a player took 0 threes,
  `fg_percent_from_x16_3p_range` when the player took no long twos. With a minutes filter (≥500 MP → 379 players)
  only shooting-% columns have any NaNs.
- Many low-minute players have extreme rate stats (small samples) — filter on `mp` before clustering.
