"""
Build the data/ folder for the NBA positionless-archetypes project.

Source: Basketball-Reference season tables, as mirrored (and kept up to date) in
the public GitHub repo sumitrodatta/bball-reference-datasets.
(Basketball-Reference / stats.nba.com / Kaggle were not reachable from the
environment the data was first pulled in, so a GitHub mirror is used.)

Usage
-----
    python scripts/fetch_data.py                    # clones the source repo (shallow) into a temp dir
    python scripts/fetch_data.py --src /path/Data   # use an existing local copy of the Data/ folder
    python scripts/fetch_data.py --season 2026      # season = year the season ENDS (2026 -> 2025-26)

Outputs
-------
data/raw/<table>_<season>.csv      season-filtered copy of each source table (every team stint)
data/nba_<yy-yy>_players.csv       one row per player, all tables merged
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/sumitrodatta/bball-reference-datasets"

# source file name -> short name used in output files / column prefixes
TABLES = {
    "Player Per Game": "per_game",
    "Per 100 Poss": "per_100",
    "Advanced": "advanced",
    "Player Totals": "totals",
    "Player Shooting": "shooting",
    "Player Play By Play": "play_by_play",
}
META = ["season", "lg", "player", "player_id", "age", "team", "pos", "g", "gs"]
MULTI_TEAM = {"TOT", "2TM", "3TM", "4TM", "5TM"}  # season-aggregate rows for traded players


def get_source_dir(src: str | None) -> Path:
    if src:
        return Path(src)
    tmp = Path(tempfile.mkdtemp(prefix="bbref_"))
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(tmp / "repo")], check=True)
    return tmp / "repo" / "Data"


def one_row_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the season-total row for traded players (2TM/3TM...), else the single team row."""
    df = df.copy()
    df["is_multi"] = df["team"].isin(MULTI_TEAM)
    traded = df.loc[df["is_multi"], "player_id"].unique()
    keep = df["is_multi"] | ~df["player_id"].isin(traded)
    out = df[keep].drop(columns="is_multi")
    assert out["player_id"].is_unique, "expected exactly one row per player"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="path to an existing bball-reference-datasets/Data folder")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = ap.parse_args()

    src, out = get_source_dir(args.src), Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    merged = None
    for fname, short in TABLES.items():
        df = pd.read_csv(src / f"{fname}.csv")
        df = df[(df["season"] == args.season) & (df["lg"] == "NBA")]
        df.to_csv(out / "raw" / f"{short}_{args.season}.csv", index=False)
        print(f"{short:13s} {len(df):4d} rows, {df['player_id'].nunique()} players")

        df = one_row_per_player(df)
        if merged is None:
            merged = df
            continue
        # 'mp' means total minutes in every table except per_game (mp_per_game) -> keep first copy only
        dup = [c for c in df.columns if c in merged.columns and c != "player_id"]
        # same-named stat columns (e.g. fg_percent) are identical across tables -> drop them;
        # only keep a suffixed copy if the values genuinely differ
        aligned = merged.set_index("player_id").reindex(df["player_id"])
        clash = [c for c in dup if c not in META + ["mp"]
                 and not df[c].reset_index(drop=True).equals(aligned[c].reset_index(drop=True))]
        df = df.rename(columns={c: f"{c}_{short}" for c in clash}).drop(
            columns=[c for c in dup if c not in clash])
        merged = merged.merge(df, on="player_id", how="left", validate="one_to_one")

    yy = f"{args.season - 1}-{str(args.season)[2:]}"
    path = out / f"nba_{yy.replace('-', '_')}_players.csv"
    merged.to_csv(path, index=False)
    print(f"\nwrote {path}  shape={merged.shape}")


if __name__ == "__main__":
    main()
