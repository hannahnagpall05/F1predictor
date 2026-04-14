"""
features.py  (v2 — richer feature engineering)
===============================================
Builds the feature table used for training.

New features added over v1:
  5.  driver_recent_form      → avg finish position in last 3 races (lower = better)
  6.  driver_season_win_pct   → wins / races so far this season
  7.  constructor_season_wins → team's wins so far this season
  8.  circuit_podium_rate     → driver's historical top-3 rate at this circuit
  9.  circuit_type_enc        → encoded circuit type (street / power / technical)
  10. grid_squared            → grid_position^2  (pole advantage is non-linear)
  11. driver_poles_pct        → % of races this season the driver started on pole
  12. dnf_rate                → driver's historical DNF rate (reliability proxy)
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Circuit type mapping — encodes track character
CIRCUIT_TYPE = {
    "monaco":        "street",   "baku":          "street",
    "marina_bay":    "street",   "vegas":          "street",
    "americas":      "technical","hungaroring":    "technical",
    "suzuka":        "technical","silverstone":    "technical",
    "spa":           "technical","zandvoort":      "technical",
    "monza":         "power",    "jeddah":         "power",
    "albert_park":   "power",    "bahrain":        "technical",
    "catalunya":     "technical","villeneuve":     "street",
    "red_bull_ring": "power",    "imola":          "technical",
    "interlagos":    "technical","rodriguez":      "technical",
    "yas_marina":    "technical","losail":         "power",
    "shanghai":      "technical","zolder":         "technical",
    "magny_cours":   "technical","nurburgring":    "technical",
    "indianapolis":  "power",    "sepang":         "technical",
    "istanbul":      "technical","fuji":           "technical",
    "valencia":      "street",   "korea":          "street",
    "delhi":         "street",   "sochi":          "street",
    "portimao":      "technical","mugello":        "technical",
    "paulo_afonso":  "street",
}
CIRCUIT_TYPE_MAP = {"technical": 0, "street": 1, "power": 2}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "race_results.csv"))


def add_cumulative_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["driver_champ_points"] = (
        df.groupby(["season", "driver_id"])["points"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    df["team_champ_points"] = (
        df.groupby(["season", "constructor_id"])["points"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    return df


def add_circuit_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["won"] = (df["finish_pos"] == 1).astype(int)
    df["circuit_starts_before"] = df.groupby(["driver_id", "circuit_id"]).cumcount()
    df["circuit_wins_before"] = (
        df.groupby(["driver_id", "circuit_id"])["won"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    df["circuit_win_rate"] = 0.0
    mask = df["circuit_starts_before"] > 0
    df.loc[mask, "circuit_win_rate"] = (
        df.loc[mask, "circuit_wins_before"] / df.loc[mask, "circuit_starts_before"]
    )
    return df


def add_circuit_podium_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Historical top-3 rate at each circuit — more signal than wins alone."""
    df["podium"] = (df["finish_pos"] <= 3).astype(int)
    df["circuit_podiums_before"] = (
        df.groupby(["driver_id", "circuit_id"])["podium"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    df["circuit_podium_rate"] = 0.0
    mask = df["circuit_starts_before"] > 0
    df.loc[mask, "circuit_podium_rate"] = (
        df.loc[mask, "circuit_podiums_before"] / df.loc[mask, "circuit_starts_before"]
    )
    return df


def add_recent_form(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Rolling average finish position over the last `window` races.
    Lower = better. Uses only past races (shift before rolling).
    """
    df = df.sort_values(["driver_id", "season", "round"]).reset_index(drop=True)
    df["driver_recent_form"] = (
        df.groupby("driver_id")["finish_pos"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    ).fillna(10.0)  # default: mid-pack if no history
    return df


def add_season_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Wins-per-race so far this season for driver and constructor."""
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    df["driver_season_wins_before"] = (
        df.groupby(["season", "driver_id"])["won"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    df["driver_season_races_before"] = (
        df.groupby(["season", "driver_id"]).cumcount()
    )
    df["driver_season_win_pct"] = 0.0
    mask = df["driver_season_races_before"] > 0
    df.loc[mask, "driver_season_win_pct"] = (
        df.loc[mask, "driver_season_wins_before"]
        / df.loc[mask, "driver_season_races_before"]
    )

    df["constructor_season_wins"] = (
        df.groupby(["season", "constructor_id"])["won"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    return df


def add_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical DNF rate as a reliability proxy.
    A DNF is any status that is NOT 'Finished' and NOT a +N laps string.
    """
    finished_statuses = {"Finished"}
    def is_dnf(status):
        s = str(status)
        if s in finished_statuses:
            return 0
        if s.startswith("+") and "Lap" in s:
            return 0
        return 1

    df["is_dnf"] = df["status"].apply(is_dnf)
    df["dnf_rate"] = (
        df.groupby("driver_id")["is_dnf"]
        .transform(lambda x: x.shift(1).fillna(0).expanding().mean())
    ).fillna(0.1)
    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Non-linear grid encoding and pole flag."""
    df["grid_squared"]  = df["grid_position"] ** 2
    df["is_pole"]       = (df["grid_position"] == 1).astype(int)

    df["driver_poles_before"] = (
        df.groupby(["season", "driver_id"])["is_pole"]
        .transform(lambda x: x.shift(1).fillna(0).cumsum())
    )
    df["driver_season_poles_pct"] = 0.0
    mask = df["driver_season_races_before"] > 0
    df.loc[mask, "driver_season_poles_pct"] = (
        df.loc[mask, "driver_poles_before"]
        / df.loc[mask, "driver_season_races_before"]
    )
    return df


def add_circuit_type(df: pd.DataFrame) -> pd.DataFrame:
    df["circuit_type_enc"] = (
        df["circuit_id"]
        .map(CIRCUIT_TYPE)
        .map(CIRCUIT_TYPE_MAP)
        .fillna(0)
        .astype(int)
    )
    return df


def add_constructor_avg_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Constructor's average qualifying position this season so far."""
    df = df.sort_values(["season", "round"]).reset_index(drop=True)
    df["constructor_avg_grid"] = (
        df.groupby(["season", "constructor_id"])["grid_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    ).fillna(10.0)
    return df


def build_feature_table() -> pd.DataFrame:
    print("Loading raw race results...")
    df = load_raw()
    # Rename 'grid' -> 'grid_position' immediately so every helper can use it
    df = df.rename(columns={"grid": "grid_position"})
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    print("Building cumulative championship points...")
    df = add_cumulative_points(df)

    print("Adding circuit win rate...")
    df = add_circuit_win_rate(df)

    print("Adding circuit podium rate...")
    df = add_circuit_podium_rate(df)

    print("Adding recent form (last 3 races)...")
    df = add_recent_form(df, window=3)

    print("Adding season win % and constructor wins...")
    df = add_season_win_pct(df)

    print("Adding DNF rate...")
    df = add_dnf_rate(df)

    print("Adding grid features (squared, pole flag)...")
    df = add_grid_features(df)

    print("Adding circuit type encoding...")
    df = add_circuit_type(df)

    print("Adding constructor avg qualifying position...")
    df = add_constructor_avg_grid(df)

    df = df[df["grid_position"] > 0].copy()

    feature_cols = [
        "season", "round", "race_name", "circuit_id",
        "driver_id", "constructor_id",
        # Core
        "grid_position", "grid_squared", "is_pole",
        "driver_champ_points", "team_champ_points",
        # Circuit-specific
        "circuit_win_rate", "circuit_podium_rate", "circuit_type_enc",
        # Form
        "driver_recent_form", "driver_season_win_pct",
        "driver_season_poles_pct", "constructor_season_wins",
        "constructor_avg_grid",
        # Reliability
        "dnf_rate",
        # Target
        "won",
    ]

    # Keep only cols that exist (guard against edge cases)
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df[feature_cols]

    path = os.path.join(DATA_DIR, "features.csv")
    df.to_csv(path, index=False)
    print(f"\n✅ Feature table saved → {path}")
    print(f"   {len(df):,} rows  |  {int(df['won'].sum())} race winners  |  {df.shape[1]-7} model features")
    return df


if __name__ == "__main__":
    build_feature_table()
