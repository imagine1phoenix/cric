"""
conditions_features.py — Weather and pitch condition features.

Uses WeatherService for live/forecast weather and historical venue stats
for pitch type estimation.
"""

import logging
import numpy as np
import pandas as pd
from collections import defaultdict

logger = logging.getLogger("criccric")


def compute_venue_pitch_features(df):
    """
    Compute venue pitch-type proxy features from historical data.

    Adds (all leakage-safe — expanding/cumulative with shift(1)):
      - venue_avg_wickets_per_match
      - venue_avg_run_rate
      - venue_pace_vs_spin (placeholder: uses wickets per innings as proxy)

    Parameters
    ----------
    df : DataFrame sorted by date

    Returns
    -------
    df : DataFrame with new columns
    """
    df = df.copy()

    # Total wickets per match
    df["_total_wkts"] = df["innings1_wickets"] + df["innings2_wickets"]
    df["_total_runs"] = df["innings1_total_runs"] + df["innings2_total_runs"]

    # Venue average wickets per match (expanding, shifted)
    df["venue_avg_wickets"] = (
        df.groupby("venue")["_total_wkts"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(df["_total_wkts"].mean())
        .astype(np.float32)
    )

    # Venue average run rate (runs per wicket as proxy)
    df["venue_avg_rpw"] = (
        df.groupby("venue")["_total_runs"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(df["_total_runs"].mean())
        .astype(np.float32)
    )

    # Batting-friendly index: higher = more runs per wicket
    safe_wkts = df["venue_avg_wickets"].clip(lower=1)
    df["venue_batting_index"] = (df["venue_avg_rpw"] / safe_wkts).astype(np.float32)

    df.drop(["_total_wkts", "_total_runs"], axis=1, inplace=True)
    return df, ["venue_avg_wickets", "venue_avg_rpw", "venue_batting_index"]


def compute_dew_feature(df):
    """
    Compute dew factor for day-night matches.

    Approximate: if match is day-night AND in known dew-prone cities.
    Since we can't detect day-night from all datasets, we use "is_day_night"
    column if it exists, or fallback to always 0.

    Returns
    -------
    df with 'dew_factor' column
    """
    from services.weather import DEW_CITIES

    df = df.copy()

    if "is_day_night" in df.columns:
        df["dew_factor"] = 0
        for idx in df.index:
            city = str(df.loc[idx, "city"]).strip()
            if df.loc[idx, "is_day_night"] == 1:
                if any(dc.lower() in city.lower() for dc in DEW_CITIES):
                    df.at[idx, "dew_factor"] = 1
    else:
        df["dew_factor"] = 0

    df["dew_factor"] = df["dew_factor"].astype(np.int8)
    return df, ["dew_factor"]
