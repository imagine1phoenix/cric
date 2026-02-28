"""
csv_parser.py — Adapter to normalize Kaggle CSV datasets into the standard DataFrame schema.

Scans data/raw_csv/ for directories containing matches.csv and deliveries.csv,
normalizes their column names, aggregates innings scores, and returns a unified DataFrame.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("criccric")

# Desired standard schema for our application's json parser output
EXPECTED_COLUMNS = [
    "match_id", "date", "team1", "team2", "venue", "city", 
    "toss_winner", "toss_decision", "innings1_total_runs", "innings1_wickets",
    "innings2_total_runs", "innings2_wickets", "winner", "match_type",
    "gender", "team_type", "league", "season"
]

def parse_all_csvs(raw_csv_dir="data/raw_csv"):
    """Parse all Kaggle CSV datasets and return a single normalized DataFrame."""
    if not os.path.exists(raw_csv_dir):
        logger.warning(f"CSV directory {raw_csv_dir} does not exist.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    dataset_dirs = [d for d in glob.glob(os.path.join(raw_csv_dir, "*")) if os.path.isdir(d)]
    all_dfs = []

    for ddir in dataset_dirs:
        dataset_name = os.path.basename(ddir)
        logger.info(f"Parsing CSV dataset: {dataset_name}")
        
        # Heuristically find the matches and deliveries files
        csv_files = glob.glob(os.path.join(ddir, "*.csv"))
        
        matches_file = next((f for f in csv_files if "match" in os.path.basename(f).lower()), None)
        deliveries_file = next((f for f in csv_files if "deliver" in os.path.basename(f).lower() or "ball" in os.path.basename(f).lower()), None)

        if not matches_file:
            logger.warning(f"Could not find matches CSV in {dataset_name}. Skipping.")
            continue

        try:
            df_m = pd.read_csv(matches_file, low_memory=False)
            df_norm = _normalize_matches(df_m, dataset_name)
            
            if deliveries_file:
                df_d = pd.read_csv(deliveries_file, low_memory=False)
                df_norm = _aggregate_deliveries(df_norm, df_d)
            else:
                # If no deliveries, assume default avg values so they don't crash
                df_norm["innings1_total_runs"] = 150
                df_norm["innings1_wickets"] = 5
                df_norm["innings2_total_runs"] = 140
                df_norm["innings2_wickets"] = 6
                
            # Filter valid outcomes
            df_norm = df_norm.dropna(subset=["winner", "team1", "team2"])
            df_norm = df_norm[~df_norm["winner"].str.lower().isin(["no result", "tie", "draw", "abandoned"])]
            
            # Ensure all expected columns exist
            for col in EXPECTED_COLUMNS:
                if col not in df_norm.columns:
                    # Fill missing categoricals with Unknown, numerics with 0
                    if col in ["innings1_total_runs", "innings2_total_runs", "innings1_wickets", "innings2_wickets"]:
                        df_norm[col] = 0
                    else:
                        df_norm[col] = "Unknown"
                        
            all_dfs.append(df_norm[EXPECTED_COLUMNS])
            logger.info(f"  -> Extracted {len(df_norm)} valid matches from {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to parse {dataset_name}: {e}")

    if not all_dfs:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Drop ultimate duplicates
    final_df = final_df.drop_duplicates(subset=["date", "team1", "team2"])
    logger.info(f"CSV Parsing Complete. Total normalized CSV matches: {len(final_df)}")
    
    return final_df


def _normalize_matches(df, dataset_name):
    """Map wildly varying Kaggle column names to our standard ones."""
    # Lowercase everything for easier matching
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    rename_map = {}
    
    # Match ID
    if "id" in df.columns: rename_map["id"] = "match_id"
    elif "match_id" in df.columns: rename_map["match_id"] = "match_id"
    
    # Teams
    if "team1" in df.columns: rename_map["team1"] = "team1"
    elif "team 1" in df.columns: rename_map["team 1"] = "team1"
    
    if "team2" in df.columns: rename_map["team2"] = "team2"
    elif "team 2" in df.columns: rename_map["team 2"] = "team2"
    
    # Venue / City
    if "venue" in df.columns: rename_map["venue"] = "venue"
    elif "ground" in df.columns: rename_map["ground"] = "venue"
        
    if "city" in df.columns: rename_map["city"] = "city"
    
    # Toss
    if "toss_winner" in df.columns: rename_map["toss_winner"] = "toss_winner"
    if "toss_decision" in df.columns: rename_map["toss_decision"] = "toss_decision"
    elif "toss_name" in df.columns: rename_map["toss_name"] = "toss_decision"
    
    # Winner
    if "winner" in df.columns: rename_map["winner"] = "winner"
    elif "winning_team" in df.columns: rename_map["winning_team"] = "winner"
    
    # Date
    if "date" in df.columns: rename_map["date"] = "date"
    elif "start_date" in df.columns: rename_map["start_date"] = "date"
    
    # Season
    if "season" in df.columns: rename_map["season"] = "season"
    
    df = df.rename(columns=rename_map)
    
    # Fallbacks and generated constants
    if "match_id" not in df.columns:
        df["match_id"] = [f"{dataset_name}_{i}" for i in range(len(df))]
        
    if "city" not in df.columns:
        df["city"] = df.get("venue", "Unknown")
        
    if "league" not in df.columns:
        # Try to infer league from dataset name
        if "ipl" in dataset_name.lower(): df["league"] = "Indian Premier League"
        elif "bbl" in dataset_name.lower(): df["league"] = "Big Bash League"
        elif "psl" in dataset_name.lower(): df["league"] = "Pakistan Super League"
        elif "world-cup" in dataset_name.lower(): df["league"] = "T20 World Cup"
        elif "womens" in dataset_name.lower(): df["league"] = "Women's Internationals"
        else: df["league"] = "Internationals"
        
    if "match_type" not in df.columns:
        if "odi" in dataset_name.lower(): df["match_type"] = "ODI"
        else: df["match_type"] = "T20"
        
    if "gender" not in df.columns:
        if "women" in dataset_name.lower(): df["gender"] = "female"
        else: df["gender"] = "male"
        
    if "team_type" not in df.columns:
        df["team_type"] = "club" if df["league"].iloc[0] not in ["T20 World Cup", "Internationals", "Women's Internationals"] else "international"

    return df


def _aggregate_deliveries(df_m, df_d):
    """Group delivery rows to compute total runs and wickets per match & inning."""
    df_d.columns = [str(c).lower().strip() for c in df_d.columns]
    
    # Identify link column
    match_col = "match_id" if "match_id" in df_d.columns else "id"
    if match_col not in df_d.columns:
        logger.warning(f"Deliveries CSV missing match link column. Reverting to stub scores.")
        df_m["innings1_total_runs"] = 150
        df_m["innings1_wickets"] = 5
        df_m["innings2_total_runs"] = 140
        df_m["innings2_wickets"] = 6
        return df_m
        
    # Identify runs / wickets columns
    run_col = "total_runs" if "total_runs" in df_d.columns else "runs_off_bat"
    wkt_col = "is_wicket" if "is_wicket" in df_d.columns else "player_dismissed"
    inning_col = "inning" if "inning" in df_d.columns else "innings"
    
    if inning_col not in df_d.columns or run_col not in df_d.columns:
        df_m["innings1_total_runs"] = 150
        df_m["innings2_total_runs"] = 140
        df_m["innings1_wickets"] = 5
        df_m["innings2_wickets"] = 6
        return df_m

    # Compute Wickets (is_wicket is binary, or player_dismissed is string)
    if wkt_col == "player_dismissed":
        df_d["_wkt"] = df_d["player_dismissed"].notna().astype(int)
    else:
        df_d["_wkt"] = pd.to_numeric(df_d[wkt_col], errors="coerce").fillna(0)
    
    df_d["_runs"] = pd.to_numeric(df_d[run_col], errors="coerce").fillna(0)

    # Group by Match and Inning
    agg = df_d.groupby([match_col, inning_col]).agg(
        total_runs=("_runs", "sum"),
        total_wkts=("_wkt", "sum")
    ).reset_index()

    # Split into inning 1 and 2
    inn1 = agg[agg[inning_col] == 1].set_index(match_col)[["total_runs", "total_wkts"]]
    inn2 = agg[agg[inning_col] == 2].set_index(match_col)[["total_runs", "total_wkts"]]

    # Merge back to Matches
    df_m = df_m.join(inn1, on="match_id", rsuffix="_inn1")
    df_m = df_m.rename(columns={"total_runs": "innings1_total_runs", "total_wkts": "innings1_wickets"})
    
    df_m = df_m.join(inn2, on="match_id", rsuffix="_inn2")
    df_m = df_m.rename(columns={"total_runs": "innings2_total_runs", "total_wkts": "innings2_wickets"})

    # Fill NaNs
    df_m["innings1_total_runs"] = df_m["innings1_total_runs"].fillna(0)
    df_m["innings1_wickets"] = df_m["innings1_wickets"].fillna(0)
    df_m["innings2_total_runs"] = df_m["innings2_total_runs"].fillna(0)
    df_m["innings2_wickets"] = df_m["innings2_wickets"].fillna(0)
    
    return df_m
