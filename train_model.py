"""
train_model.py — Train the cricket prediction model locally.
Parses JSON match files from kagglehub, engineers features (including
20 advanced features + Elo ratings), trains a RandomForest, and saves
compressed .joblib files to model/artifacts/.

Run:  python3 train_model.py
"""

import os, gc, glob, json, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(LEGACY_DIR, exist_ok=True)

# ─── 1. Download dataset ───
print("📥 Downloading dataset via kagglehub...")
import kagglehub
path = kagglehub.dataset_download("pritthacker/cricket")
print(f"   Downloaded to: {path}\n")

# ─── 2. Discover JSON files ───
json_files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
print(f"Found {len(json_files)} JSON match file(s)")
if not json_files:
    raise RuntimeError("No JSON match files found!")

# ─── 3. Parse JSON files → match-level dataframe ───
print("\n🔧 Parsing JSON files into match-level features...\n")

records = []
parse_errors = 0

for i, jf in enumerate(json_files):
    try:
        with open(jf) as fh:
            d = json.load(fh)

        info = d.get("info", {})
        innings_data = d.get("innings", [])

        teams = info.get("teams", [])
        if len(teams) < 2:
            continue

        outcome = info.get("outcome", {})
        winner = outcome.get("winner", None)
        if not winner:
            continue

        result_type = outcome.get("result", "").lower()
        if result_type in ("no result", "tie", "draw"):
            continue

        venue = info.get("venue", "Unknown")
        city = info.get("city", "Unknown")
        toss = info.get("toss", {})
        toss_winner = toss.get("winner", "")
        toss_decision = toss.get("decision", "")
        dates = info.get("dates", [])
        match_date = dates[0] if dates else None
        match_id = os.path.splitext(os.path.basename(jf))[0]
        match_type = info.get("match_type", "Unknown")
        gender = info.get("gender", "male")
        team_type = info.get("team_type", "international")
        season = str(info.get("season", "Unknown"))

        # Extract league / event name
        event = info.get("event", {})
        if isinstance(event, dict):
            league = event.get("name", "Unknown")
        else:
            league = "Unknown"

        # Determine team1 (batting first) and team2
        if len(innings_data) >= 1:
            team1 = innings_data[0].get("team", teams[0])
            team2 = teams[1] if teams[0] == team1 else teams[0]
        else:
            team1, team2 = teams[0], teams[1]

        # Aggregate innings runs and wickets
        def agg_innings(inn_data):
            total_runs = 0
            total_wickets = 0
            for ov in inn_data.get("overs", []):
                for deliv in ov.get("deliveries", []):
                    runs_info = deliv.get("runs", {})
                    total_runs += runs_info.get("total", 0)
                    if "wickets" in deliv:
                        total_wickets += len(deliv["wickets"])
            return total_runs, total_wickets

        inn1_runs, inn1_wkts = (0, 0)
        inn2_runs, inn2_wkts = (0, 0)
        if len(innings_data) >= 1:
            inn1_runs, inn1_wkts = agg_innings(innings_data[0])
        if len(innings_data) >= 2:
            inn2_runs, inn2_wkts = agg_innings(innings_data[1])

        records.append({
            "match_id": match_id,
            "date": match_date,
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "city": city,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "innings1_total_runs": inn1_runs,
            "innings1_wickets": inn1_wkts,
            "innings2_total_runs": inn2_runs,
            "innings2_wickets": inn2_wkts,
            "winner": winner,
            "match_type": match_type,
            "gender": gender,
            "team_type": team_type,
            "league": league,
            "season": season,
        })

    except Exception:
        parse_errors += 1

    if (i + 1) % 2000 == 0:
        print(f"   Parsed {i+1}/{len(json_files)} files ({len(records)} valid matches)...")

print(f"\n✅ Parsed {len(records)} valid matches from {len(json_files)} files ({parse_errors} errors)")

md = pd.DataFrame(records)
print(f"   Match types: {md['match_type'].value_counts().to_dict()}")

# ─── 3.5. Parse CSV Datasets ───
print("\n🗄️ Parsing new Kaggle CSV datasets...")
from data.csv_parser import parse_all_csvs
csv_df = parse_all_csvs("data/raw_csv")

if not csv_df.empty:
    print(f"Merging {len(md)} JSON matches with {len(csv_df)} CSV matches...")
    # Concatenate matching schemas
    md = pd.concat([md, csv_df], ignore_index=True)
    # Drop hard duplicates across datasets
    initial_len = len(md)
    md = md.drop_duplicates(subset=["date", "team1", "team2", "winner"])
    print(f"Combined total: {len(md)} matches (dropped {initial_len - len(md)} duplicates)")

# ─── 4. Target + date sort ───
md["target"] = (md["team1"].str.strip().str.lower()
                == md["winner"].str.strip().str.lower()).astype(int)
print(f"\n🎯 Total Target distribution:\n{md['target'].value_counts()}")

md["toss_decision_bat_first"] = md["toss_decision"].str.lower().str.strip().isin(
    ["bat", "batting"]
).astype(np.int8)

md["date"] = pd.to_datetime(md["date"], errors="coerce")
md = md.sort_values("date").reset_index(drop=True)
print(f"📅 Date range: {md['date'].min()} → {md['date'].max()}")

# ─── 5. Original leakage-safe features ───
md["venue_avg_score"] = md.groupby("venue")["innings1_total_runs"].transform(
    lambda x: x.expanding().mean().shift(1)
).fillna(md["innings1_total_runs"].mean()).astype(np.float32)

def team_form(df, team_col, target_col, w=5):
    form = pd.Series(0.5, index=df.index, dtype=np.float32)
    hist = defaultdict(list)
    for i in df.index:
        t = str(df.loc[i, team_col]).strip().lower()
        if hist[t]:
            form.at[i] = np.mean(hist[t][-w:])
        hist[t].append(int(df.loc[i, target_col]))
    return form

print("⏳ Computing team form (5-match)...")
md["team1_recent_form"] = team_form(md, "team1", "target")
md["_t2w"] = 1 - md["target"]
md["team2_recent_form"] = team_form(md, "team2", "_t2w")
md.drop("_t2w", axis=1, inplace=True)

# ─── 6. Advanced features (20 new) ───
print("\n🚀 Computing 20 advanced features...")
from features.advanced_features import compute_all_advanced_features
md, advanced_feat_names = compute_all_advanced_features(md)

# ─── 7. Elo ratings ───
print("\n♟️  Computing Elo ratings (per-format)...")
from features.elo import EloSystem
elo = EloSystem(k=32, k_final=48, initial=1500, decay_days=180, decay_factor=0.10)
md = elo.compute_elo_features(md)
print(f"   Elo computed for {len(elo.get_all_ratings())} team-format combinations")

# ─── 8. Encode + split + train ───
NUM_FEATS = [
    # Original 8
    "innings1_total_runs", "innings2_total_runs",
    "innings1_wickets", "innings2_wickets",
    "toss_decision_bat_first", "venue_avg_score",
    "team1_recent_form", "team2_recent_form",
    # Elo (2)
    "elo_team1", "elo_team2",
] + advanced_feat_names  # 20 advanced features

CAT_FEATS = ["venue", "team1", "team2", "toss_winner",
             "match_type", "gender", "league", "city"]

encoders = {}
for col in CAT_FEATS:
    le = LabelEncoder()
    md[col] = md[col].astype(str).fillna("UNKNOWN")
    le.fit(md[col])
    md[col + "_enc"] = le.transform(md[col])
    encoders[col] = le

enc_feats = [c + "_enc" for c in CAT_FEATS]
all_feats = NUM_FEATS + enc_feats

X = md[all_feats].fillna(0)
y = md["target"]

split = int(len(X) * 0.8)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

scaler = StandardScaler()
scaler.fit(X_tr)

print(f"\n📏 Train: {len(X_tr):,}  |  Test: {len(X_te):,}")
print(f"📊 Features ({len(all_feats)}): {all_feats}\n")
print("⏳ Training Stacking Ensemble (RF, XGB, LGBM, LR)...")
from model.ensemble import EnsemblePredictor
model = EnsemblePredictor(artifacts_dir=os.path.join(BASE_DIR, "model", "artifacts", "ensemble"))

# Train Ensemble
df_train = X_tr.copy()
df_train["target"] = y_tr
model.train(df_train, target_col="target", n_splits=5)

# ─── 9. Evaluate + Bootstrap CI ───
y_proba = model.predict_proba(X_te)
y_pred = (y_proba >= 0.5).astype(int)
acc = accuracy_score(y_te, y_pred)
try:
    auc = roc_auc_score(y_te, y_proba)
except:
    auc = None

boot_accs = []
rng = np.random.RandomState(42)
n = len(y_te)
y_te_arr, y_pred_arr = np.array(y_te), np.array(y_pred)
for _ in range(200):
    idx = rng.choice(n, n, replace=True)
    if len(np.unique(y_te_arr[idx])) < 2: continue
    boot_accs.append(accuracy_score(y_te_arr[idx], y_pred_arr[idx]))
ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])

print(f"\n{'='*55}")
print(f"  ✅ Accuracy: {acc:.3f}  (95% CI: {ci_lo:.3f}–{ci_hi:.3f})")
if auc: print(f"     ROC-AUC:  {auc:.3f}")
print(f"{'='*55}")

# Feature Importances will be extracted directly from Level-1 models in analysis module
print("\n📊 Check feature_report.json for importance metrics extracted via permutation and SHAP.")

# ─── 10. Build dropdown metadata ───
league_teams = defaultdict(set)
league_venues = defaultdict(set)
league_cities = defaultdict(set)

for _, row in md.iterrows():
    lg = str(row["league"])
    league_teams[lg].add(str(row["team1"]))
    league_teams[lg].add(str(row["team2"]))
    league_venues[lg].add(str(row["venue"]))
    league_cities[lg].add(str(row["city"]))

dropdown_data = {
    "leagues": sorted(set(md["league"].unique()) - {"Unknown"}),
    "match_types": sorted(md["match_type"].unique().tolist()),
    "genders": sorted(md["gender"].unique().tolist()),
    "cities": sorted(set(md["city"].unique()) - {"Unknown"}),
    "all_teams": sorted(set(md["team1"].unique()) | set(md["team2"].unique())),
    "all_venues": sorted(md["venue"].unique().tolist()),
    "league_teams": {k: sorted(v) for k, v in league_teams.items()},
    "league_venues": {k: sorted(v) for k, v in league_venues.items()},
    "league_cities": {k: sorted(v) for k, v in league_cities.items()},
}

# ─── 11. Save ───
artifacts_dict = {
    "encoders": encoders,
    "scaler": scaler,
    "feature_list": all_feats,
    "numeric_features": NUM_FEATS,
    "categorical_features": CAT_FEATS,
    "encoded_cat_features": enc_feats,
    "advanced_feature_names": advanced_feat_names,
    "train_venue_avg": md.groupby("venue")["innings1_total_runs"].mean().to_dict(),
    "global_avg_score": float(md["innings1_total_runs"].mean()),
    "dropdown_data": dropdown_data,
    "model_metadata": {
        "accuracy": float(acc),
        "roc_auc": float(auc) if auc else None,
        "bootstrap_ci_accuracy": [float(ci_lo), float(ci_hi)],
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "features": all_feats,
    }
}

# Save compressed format metadata (model itself already saved by EnsemblePredictor)
COMPRESSED_DIR = os.path.join(BASE_DIR, "model", "artifacts")
with open(os.path.join(COMPRESSED_DIR, "metadata.json"), "w") as f:
    json.dump({
        "features": all_feats,
        "accuracy": acc,
        "auc": auc
    }, f)

# Save artifact dict for transformers
joblib.dump(artifacts_dict, os.path.join(COMPRESSED_DIR, "artifacts_dict.joblib"), compress=3)
print(f"\n💾 Pipeline artifacts saved to {COMPRESSED_DIR}")

# ─── 12. Generate precomputed dropdowns JSON ───
print("\n📋 Generating precomputed dropdowns...")
from data.dropdown_precompute import generate_dropdown_json
dropdown_output = os.path.join(BASE_DIR, "static", "data", "dropdowns.json")
generate_dropdown_json(dropdown_data, "ensemble_v1", dropdown_output)

print(f"\n🏏 Leagues: {len(dropdown_data['leagues'])}")
print(f"🏏 Teams:   {len(dropdown_data['all_teams'])}")
print(f"🏏 Venues:  {len(dropdown_data['all_venues'])}")
print(f"🌆 Cities:  {len(dropdown_data['cities'])}")

print(f"\n🎉 Done! Restart the Flask server and visit http://127.0.0.1:5000")
