"""
dropdown_precompute.py — Pre-generate dropdown data as static JSON.

Eliminates per-request file/data queries for dropdown population.
Can be run standalone:
    python -m data.dropdown_precompute --output static/data/dropdowns.json

Or called from train_model.py:
    from data.dropdown_precompute import generate_dropdown_json
    generate_dropdown_json(dropdown_data, version_hash, output_path)
"""

import os
import json
import gzip
import argparse
from datetime import datetime


def generate_dropdown_json(dropdown_data, model_version, output_path):
    """
    Save dropdown data as a structured JSON file.

    Parameters
    ----------
    dropdown_data : dict
        Must contain: leagues, match_types, genders, cities,
        all_teams, all_venues, league_teams, league_venues, league_cities
    model_version : str
        Model version hash for cache-busting
    output_path : str
        Path to write the JSON file (e.g. static/data/dropdowns.json)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "leagues": dropdown_data.get("leagues", []),
        "formats": dropdown_data.get("match_types", []),
        "teams": dropdown_data.get("all_teams", []),
        "venues": dropdown_data.get("all_venues", []),
        "cities": dropdown_data.get("cities", []),
        "genders": dropdown_data.get("genders", []),
        "league_mappings": {},
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": model_version,
    }

    # Build league_mappings in the requested structure
    lt = dropdown_data.get("league_teams", {})
    lv = dropdown_data.get("league_venues", {})
    lc = dropdown_data.get("league_cities", {})

    for league in payload["leagues"]:
        payload["league_mappings"][league] = {
            "teams": lt.get(league, []),
            "venues": lv.get(league, []),
            "cities": lc.get(league, []),
        }

    # Write uncompressed JSON
    with open(output_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = os.path.getsize(output_path) / 1024

    # Also write gzipped version
    gz_path = output_path + ".gz"
    with open(output_path, "rb") as f_in:
        with gzip.open(gz_path, "wb", compresslevel=9) as f_out:
            f_out.write(f_in.read())

    gz_size_kb = os.path.getsize(gz_path) / 1024

    print(f"   📋 Dropdowns JSON → {output_path} ({size_kb:.1f} KB)")
    print(f"   📋 Gzipped        → {gz_path} ({gz_size_kb:.1f} KB)")

    return output_path


# ── CLI entrypoint ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate precomputed dropdown data from model artifacts."
    )
    parser.add_argument(
        "--output", "-o",
        default="static/data/dropdowns.json",
        help="Output path for the JSON file"
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Path to model artifacts directory"
    )
    args = parser.parse_args()

    # Try to load dropdown data from existing artifacts
    import joblib

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = args.artifacts_dir or os.path.join(base, "model", "artifacts")
    artifacts_path = os.path.join(artifacts_dir, "artifacts.joblib")
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")

    if not os.path.exists(artifacts_path):
        # Try legacy
        artifacts_path = os.path.join(base, "models", "cricket_preprocess.pkl")

    if not os.path.exists(artifacts_path):
        print("❌ No artifacts found. Run train_model.py first.")
        return

    print(f"Loading artifacts from {artifacts_path}...")
    artifacts = joblib.load(artifacts_path)
    dropdown_data = artifacts.get("dropdown_data", {})

    # Get version hash
    version = "unknown"
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            version = json.load(f).get("version_hash", "unknown")

    output = os.path.join(base, args.output) if not os.path.isabs(args.output) \
        else args.output
    generate_dropdown_json(dropdown_data, version, output)
    print("✅ Done!")


if __name__ == "__main__":
    main()
