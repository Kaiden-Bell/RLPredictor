import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def verify_predictions(log_path="data/prediction_log.csv", cache_path=".bc_cache.json"):
    """
    Match logged predictions against actual outcomes in the cache.
    Updates the log CSV with a 'label' column (1.0 for OVER hit, 0.0 for UNDER hit).
    """
    if not os.path.exists(log_path):
        print("No prediction log found at data/prediction_log.csv")
        return

    df_log = pd.read_csv(log_path)
    if df_log.empty:
        print("Prediction log is empty.")
        return

    # Load cache
    if not os.path.exists(cache_path):
        print(f"No cache found at {cache_path}")
        return
        
    with open(cache_path, "r") as f:
        cache = json.load(f)

    print(f"Analyzing {len(df_log)} predictions against cached replays...")

    # Extract all player stats from cache for matching
    all_cached_stats = []
    for key, data in cache.items():
        if isinstance(data, dict) and "blue" in data and "orange" in data:
            date_str = data.get("date")
            if not date_str: continue
            
            # Use same stat extraction as features.py
            from features import _extract_player_stats
            rows = _extract_player_stats(data)
            for r in rows:
                r["date"] = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                all_cached_stats.append(r)
    
    if not all_cached_stats:
        print("No valid replay details found in cache to verify against.")
        return

    df_cache = pd.DataFrame(all_cached_stats)
    
    # We'll add a 'label' column to the log if it doesn't exist
    if "label" not in df_log.columns:
        df_log["label"] = np.nan

    matches_found = 0
    for i, row in df_log.iterrows():
        # Only verify if not already labeled
        if not np.isnan(row["label"]):
            continue

        p_name = row["player"]
        stat = row["stat"]
        thresh = row["threshold"]
        pred_date = datetime.fromisoformat(row["timestamp"])
        # Ensure pred_date is timezone-aware (assume local -> UTC)
        if pred_date.tzinfo is None:
            pred_date = pred_date.astimezone(timezone.utc)

        # Find games for this player AFTER the prediction was made
        potential_games = df_cache[
            (df_cache["Player"].str.lower() == p_name.lower()) & 
            (df_cache["date"] > pred_date)
        ].sort_values("date")

        if potential_games.empty:
            continue

        # For simplicity, we take the FIRST game found after the prediction
        # In multi-game bets, this logic would need to aggregate
        actual_val = potential_games.iloc[0].get(stat)
        if actual_val is not None:
            label = 1.0 if actual_val > thresh else 0.0
            df_log.at[i, "label"] = label
            matches_found += 1
            print(f"Matched: {p_name} | {stat} > {thresh} | Actual: {actual_val} | Label: {label}")

    if matches_found > 0:
        df_log.to_csv(log_path, index=False)
        print(f"\nUpdated {matches_found} new outcomes in {log_path}")
    else:
        print("\nNo new matching replays found yet. Make sure to run the scraper after games finish!")

if __name__ == "__main__":
    verify_predictions()
