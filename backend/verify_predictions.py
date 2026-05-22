"""
verify_predictions.py — Auto-verify prediction outcomes from Ballchasing data.

Matches logged predictions against actual game outcomes by:
1. Searching the Ballchasing cache for replays from the predicted match
2. Aggregating stats across num_games for multi-game bets
3. Writing labels back to prediction_log.csv for training

Usage:
    python verify_predictions.py                     # Verify from cache
    python verify_predictions.py --group <group_id>  # Fetch replays from a BC group first
    python verify_predictions.py --manual             # Interactively label predictions
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load BALLCHASING_API_KEY from .env
except ImportError:
    pass  # dotenv not available (e.g., tests); key must be in environment


def _load_cache(cache_path):
    """Load the Ballchasing cache file."""
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def _extract_player_game_stats(cache):
    """Extract per-player per-game stats from all replay details in cache."""
    from features import _extract_player_stats, _is_replay_detail

    rows = []
    for key, data in cache.items():
        if not _is_replay_detail(data):
            continue
        replay_date = data.get("date", "")
        player_rows = _extract_player_stats(data)
        for r in player_rows:
            r["replay_date"] = replay_date
            r["replay_id"] = data.get("id", key)
        rows.extend(player_rows)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _find_player_games(df_cache, player_name, num_games, pred_timestamp):
    """
    Find the most relevant games for a player near the prediction time.
    Returns up to num_games rows.
    """
    if df_cache.empty:
        return pd.DataFrame()

    # Match player (case-insensitive)
    mask = df_cache["Player"].str.lower() == player_name.lower()
    p_games = df_cache[mask].copy()

    if p_games.empty:
        return pd.DataFrame()

    # Parse replay dates and sort
    def _parse_date(d):
        try:
            return datetime.fromisoformat(str(d).replace("Z", "+00:00"))
        except Exception:
            return None

    p_games["_parsed_date"] = p_games["replay_date"].apply(_parse_date)
    p_games = p_games.dropna(subset=["_parsed_date"])

    # Parse prediction timestamp
    try:
        pred_dt = datetime.fromisoformat(pred_timestamp)
        if pred_dt.tzinfo is None:
            pred_dt = pred_dt.replace(tzinfo=timezone.utc)
    except Exception:
        pred_dt = None

    if pred_dt is not None:
        # Find games closest to (and ideally after) the prediction
        p_games = p_games.sort_values("_parsed_date")
        # Take games from within ±3 days of prediction
        from datetime import timedelta
        window_start = pred_dt - timedelta(days=1)
        window_end = pred_dt + timedelta(days=3)
        candidate = p_games[
            (p_games["_parsed_date"] >= window_start) &
            (p_games["_parsed_date"] <= window_end)
        ]
        if not candidate.empty:
            p_games = candidate
    
    # Take the first num_games (or all if fewer)
    n = num_games if num_games and num_games > 0 else 1
    return p_games.head(n)


def fetch_group_replays(group_id, cache_path=".bc_cache.json"):
    """
    Fetch all replays from a Ballchasing group (and sub-groups) into the cache.

    This walks the group hierarchy and fetches replay details for all
    replays found, caching them for verification.
    """
    from scrapers import Ballchasing
    bc = Ballchasing()

    print(f"Fetching group: {group_id}")
    group_data = bc.getGroup(group_id)
    group_name = group_data.get("name", group_id)
    direct = group_data.get("direct_replays", 0)
    indirect = group_data.get("indirect_replays", 0)
    print(f"  Group: {group_name} ({direct} direct, {indirect} total replays)")

    # List replays in this group
    replay_ids = []
    try:
        replays_data = bc.listReplays(group=group_id, count=200)
        replay_list = replays_data.get("list", []) or []
        replay_ids = [r["id"] for r in replay_list if r.get("id")]
        print(f"  Found {len(replay_ids)} replays in group")
    except Exception as e:
        print(f"  Could not list replays: {e}")

    # Fetch each replay detail (will be cached)
    fetched = 0
    for i, rid in enumerate(replay_ids):
        try:
            bc.getReplay(rid)
            fetched += 1
            if (i + 1) % 5 == 0 or i == len(replay_ids) - 1:
                sys.stdout.write(f"\r  Fetched {i+1}/{len(replay_ids)} replay details...")
                sys.stdout.flush()
        except Exception as e:
            print(f"\n  Error fetching {rid}: {e}")

    if replay_ids:
        print(f"\n  Cached {fetched} replay details")

    # Also recurse into sub-groups
    try:
        sub_data = bc.listReplays()  # We need to list sub-groups differently
        # Use the groups API to find children
        import hashlib
        from urllib.parse import urlencode
        key_str = f"/groups?{urlencode({'group': group_id, 'count': 200})}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()

        if cache_key not in bc.cache:
            url = "https://ballchasing.com/api/groups"
            r = bc.sess.get(url, params={"group": group_id, "count": 200}, timeout=30)
            r.raise_for_status()
            sub_groups_data = r.json()
            bc.cache[cache_key] = sub_groups_data
            bc._save_cache()
        else:
            sub_groups_data = bc.cache[cache_key]

        sub_groups = sub_groups_data.get("list", []) or []
        if sub_groups:
            print(f"\n  Found {len(sub_groups)} sub-groups:")
            for sg in sub_groups:
                sg_name = sg.get("name", "?")
                sg_id = sg.get("id", "")
                sg_replays = sg.get("direct_replays", 0)
                print(f"    - {sg_name} ({sg_replays} replays) [{sg_id}]")
                if sg_replays > 0:
                    fetch_group_replays(sg_id, cache_path)

    except Exception as e:
        print(f"  Sub-group fetch: {e}")

    return fetched


def _clear_group_cache(group_id):
    """
    Remove cached API responses for a specific group so fresh data is fetched.
    Clears: group details, replay listing, and individual replay details.
    """
    import hashlib
    from urllib.parse import urlencode

    cache_path = ".bc_cache.json"
    if not os.path.exists(cache_path):
        return 0

    with open(cache_path, "r") as f:
        cache = json.load(f)

    original_size = len(cache)

    # Build cache keys to remove
    keys_to_remove = set()

    # Group detail key
    group_key = hashlib.md5(f"/groups/{group_id}?".encode()).hexdigest()
    keys_to_remove.add(group_key)

    # Replay listing key (with group param)
    list_key_str = f"/replays?{urlencode({'group': group_id, 'count': 200})}"
    keys_to_remove.add(hashlib.md5(list_key_str.encode()).hexdigest())

    # Also try to find the replay IDs from the cached listing so we can clear those too
    for key_str_variant in [list_key_str]:
        ck = hashlib.md5(key_str_variant.encode()).hexdigest()
        if ck in cache:
            listing = cache[ck]
            for replay in (listing.get("list", []) or []):
                rid = replay.get("id", "")
                if rid:
                    replay_key = hashlib.md5(f"/replays/{rid}?".encode()).hexdigest()
                    keys_to_remove.add(replay_key)

    removed = 0
    for k in keys_to_remove:
        if k in cache:
            del cache[k]
            removed += 1

    if removed > 0:
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    return removed


def fetch_match_group_replays(group_id, refresh=False):
    """
    Fetch replays from a SPECIFIC match group (e.g., 'vp-vs-dig-8x8pkxxs6y').
    Returns a DataFrame of player-game stats from ONLY that group's replays.
    
    Unlike fetch_group_replays, this does NOT recurse into sub-groups
    and returns the stats directly rather than relying on the full cache.
    
    If refresh=True, clears cached data for this group first.
    """
    if refresh:
        removed = _clear_group_cache(group_id)
        print(f"  🔄 Cleared {removed} cached entries for this group")

    from scrapers import Ballchasing
    from features import _extract_player_stats
    bc = Ballchasing()

    print(f"Fetching match group: {group_id}")
    group_data = bc.getGroup(group_id)
    group_name = group_data.get("name", group_id)
    direct = group_data.get("direct_replays", 0)
    print(f"  Match: {group_name} ({direct} replays)")

    # List replays in this specific group only
    replay_ids = []
    try:
        replays_data = bc.listReplays(group=group_id, count=200)
        replay_list = replays_data.get("list", []) or []
        replay_ids = [r["id"] for r in replay_list if r.get("id")]
        print(f"  Found {len(replay_ids)} replays")
    except Exception as e:
        print(f"  Could not list replays: {e}")
        return pd.DataFrame()

    # Fetch each replay detail and extract stats
    all_rows = []
    for i, rid in enumerate(replay_ids):
        try:
            detail = bc.getReplay(rid)
            player_rows = _extract_player_stats(detail)
            for r in player_rows:
                r["replay_date"] = detail.get("date", "")
                r["replay_id"] = rid
            all_rows.extend(player_rows)
        except Exception as e:
            print(f"  Error fetching {rid}: {e}")

    if not all_rows:
        print("  No player stats extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    players = df["Player"].nunique()
    print(f"  Extracted {len(df)} player-game records ({players} unique players)")
    return df


def verify_from_match_group(group_id, log_path="data/prediction_log.csv", refresh=False):
    """
    Verify predictions using ONLY replays from a specific match group.
    
    This prevents stats from other matches leaking into the verification.
    For example, if Evoh played in VP vs Dig AND Dig vs FUT, and you only
    want to verify predictions from VP vs Dig, use:
        --match-group vp-vs-dig-8x8pkxxs6y
    """
    if not os.path.exists(log_path):
        print("No prediction log found.")
        return

    df_log = pd.read_csv(log_path)
    if df_log.empty:
        print("Prediction log is empty.")
        return

    # Fetch stats from ONLY this match group
    df_match = fetch_match_group_replays(group_id, refresh=refresh)
    if df_match.empty:
        print("No replay data from this match group.")
        return

    # Show which players are in the match
    match_players = sorted(df_match["Player"].unique())
    print(f"\n  Players in this match: {', '.join(match_players)}")

    # Ensure label column
    if "label" not in df_log.columns:
        df_log["label"] = np.nan

    matches_found = 0
    for i, row in df_log.iterrows():
        if pd.notna(row.get("label")):
            continue

        p_name = row["player"]
        stat = row["stat"]
        thresh = float(row["threshold"])
        num_games = int(row.get("num_games", 1))

        # Only match against players actually in this match group
        mask = df_match["Player"].str.lower() == p_name.lower()
        p_games = df_match[mask]

        if p_games.empty:
            continue

        if stat not in p_games.columns:
            continue

        # Take exactly num_games (sorted by date)
        p_games = p_games.sort_values("replay_date").head(num_games)
        actual_found = len(p_games)

        if actual_found < num_games:
            print(f"  ⚠ {p_name} {stat}: Only found {actual_found}/{num_games} games in this match, skipping")
            continue

        # Aggregate
        if num_games > 1:
            actual_total = p_games[stat].sum()
            detail_str = f" ({actual_total} total across {actual_found} games)"
        else:
            actual_total = p_games[stat].iloc[0]
            detail_str = ""

        label = 1.0 if actual_total >= thresh else 0.0
        df_log.at[i, "label"] = label

        is_over = row.get("is_over", True)
        if isinstance(is_over, str):
            is_over = is_over.lower() == "true"
        nn_pick = "OVER" if is_over else "UNDER"
        correct = "✅" if (is_over and label == 1.0) or (not is_over and label == 0.0) else "❌"

        matches_found += 1
        print(f"  {p_name} | {stat} {'≥' if is_over else '<'} {thresh} | Actual: {actual_total}{detail_str} → Picked {nn_pick} {correct}")

    if matches_found > 0:
        df_log.to_csv(log_path, index=False)
        print(f"\n✅ Verified {matches_found} predictions from match group")

        # Accuracy summary
        labeled = df_log.dropna(subset=["label"])
        correct = 0
        for _, r in labeled.iterrows():
            is_over = r.get("is_over", True)
            if isinstance(is_over, str):
                is_over = is_over.lower() == "true"
            if is_over == (r["label"] == 1.0):
                correct += 1
        print(f"📊 Overall accuracy: {correct}/{len(labeled)} ({correct/len(labeled):.0%})")
    else:
        print(f"\nNo unlabeled predictions matched players in this match group.")
        print(f"  Players found: {', '.join(match_players)}")


def verify_predictions(log_path="data/prediction_log.csv", cache_path=".bc_cache.json"):
    """
    Match logged predictions against actual outcomes in the cache.
    Updates the log CSV with a 'label' column (1.0 for OVER hit, 0.0 for UNDER hit).

    For multi-game bets (num_games > 1), aggregates stats across that many games.
    """
    if not os.path.exists(log_path):
        print("No prediction log found at data/prediction_log.csv")
        return

    df_log = pd.read_csv(log_path)
    if df_log.empty:
        print("Prediction log is empty.")
        return

    # Load cache
    cache = _load_cache(cache_path)
    if not cache:
        print(f"No cache found at {cache_path}")
        return

    print(f"Analyzing {len(df_log)} predictions against cached replays...")

    # Extract all player stats from cache
    df_cache = _extract_player_game_stats(cache)
    if df_cache.empty:
        print("No valid replay details found in cache to verify against.")
        return

    print(f"  Cache contains {len(df_cache)} player-game records from {df_cache['replay_id'].nunique()} replays")

    # Ensure label column exists
    if "label" not in df_log.columns:
        df_log["label"] = np.nan

    matches_found = 0
    for i, row in df_log.iterrows():
        # Only verify if not already labeled
        if pd.notna(row.get("label")):
            continue

        p_name = row["player"]
        stat = row["stat"]
        thresh = float(row["threshold"])
        num_games = int(row.get("num_games", 1))
        pred_ts = row["timestamp"]

        # Find matching games
        games = _find_player_games(df_cache, p_name, num_games, pred_ts)

        if games.empty:
            continue

        if stat not in games.columns:
            continue

        actual_games_found = len(games)

        # For multi-game bets: aggregate (sum) across games
        if num_games > 1:
            actual_total = games[stat].sum()
            per_game = f" ({actual_total} total across {actual_games_found} games)"
        else:
            actual_total = games[stat].iloc[0]
            per_game = ""

        # Check if we found enough games
        if actual_games_found < num_games:
            print(f"  ⚠ {p_name} {stat}: Only found {actual_games_found}/{num_games} games, skipping")
            continue

        # Label: 1.0 = the actual result was OVER/equal the threshold, 0.0 = UNDER
        label = 1.0 if actual_total >= thresh else 0.0
        df_log.at[i, "label"] = label

        is_over = row.get("is_over", True)
        if isinstance(is_over, str):
            is_over = is_over.lower() == "true"
        nn_pick = "OVER" if is_over else "UNDER"
        correct = "✅" if (is_over and label == 1.0) or (not is_over and label == 0.0) else "❌"

        matches_found += 1
        print(f"  {p_name} | {stat} {'≥' if is_over else '<'} {thresh} | Actual: {actual_total}{per_game} → Picked {nn_pick} {correct}")

    if matches_found > 0:
        df_log.to_csv(log_path, index=False)
        print(f"\n✅ Updated {matches_found} prediction outcomes in {log_path}")

        # Summary
        labeled = df_log.dropna(subset=["label"])
        if not labeled.empty:
            total = len(labeled)
            # Check prediction accuracy
            correct = 0
            for _, r in labeled.iterrows():
                is_over = r.get("is_over", True)
                if isinstance(is_over, str):
                    is_over = is_over.lower() == "true"
                predicted_over = is_over
                actual_over = r["label"] == 1.0
                if predicted_over == actual_over:
                    correct += 1
            print(f"📊 Overall accuracy: {correct}/{total} ({correct/total:.0%})")
    else:
        print("\nNo new matching replays found. Tips:")
        print("  1. Fetch replays: python verify_predictions.py --group <group_id>")
        print("  2. Or manually label: python verify_predictions.py --manual")


def manual_label(log_path="data/prediction_log.csv"):
    """Interactively label predictions that don't have outcomes yet."""
    if not os.path.exists(log_path):
        print("No prediction log found.")
        return

    df_log = pd.read_csv(log_path)
    if "label" not in df_log.columns:
        df_log["label"] = np.nan

    unlabeled = df_log[df_log["label"].isna()]
    if unlabeled.empty:
        print("All predictions are already labeled!")
        return

    print(f"\n{len(unlabeled)} unlabeled predictions. Enter actual values or 's' to skip.\n")

    for i, row in unlabeled.iterrows():
        p = row["player"]
        stat = row["stat"]
        thresh = row["threshold"]
        num_games = int(row.get("num_games", 1))
        is_over = row.get("is_over", True)
        if isinstance(is_over, str):
            is_over = is_over.lower() == "true"
        ou = "Over" if is_over else "Under"
        games_str = f" in {num_games} games" if num_games > 1 else ""

        print(f"  [{i}] {p} — {ou} {thresh} {stat}{games_str}")
        ans = input(f"  Actual {stat} total? (or 's' to skip, 'q' to quit): ").strip()

        if ans.lower() == 'q':
            break
        if ans.lower() == 's':
            continue

        try:
            actual = float(ans)
            label = 1.0 if actual > thresh else 0.0
            df_log.at[i, "label"] = label
            result = "OVER" if label == 1.0 else "UNDER"
            correct = "✅" if (is_over and label == 1.0) or (not is_over and label == 0.0) else "❌"
            print(f"    → {result} (actual: {actual}) {correct}\n")
        except ValueError:
            print("    → Skipped (invalid number)\n")

    df_log.to_csv(log_path, index=False)
    labeled = df_log.dropna(subset=["label"])
    print(f"\nSaved! {len(labeled)} total labeled predictions.")


def main():
    parser = argparse.ArgumentParser(description="Verify prediction outcomes")
    parser.add_argument("--group", type=str, help="Ballchasing group ID to fetch replays from (walks sub-groups)")
    parser.add_argument("--match-group", type=str, dest="match_group",
                        help="Ballchasing match group ID — verify using ONLY replays from this specific series")
    parser.add_argument("--refresh", action="store_true",
                        help="Clear cached data for the group and re-fetch from Ballchasing API")
    parser.add_argument("--cache", type=str, default=".bc_cache.json", help="Cache file path")
    parser.add_argument("--log", type=str, default="data/prediction_log.csv", help="Prediction log path")
    parser.add_argument("--manual", action="store_true", help="Manually label predictions interactively")
    args = parser.parse_args()

    if args.manual:
        manual_label(args.log)
        return

    if args.match_group:
        print(f"\n=== Verifying from match group: {args.match_group} ===\n")
        verify_from_match_group(args.match_group, log_path=args.log, refresh=args.refresh)
        return

    if args.group:
        print(f"\n=== Fetching replays from Ballchasing group: {args.group} ===\n")
        fetch_group_replays(args.group, args.cache)
        print()

    print(f"\n=== Verifying predictions ===\n")
    verify_predictions(log_path=args.log, cache_path=args.cache)


if __name__ == "__main__":
    main()
