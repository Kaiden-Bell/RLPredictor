"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Automatically verifies prediction outcomes from Ballchasing statistics.
- Usage: Runs standalone or as a helper to verify prediction correctness.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from scrapers.h2h_ballchasing import Ballchasing
from utils.database import (
    initialize_database,
    get_connection,
    get_cached_api_response,
    cache_api_response,
)
from utils.player_identity import normalize_player_name, resolve_player_alias


def load_cache(cache_path):
    """
    Description:
        Loads the old JSON API cache file.
    Arguments:
        cache_path: Path to the JSON cache file.
    Returns:
        Dictionary of cache data or empty dictionary.
    """
    if not os.path.exists(cache_path): return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def extract_player_game_stats_from_db():
    """
    Description:
        Extracts per-player, per-game stats from the SQLite player_stats table.
    Arguments:
        None
    Returns:
        Pandas DataFrame containing raw player game stats.
    """
    initialize_database()
    conn = get_connection()
    rows = conn.execute("""
        SELECT canonical_player_id,
               canonical_name  AS Player,
               display_name_seen,
               goals  AS Goals,
               shots  AS Shots,
               saves  AS Saves,
               demos  AS Demos,
               score  AS Score,
               date   AS replay_date,
               replay_id
        FROM player_stats
    """).fetchall()
    conn.close()
    if not rows: return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def resolve_player_to_canonical(player_name):
    """
    Description:
        Resolves a player name to its canonical platform identity.
    Arguments:
        player_name: String representing the player name.
    Returns:
        Tuple: (canonical_player_id, canonical_name) or (None, None).
    """
    conn = get_connection()
    norm = normalize_player_name(player_name)

    row = conn.execute(
        "SELECT canonical_player_id FROM player_aliases WHERE alias = ?", (norm,)
    ).fetchone()
    if row:
        cid = row["canonical_player_id"]
        cname_row = conn.execute(
            "SELECT canonical_name FROM canonical_players WHERE canonical_player_id = ?", (cid,)
        ).fetchone()
        conn.close()
        return cid, (cname_row["canonical_name"] if cname_row else player_name)

    row = conn.execute(
        "SELECT canonical_player_id, canonical_name FROM canonical_players WHERE canonical_player_id = ?", (norm,)
    ).fetchone()
    if row:
        conn.close()
        return row["canonical_player_id"], row["canonical_name"]

    # Fuzzy substring resolution to match player aliases
    all_aliases = conn.execute(
        "SELECT alias, canonical_player_id FROM player_aliases"
    ).fetchall()
    for r in all_aliases:
        alias = r["alias"]
        if norm in alias or alias in norm:
            cid = r["canonical_player_id"]
            cname_row = conn.execute(
                "SELECT canonical_name FROM canonical_players WHERE canonical_player_id = ?", (cid,)
            ).fetchone()
            conn.close()
            return cid, (cname_row["canonical_name"] if cname_row else player_name)

    conn.close()
    return None, None


def find_player_games(df_cache, canonical_player_id, num_games, pred_timestamp):
    """
    Description:
        Finds game replay records for a player close to the prediction window.
    Arguments:
        df_cache: Pandas DataFrame of player stats.
        canonical_player_id: Canonical player ID to search for.
        num_games: Number of games to retrieve.
        pred_timestamp: Prediction timestamp.
    Returns:
        Pandas DataFrame of matched player games.
    """
    if df_cache.empty: return pd.DataFrame()

    mask = df_cache["canonical_player_id"] == canonical_player_id
    p_games = df_cache[mask].copy()
    if p_games.empty: return pd.DataFrame()

    def _parse_date(d):
        try:
            return datetime.fromisoformat(str(d).replace("Z", "+00:00"))
        except Exception:
            return None

    p_games["_parsed_date"] = p_games["replay_date"].apply(_parse_date)
    p_games = p_games.dropna(subset=["_parsed_date"])

    try:
        pred_dt = datetime.fromisoformat(pred_timestamp)
        if pred_dt.tzinfo is None:
            pred_dt = pred_dt.replace(tzinfo=timezone.utc)
    except Exception:
        pred_dt = None

    if pred_dt is not None:
        p_games = p_games.sort_values("_parsed_date")
        window_start = pred_dt - timedelta(days=1)
        window_end = pred_dt + timedelta(days=3)
        candidate = p_games[
            (p_games["_parsed_date"] >= window_start) &
            (p_games["_parsed_date"] <= window_end)
        ]
        if not candidate.empty: p_games = candidate
    
    n = num_games if num_games and num_games > 0 else 1
    return p_games.head(n)


def fetch_group_replays(group_id, cache_path=".bc_cache.json"):
    """
    Description:
        Fetches all replay details in a Ballchasing group and populates the DB cache.
    Arguments:
        group_id: Ballchasing group ID.
        cache_path: Path to old JSON cache (fallback).
    Returns:
        Integer: Number of successfully fetched replays.
    """
    bc = Ballchasing()

    print(f"Fetching group: {group_id}")
    group_data = bc.get_group(group_id)
    group_name = group_data.get("name", group_id)
    direct = group_data.get("direct_replays", 0)
    indirect = group_data.get("indirect_replays", 0)
    print(f"  Group: {group_name} ({direct} direct, {indirect} total replays)")

    replay_ids = []
    try:
        replays_data = bc.list_replays(group=group_id, count=200)
        replay_list = replays_data.get("list", []) or []
        replay_ids = [r["id"] for r in replay_list if r.get("id")]
        print(f"  Found {len(replay_ids)} replays in group")
    except Exception as e:
        print(f"  Could not list replays: {e}")

    fetched = 0
    for i, rid in enumerate(replay_ids):
        try:
            bc.get_replay(rid)
            fetched += 1
            if (i + 1) % 5 == 0 or i == len(replay_ids) - 1:
                sys.stdout.write(f"\r  Fetched {i+1}/{len(replay_ids)} replay details...")
                sys.stdout.flush()
        except Exception as e:
            print(f"\n  Error fetching {rid}: {e}")

    if replay_ids: print(f"\n  Cached {fetched} replay details")

    try:
        key_str = f"/groups?{urlencode({'group': group_id, 'count': 200})}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()

        sub_groups_data = get_cached_api_response(cache_key)
        if sub_groups_data is None:
            url = "https://ballchasing.com/api/groups"
            r = bc.sess.get(url, params={"group": group_id, "count": 200}, timeout=30)
            r.raise_for_status()
            sub_groups_data = r.json()
            cache_api_response(cache_key, "/groups", json.dumps(sub_groups_data))

        sub_groups = sub_groups_data.get("list", []) or []
        if sub_groups:
            print(f"\n  Found {len(sub_groups)} sub-groups:")
            for sg in sub_groups:
                sg_name = sg.get("name", "?")
                sg_id = sg.get("id", "")
                sg_replays = sg.get("direct_replays", 0)
                print(f"    - {sg_name} ({sg_replays} replays) [{sg_id}]")
                if sg_replays > 0: fetch_group_replays(sg_id, cache_path)

    except Exception as e:
        print(f"  Sub-group fetch: {e}")

    return fetched


def clear_group_cache(group_id):
    """
    Description:
        Deletes cached group responses from the SQLite DB.
    Arguments:
        group_id: Ballchasing group ID.
    Returns:
        Integer: Number of deleted cache rows.
    """
    initialize_database()
    conn = get_connection()

    keys_to_remove = set()

    group_key = hashlib.md5(f"/groups/{group_id}?".encode()).hexdigest()
    keys_to_remove.add(group_key)

    list_key_str = f"/replays?{urlencode({'group': group_id, 'count': 200})}"
    list_key = hashlib.md5(list_key_str.encode()).hexdigest()
    keys_to_remove.add(list_key)

    cached_listing = get_cached_api_response(list_key, conn=conn)
    if cached_listing is not None:
        for replay in (cached_listing.get("list", []) or []):
            rid = replay.get("id", "")
            if rid:
                replay_key = hashlib.md5(f"/replays/{rid}?".encode()).hexdigest()
                keys_to_remove.add(replay_key)
                conn.execute("DELETE FROM player_stats WHERE replay_id = ?", (rid,))
                conn.execute("DELETE FROM replays WHERE replay_id = ?", (rid,))

    removed = 0
    for k in keys_to_remove:
        cur = conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (k,))
        removed += cur.rowcount

    conn.commit()
    conn.close()
    return removed


def fetch_match_group_replays(group_id, refresh=False):
    """
    Description:
        Fetches replays specifically from a given series match group.
    Arguments:
        group_id: Ballchasing series group ID.
        refresh: If True, purges local cache before querying.
    Returns:
        Pandas DataFrame of matched players stats.
    """
    if refresh:
        removed = clear_group_cache(group_id)
        print(f"  [RESET] Cleared {removed} cached entries for this group")

    bc = Ballchasing()

    print(f"Fetching match group: {group_id}")
    group_data = bc.get_group(group_id)
    group_name = group_data.get("name", group_id)
    direct = group_data.get("direct_replays", 0)
    print(f"  Match: {group_name} ({direct} replays)")

    replay_ids = []
    try:
        replays_data = bc.list_replays(group=group_id, count=200)
        replay_list = replays_data.get("list", []) or []
        replay_ids = [r["id"] for r in replay_list if r.get("id")]
        print(f"  Found {len(replay_ids)} replays")
    except Exception as e:
        print(f"  Could not list replays: {e}")
        return pd.DataFrame()

    for i, rid in enumerate(replay_ids):
        try:
            bc.get_replay(rid)
        except Exception as e:
            print(f"  Error fetching {rid}: {e}")

    if not replay_ids:
        print("  No replays found.")
        return pd.DataFrame()

    conn = get_connection()
    placeholders = ",".join(["?"] * len(replay_ids))
    rows = conn.execute(f"""
        SELECT canonical_player_id,
               canonical_name  AS Player,
               display_name_seen,
               goals  AS Goals,
               shots  AS Shots,
               saves  AS Saves,
               demos  AS Demos,
               score  AS Score,
               date   AS replay_date,
               replay_id
        FROM player_stats
        WHERE replay_id IN ({placeholders})
    """, replay_ids).fetchall()
    conn.close()

    if not rows:
        print("  No player stats found in DB for these replays.")
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    players = df["canonical_player_id"].nunique()
    print(f"  Extracted {len(df)} player-game records ({players} unique players)")
    return df


def verify_from_match_group(group_id, log_path="data/prediction_log.csv", refresh=False):
    """
    Description:
        Verifies predictions utilizing ONLY replays from a specified match group.
    Arguments:
        group_id: Ballchasing match series group ID.
        log_path: Path to logged predictions CSV.
        refresh: True if caching should be bypassed.
    Returns:
        None
    """
    if not os.path.exists(log_path):
        print("No prediction log found.")
        return

    df_log = pd.read_csv(log_path)
    if df_log.empty:
        print("Prediction log is empty.")
        return

    df_match = fetch_match_group_replays(group_id, refresh=refresh)
    if df_match.empty:
        print("No replay data from this match group.")
        return

    match_players = sorted(df_match["Player"].unique())
    print(f"\n  Players in this match: {', '.join(match_players)}")

    if "label" not in df_log.columns: df_log["label"] = np.nan

    matches_found = 0
    for i, row in df_log.iterrows():
        if pd.notna(row.get("label")): continue

        p_name = row["player"]
        stat = row["stat"]
        thresh = float(row["threshold"])
        num_games = int(row.get("num_games", 1))

        cid, _ = resolve_player_to_canonical(p_name)
        if not cid: continue

        mask = df_match["canonical_player_id"] == cid
        p_games = df_match[mask]
        if p_games.empty: continue
        if stat not in p_games.columns: continue

        p_games = p_games.sort_values("replay_date").head(num_games)
        actual_found = len(p_games)

        if actual_found < num_games:
            print(f"  [WARNING] {p_name} {stat}: Only found {actual_found}/{num_games} games in this match, skipping")
            continue

        if num_games > 1:
            actual_total = p_games[stat].sum()
            detail_str = f" ({actual_total} total across {actual_found} games)"
        else:
            actual_total = p_games[stat].iloc[0]
            detail_str = ""

        label = 1.0 if actual_total >= thresh else 0.0
        df_log.at[i, "label"] = label

        is_over = row.get("is_over", True)
        if isinstance(is_over, str): is_over = is_over.lower() == "true"
        nn_pick = "OVER" if is_over else "UNDER"
        correct = "[v]" if (is_over and label == 1.0) or (not is_over and label == 0.0) else "[x]"

        matches_found += 1
        print(f"  {p_name} | {stat} {'≥' if is_over else '<'} {thresh} | Actual: {actual_total}{detail_str} -> Picked {nn_pick} {correct}")

    if matches_found > 0:
        df_log.to_csv(log_path, index=False)
        print(f"\n[OK] Verified {matches_found} predictions from match group")

        labeled = df_log.dropna(subset=["label"])
        correct = 0
        for _, r in labeled.iterrows():
            is_over = r.get("is_over", True)
            if isinstance(is_over, str): is_over = is_over.lower() == "true"
            if is_over == (r["label"] == 1.0): correct += 1
        print(f"[STATS] Overall accuracy: {correct}/{len(labeled)} ({correct/len(labeled):.0%})")
    else:
        print(f"\nNo unlabeled predictions matched players in this match group.")
        print(f"  Players found: {', '.join(match_players)}")


def verify_predictions(log_path="data/prediction_log.csv", cache_path=".bc_cache.json"):
    """
    Description:
        Matches logged predictions against actual outcomes recorded in the DB cache.
    Arguments:
        log_path: Path to persistent logged predictions CSV file.
        cache_path: Old JSON API cache backup file path.
    Returns:
        None
    """
    if not os.path.exists(log_path):
        print("No prediction log found at data/prediction_log.csv")
        return

    df_log = pd.read_csv(log_path)
    if df_log.empty:
        print("Prediction log is empty.")
        return

    print(f"Analyzing {len(df_log)} predictions against cached replays...")

    df_cache = extract_player_game_stats_from_db()
    if df_cache.empty:
        print("No replay data found in database to verify against.")
        print("  Tip: fetch replays first with --group <id> or --match-group <id>")
        return

    print(f"  DB contains {len(df_cache)} player-game records from {df_cache['replay_id'].nunique()} replays")

    if "label" not in df_log.columns: df_log["label"] = np.nan

    matches_found = 0
    for i, row in df_log.iterrows():
        if pd.notna(row.get("label")): continue

        p_name = row["player"]
        stat = row["stat"]
        thresh = float(row["threshold"])
        num_games = int(row.get("num_games", 1))
        pred_ts = row["timestamp"]

        cid, _ = resolve_player_to_canonical(p_name)
        if not cid: continue

        games = find_player_games(df_cache, cid, num_games, pred_ts)
        if games.empty: continue
        if stat not in games.columns: continue

        actual_games_found = len(games)

        if num_games > 1:
            actual_total = games[stat].sum()
            per_game = f" ({actual_total} total across {actual_games_found} games)"
        else:
            actual_total = games[stat].iloc[0]
            per_game = ""

        if actual_games_found < num_games:
            print(f"  [WARNING] {p_name} {stat}: Only found {actual_games_found}/{num_games} games, skipping")
            continue

        label = 1.0 if actual_total >= thresh else 0.0
        df_log.at[i, "label"] = label

        is_over = row.get("is_over", True)
        if isinstance(is_over, str): is_over = is_over.lower() == "true"
        nn_pick = "OVER" if is_over else "UNDER"
        correct = "[v]" if (is_over and label == 1.0) or (not is_over and label == 0.0) else "[x]"

        matches_found += 1
        print(f"  {p_name} | {stat} {'≥' if is_over else '<'} {thresh} | Actual: {actual_total}{per_game} -> Picked {nn_pick} {correct}")

    if matches_found > 0:
        df_log.to_csv(log_path, index=False)
        print(f"\n[OK] Updated {matches_found} prediction outcomes in {log_path}")

        labeled = df_log.dropna(subset=["label"])
        if not labeled.empty:
            total = len(labeled)
            correct = 0
            for _, r in labeled.iterrows():
                is_over = r.get("is_over", True)
                if isinstance(is_over, str): is_over = is_over.lower() == "true"
                predicted_over = is_over
                actual_over = r["label"] == 1.0
                if predicted_over == actual_over: correct += 1
            print(f"[STATS] Overall accuracy: {correct}/{total} ({correct/total:.0%})")
    else:
        print("\nNo new matching replays found. Tips:")
        print("  1. Fetch replays: python verify_predictions.py --group <group_id>")
        print("  2. Or manually label: python verify_predictions.py --manual")


def manual_label(log_path="data/prediction_log.csv"):
    """
    Description:
        Triggers terminal interactive prediction labeling for missing records.
    Arguments:
        log_path: Path to logged predictions CSV file.
    Returns:
        None
    """
    if not os.path.exists(log_path):
        print("No prediction log found.")
        return

    df_log = pd.read_csv(log_path)
    if "label" not in df_log.columns: df_log["label"] = np.nan

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
        if isinstance(is_over, str): is_over = is_over.lower() == "true"
        ou = "Over" if is_over else "Under"
        games_str = f" in {num_games} games" if num_games > 1 else ""

        print(f"  [{i}] {p} — {ou} {thresh} {stat}{games_str}")
        ans = input(f"  Actual {stat} total? (or 's' to skip, 'q' to quit): ").strip()

        if ans.lower() == 'q': break
        if ans.lower() == 's': continue

        try:
            actual = float(ans)
            label = 1.0 if actual > thresh else 0.0
            df_log.at[i, "label"] = label
            result = "OVER" if label == 1.0 else "UNDER"
            correct = "[v]" if (is_over and label == 1.0) or (not is_over and label == 0.0) else "[x]"
            print(f"    -> {result} (actual: {actual}) {correct}\n")
        except ValueError:
            print("    -> Skipped (invalid number)\n")

    df_log.to_csv(log_path, index=False)
    labeled = df_log.dropna(subset=["label"])
    print(f"\nSaved! {len(labeled)} total labeled predictions.")


def main():
    """
    Description:
        Main CLI parsing and execution entry point.
    Arguments:
        None
    Returns:
        None
    """
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
