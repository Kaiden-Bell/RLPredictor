"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Neural Network 13-dimensional feature extraction and training sample generator.
- Usage: Imported by main.py, train.py, and chat.py to convert raw DB data to normalized tensors.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from utils.database import get_all_replay_details, initialize_database


MAX_GAMES_H2H = 30
MAX_GAMES_GEN = 150
MAX_GAMES_MOMENTUM = 50
MAX_SCORE = 1000.0
STAT_CAPS = {"Goals": 8, "Shots": 12, "Saves": 10, "Demos": 8}
STATS = ["Goals", "Shots", "Saves", "Demos"]
THRESHOLDS = {
    "Goals": [0.5, 1.5, 2.5, 3.5],
    "Shots": [1.5, 2.5, 3.5, 5.5],
    "Saves": [0.5, 1.5, 2.5, 3.5],
    "Demos": [0.5, 1.5, 2.5, 4.5],
}


def norm(val, cap):
    """
    Description:
        Normalizes a value inside [0, 1] using a specified cap limit.
    Arguments:
        val: Numerical value to normalize.
        cap: Maximum cap value.
    Returns:
        Float: Normalized value inside [0.0, 1.0].
    """
    return min(float(val) / cap, 1.0) if cap else 0.0


def extract_features(
    player_name,
    stat_name,
    threshold,
    h2h_df=None,
    gen_df=None,
    momentum_data=None,
    sentiment_data=None,
    confident_h2h=False,
    playlist_type=1,
):
    """
    Description:
        Constructs the 13-dimensional feature vector for a query.
    Arguments:
        player_name: String name of the player.
        stat_name: target statistic.
        threshold: Over/under threshold.
        h2h_df: Head-to-head DataFrame.
        gen_df: General statistics DataFrame.
        momentum_data: Ranked momentum dictionary.
        sentiment_data: Reddit sentiment dictionary.
        confident_h2h: True if high confidence direct match.
        playlist_type: playlist category float.
    Returns:
        1D Numpy float32 array representing the 13-dimensional feature vector.
    """
    stat_cap = STAT_CAPS.get(stat_name, 8)
    features = np.zeros(13, dtype=np.float32)

    if h2h_df is not None and not h2h_df.empty and player_name in h2h_df["Player"].values:
        p = h2h_df[h2h_df["Player"] == player_name]
        n = len(p)
        if n > 0 and stat_name in p.columns:
            vals = p[stat_name].values
            features[0] = norm(vals.mean(), stat_cap)
            features[1] = (vals > threshold).mean()
            features[2] = norm(n, MAX_GAMES_H2H)
    features[3] = 1.0 if confident_h2h else 0.0

    if gen_df is not None and not gen_df.empty and player_name in gen_df["Player"].values:
        p = gen_df[gen_df["Player"] == player_name]
        n = len(p)
        if n > 0 and stat_name in p.columns:
            vals = p[stat_name].values
            features[4] = norm(vals.mean(), stat_cap)
            features[5] = (vals > threshold).mean()
            features[6] = norm(n, MAX_GAMES_GEN)
            features[7] = norm(vals.std(), stat_cap)

    if momentum_data and isinstance(momentum_data, dict):
        features[8] = norm(momentum_data.get("games", 0), MAX_GAMES_MOMENTUM)
        features[9] = norm(momentum_data.get("avg_score", 0), MAX_SCORE)
        features[10] = momentum_data.get("win_rate", 0) / 100.0

    if sentiment_data and isinstance(sentiment_data, dict):
        features[11] = (sentiment_data.get("score", 0.0) + 1.0) / 2.0

    features[12] = float(playlist_type)

    return features


def extract_player_stats(detail):
    """
    Description:
        Extracts player stats from a single replay detail structure.
    Arguments:
        detail: Replay detail parsed JSON dictionary.
    Returns:
        List of dictionaries with stats.
    """
    rows = []
    for side in ("blue", "orange"):
        team = detail.get(side) or {}
        for pl in team.get("players", []) or []:
            name = pl.get("name") or (pl.get("player") or {}).get("name")
            stats = pl.get("stats") or {}
            core = stats.get("core") or {}
            demo = stats.get("demo") or {}
            rows.append({
                "Player": name,
                "Goals": core.get("goals", 0),
                "Shots": core.get("shots", 0),
                "Saves": core.get("saves", 0),
                "Demos": demo.get("inflicted", 0),
                "Score": core.get("score", 0),
            })
    return rows


def is_replay_detail(data):
    """
    Description:
        Checks if cached object is a replay detail dictionary.
    Arguments:
        data: Cache entry value.
    Returns:
        Boolean: True if structured correctly, False otherwise.
    """
    return isinstance(data, dict) and "blue" in data and "orange" in data


def is_replay_list(data):
    """
    Description:
        Checks if cached object is a list of replays.
    Arguments:
        data: Cache entry value.
    Returns:
        Boolean: True if structured correctly, False otherwise.
    """
    return isinstance(data, dict) and "list" in data and isinstance(data.get("list"), list)


def detect_playlist(data):
    """
    Description:
        Determines the category index of the playlist (ranked-2s vs private).
    Arguments:
        data: Replay detail dictionary.
    Returns:
        Float: 0.0 for doubles/2v2, 1.0 for private/scrims.
    """
    playlist = data.get("playlist_id") or data.get("playlist_name") or ""
    if isinstance(playlist, str) and "doubles" in playlist.lower(): return 0
    if isinstance(playlist, str) and "private" in playlist.lower(): return 1
    pid = data.get("playlist_id", "")
    if pid == "ranked-doubles": return 0
    return 1


def build_training_data(cache_path=".bc_cache.json", min_lookback=5):
    """
    Description:
        Generates chronological supervised training samples from cached database replays.
    Arguments:
        cache_path: Legacy JSON cache fallback path string.
        min_lookback: History window required to build features.
    Returns:
        Tuple: (features array, labels array, metadata list of dictionaries).
    """
    initialize_database()
    all_details = get_all_replay_details()

    if not all_details:
        cache_file = Path(cache_path)
        if not cache_file.exists():
            print(f"No replay data found in DB or {cache_path}!")
            return np.zeros((0, 13)), np.zeros(0), []

        print(f"DB empty, falling back to {cache_path}...")
        with open(cache_file, "r") as f:
            cache = json.load(f)
        all_details = [
            data for data in cache.values()
            if is_replay_detail(data)
        ]

    if not all_details:
        print("No replay details found.")
        return np.zeros((0, 13)), np.zeros(0), []

    all_rows = []
    replay_playlists = {}
    for data in all_details:
        rid = data.get("id", "")
        playlist_type = detect_playlist(data)
        replay_playlists[rid] = playlist_type
        player_rows = extract_player_stats(data)
        for row in player_rows:
            row["replay_id"] = rid
            row["date"] = data.get("date", "")
            row["playlist_type"] = playlist_type
        all_rows.extend(player_rows)

    df = pd.DataFrame(all_rows)
    print(f"Found {len(df)} player-game records across {df['replay_id'].nunique()} replays")

    features_list = []
    labels_list = []
    meta_list = []

    player_groups = df.groupby("Player", dropna=True)
    processed = 0

    for player_name, p_df in player_groups:
        if len(p_df) < min_lookback + 1: continue

        p_df = p_df.sort_values("date").reset_index(drop=True)

        for game_idx in range(min_lookback, len(p_df)):
            lookback = p_df.iloc[:game_idx]
            target_game = p_df.iloc[game_idx]
            playlist_type = target_game.get("playlist_type", 1)

            for stat in STATS:
                if stat not in lookback.columns: continue

                for thresh in THRESHOLDS.get(stat, []):
                    vals = lookback[stat].values
                    avg = vals.mean()
                    hit_rate = (vals > thresh).mean()
                    std_dev = vals.std() if len(vals) > 1 else 0.0
                    stat_cap = STAT_CAPS.get(stat, 8)

                    feat = np.zeros(13, dtype=np.float32)
                    feat[0] = 0.0
                    feat[1] = 0.0
                    feat[2] = 0.0
                    feat[3] = 0.0
                    feat[4] = norm(avg, stat_cap)
                    feat[5] = hit_rate
                    feat[6] = norm(len(lookback), MAX_GAMES_GEN)
                    feat[7] = norm(std_dev, stat_cap)
                    recent = lookback.tail(min(14, len(lookback)))
                    feat[8] = norm(len(recent), MAX_GAMES_MOMENTUM)
                    feat[9] = norm(recent["Score"].mean() if "Score" in recent else 0, MAX_SCORE)
                    feat[10] = 0.5
                    feat[11] = 0.5
                    feat[12] = float(playlist_type)

                    actual = target_game[stat]
                    label = 1.0 if actual > thresh else 0.0

                    features_list.append(feat)
                    labels_list.append(label)
                    meta_list.append({
                        "player": player_name,
                        "stat": stat,
                        "threshold": thresh,
                        "actual": actual,
                        "playlist": playlist_type,
                    })
                    processed += 1

    if not features_list:
        print("Not enough player history to generate training data.")
        return np.zeros((0, 13)), np.zeros(0), []

    features_arr = np.array(features_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.float32)

    print(f"Generated {len(features_arr)} training samples from {len(player_groups)} players")
    print(f"Label distribution: {labels_arr.mean():.1%} over / {1 - labels_arr.mean():.1%} under")

    return features_arr, labels_arr, meta_list
