"""
features.py — Feature extraction for the RLPredictor neural network.

Transforms raw replay data into a 13-dimensional feature vector and
generates self-supervised training rows from cached replay history.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Feature normalization caps
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


def _norm(val, cap):
    """Normalize a value to [0, 1] using a cap."""
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
    playlist_type=1,  # 0=ranked-2s, 1=private/scrims
):
    """
    Build the 13-dimensional feature vector for a single prediction query.

    Returns: np.ndarray of shape (13,)
    """
    stat_cap = STAT_CAPS.get(stat_name, 8)
    features = np.zeros(13, dtype=np.float32)

    # --- H2H features (1-4) ---
    if h2h_df is not None and not h2h_df.empty and player_name in h2h_df["Player"].values:
        p = h2h_df[h2h_df["Player"] == player_name]
        n = len(p)
        if n > 0 and stat_name in p.columns:
            vals = p[stat_name].values
            features[0] = _norm(vals.mean(), stat_cap)        # h2h_avg_stat
            features[1] = (vals > threshold).mean()            # h2h_hit_rate
            features[2] = _norm(n, MAX_GAMES_H2H)             # h2h_games
    features[3] = 1.0 if confident_h2h else 0.0               # h2h_confident

    # --- Generic form features (5-8) ---
    if gen_df is not None and not gen_df.empty and player_name in gen_df["Player"].values:
        p = gen_df[gen_df["Player"] == player_name]
        n = len(p)
        if n > 0 and stat_name in p.columns:
            vals = p[stat_name].values
            features[4] = _norm(vals.mean(), stat_cap)         # gen_avg_stat
            features[5] = (vals > threshold).mean()            # gen_hit_rate
            features[6] = _norm(n, MAX_GAMES_GEN)              # gen_games
            features[7] = _norm(vals.std(), stat_cap)           # gen_std_dev

    # --- Momentum features (9-11) ---
    if momentum_data and isinstance(momentum_data, dict):
        features[8] = _norm(momentum_data.get("games", 0), MAX_GAMES_MOMENTUM)
        features[9] = _norm(momentum_data.get("avg_score", 0), MAX_SCORE)
        features[10] = momentum_data.get("win_rate", 0) / 100.0

    # --- Sentiment (12) ---
    if sentiment_data and isinstance(sentiment_data, dict):
        # VADER score is -1 to +1, shift to 0–1
        features[11] = (sentiment_data.get("score", 0.0) + 1.0) / 2.0

    # --- Playlist type (13) ---
    features[12] = float(playlist_type)

    return features


def _extract_player_stats(detail):
    """Extract per-player stats from a single replay detail dict."""
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


def _is_replay_detail(data):
    """Check if a cached response looks like a replay detail (has blue/orange teams)."""
    return isinstance(data, dict) and "blue" in data and "orange" in data


def _is_replay_list(data):
    """Check if a cached response looks like a replay list."""
    return isinstance(data, dict) and "list" in data and isinstance(data.get("list"), list)


def _detect_playlist(data):
    """Detect playlist type from a replay detail. Returns 0 for ranked-2s, 1 for private."""
    playlist = data.get("playlist_id") or data.get("playlist_name") or ""
    if isinstance(playlist, str) and "doubles" in playlist.lower():
        return 0
    if isinstance(playlist, str) and "private" in playlist.lower():
        return 1
    # playlist_id: ranked-doubles = specific IDs, private = "private"
    pid = data.get("playlist_id", "")
    if pid == "ranked-doubles":
        return 0
    return 1  # default to private/scrims


def build_training_data(cache_path=".bc_cache.json", min_lookback=5):
    """
    Generate self-supervised training rows from cached Ballchasing replays.

    For each player found in the cache:
      - Collect their game-by-game stats
      - For each game N (where N >= min_lookback):
        - Use games 0..N-1 as the "history" (features)
        - Use game N as the "label" (did stat > threshold?)
      - Try multiple stat/threshold combos for each game

    Returns: (features_array, labels_array, metadata_list)
      features_array: np.ndarray of shape (num_samples, 13)
      labels_array:   np.ndarray of shape (num_samples,) — 0 or 1
      metadata_list:  list of dicts with player/stat/threshold info
    """
    # Load cache
    cache_file = Path(cache_path)
    if not cache_file.exists():
        print(f"⚠️  Cache file {cache_path} not found!")
        return np.zeros((0, 13)), np.zeros(0), []

    with open(cache_file, "r") as f:
        cache = json.load(f)

    # Step 1: Extract all replay details from cache
    all_rows = []
    replay_playlists = {}

    for key, data in cache.items():
        if _is_replay_detail(data):
            rid = data.get("id", key)
            playlist_type = _detect_playlist(data)
            replay_playlists[rid] = playlist_type
            player_rows = _extract_player_stats(data)
            for row in player_rows:
                row["replay_id"] = rid
                row["date"] = data.get("date", "")
                row["playlist_type"] = playlist_type
            all_rows.extend(player_rows)

    if not all_rows:
        print("⚠️  No replay details found in cache.")
        return np.zeros((0, 13)), np.zeros(0), []

    df = pd.DataFrame(all_rows)
    print(f"📊 Found {len(df)} player-game records across {df['replay_id'].nunique()} replays")

    # Step 2: Group by player and generate training rows
    features_list = []
    labels_list = []
    meta_list = []

    player_groups = df.groupby("Player", dropna=True)
    processed = 0

    for player_name, p_df in player_groups:
        if len(p_df) < min_lookback + 1:
            continue  # not enough history

        # Sort by date (oldest first for chronological lookback)
        p_df = p_df.sort_values("date").reset_index(drop=True)

        for game_idx in range(min_lookback, len(p_df)):
            lookback = p_df.iloc[:game_idx]
            target_game = p_df.iloc[game_idx]
            playlist_type = target_game.get("playlist_type", 1)

            # Try each stat/threshold combination
            for stat in STATS:
                if stat not in lookback.columns:
                    continue

                for thresh in THRESHOLDS.get(stat, []):
                    # Build features from lookback
                    vals = lookback[stat].values
                    avg = vals.mean()
                    hit_rate = (vals > thresh).mean()
                    std_dev = vals.std() if len(vals) > 1 else 0.0
                    stat_cap = STAT_CAPS.get(stat, 8)

                    feat = np.zeros(13, dtype=np.float32)
                    # H2H features — set to 0 (no H2H context in training)
                    feat[0] = 0.0
                    feat[1] = 0.0
                    feat[2] = 0.0
                    feat[3] = 0.0
                    # Generic form features (from lookback)
                    feat[4] = _norm(avg, stat_cap)
                    feat[5] = hit_rate
                    feat[6] = _norm(len(lookback), MAX_GAMES_GEN)
                    feat[7] = _norm(std_dev, stat_cap)
                    # Momentum — approximate from recent lookback
                    recent = lookback.tail(min(14, len(lookback)))
                    feat[8] = _norm(len(recent), MAX_GAMES_MOMENTUM)
                    feat[9] = _norm(recent["Score"].mean() if "Score" in recent else 0, MAX_SCORE)
                    feat[10] = 0.5  # no win/loss info in training rows
                    # Sentiment — default neutral
                    feat[11] = 0.5
                    # Playlist type
                    feat[12] = float(playlist_type)

                    # Label: did the player hit the over in the target game?
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
        print("⚠️  Not enough player history to generate training data.")
        return np.zeros((0, 13)), np.zeros(0), []

    features_arr = np.array(features_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.float32)

    print(f"✅ Generated {len(features_arr)} training samples from {len(player_groups)} players")
    print(f"   Label distribution: {labels_arr.mean():.1%} over / {1 - labels_arr.mean():.1%} under")

    return features_arr, labels_arr, meta_list
