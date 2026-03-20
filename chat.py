import os
import datetime
import re
import math
import pandas as pd
from scrapers import Ballchasing, load_player_id_map, resolve_ids
from stats import replayStats, rankedActivity
from scrapers.h2h_ballchasing import getH2HStats
from sentiment import get_player_sentiment
from features import extract_features
from model import load_model, predict as nn_predict


def log_prediction(player, stat, threshold, is_over, num_games, nn_prob, features, team1="", team2=""):
    """Save prediction metadata to data/prediction_log.csv for future learning."""
    log_path = os.path.join("data", "prediction_log.csv")
    os.makedirs("data", exist_ok=True)
    
    # Header if file is new
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,player,stat,threshold,is_over,num_games,nn_prob,features,team1,team2\n")
            
    # Serialize features (13 floats) as a semicolon-separated string
    feat_str = ";".join([f"{x:.4f}" for x in features])
    
    timestamp = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{player},{stat},{threshold},{is_over},{num_games or 1},{nn_prob:.4f},{feat_str},{team1},{team2}\n")


def parse_query(query: str, available_players=None):
    """
    Parse queries like:
      "over 2.5 demos for Zen"
      "Will LJ get over 5 saves in 3 games?"
    Returns (player_name, stat_name, is_over, threshold, num_games)
    """
    query_lower = query.lower()

    # Over/Under
    is_over = None
    if "over" in query_lower or "o/" in query_lower:
        is_over = True
    elif "under" in query_lower or "u/" in query_lower:
        is_over = False

    # Extract "in X games/maps" first so we can exclude that number
    games_match = re.search(r'in\s+(\d+)\s+(?:games?|maps?|rounds?)', query_lower)
    num_games = int(games_match.group(1)) if games_match else None

    # Find all numbers, pick the threshold (not the games count)
    all_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+(?:\.\d+)?)', query_lower)]
    threshold = None
    for num_str, pos in all_numbers:
        # Skip the number that's part of "in X games"
        if games_match and pos == games_match.start(1):
            continue
        threshold = float(num_str)
        break  # take the first non-games number

    # Stat name
    stat = None
    for s in ["goals", "shots", "saves", "demos", "assists", "score"]:
        if s in query_lower:
            stat = s.capitalize()
            break

    # Player name — build stop words, then find the first non-stop word
    known = {"over", "under", "for", "will", "get", "in", "the", "a", "an",
             "is", "o/u", "on", "he", "she", "to", "of", "maps", "map",
             "games", "game", "rounds", "round"}
    if stat:
        known.add(stat.lower())
    for num_str, _ in all_numbers:
        known.add(num_str)

    words = [w.strip("?!.,") for w in query.split()]
    player = None

    # Strategy: if we know the available players, try to match roster names first
    # Sort by longest name first so multi-word names match before fragments
    if available_players:
        sorted_players = sorted(available_players, key=lambda x: len(str(x)), reverse=True)
        for pname in sorted_players:
            pname_str = str(pname).strip()
            if len(pname_str) < 2:
                continue  # skip broken 1-char names from replay data
            # Use word boundary matching to avoid "s" matching "goals"
            if re.search(r'\b' + re.escape(pname_str.lower()) + r'\b', query_lower):
                player = pname_str.lower()
                break

    # Fallback: find the first word that isn't a stop word (allow 2-letter names)
    if not player:
        for w in words:
            w_clean = w.lower().strip("?!.,")
            if w_clean not in known and len(w_clean) >= 2:
                player = w_clean
                break

    return player, stat, is_over, threshold, num_games


def run_chat(row, bc, idMap):
    t1, t2 = row['team1'], row['team2']
    r1, r2 = row["team1_players"], row["team2_players"]

    print("\n--- O/U Predictor Chat ---")
    print(f"Matchup: {t1} vs {t2}")
    if r1 and r2:
        print(f"Rosters: {', '.join(r1)} vs {', '.join(r2)}")
    print("Ask a question like: 'Will Zen get over 2.5 demos in 3 games?'")
    print("You can specify 'in X games/maps' for multi-game bets.")
    print("Type 'q' or 'quit' to exit.\n")

    # Load neural network model (if trained)
    nn_model = load_model()
    if nn_model:
        print("Neural Net model loaded!")
    else:
        print("No trained model found. Run 'python3 train.py' to train.")

    print("Fetching H2H history and recent general games...")
    print("(This may take a few minutes on first run — subsequent runs are cached)\n")
    logs = []

    # 1. Fetch generic recent stats (Private / Scrims)
    print("[Step 1/3] Fetching generic recent stats (scrims/private)...")
    ids1 = resolve_ids(r1, idMap)
    ids2 = resolve_ids(r2, idMap)
    all_roster_ids = ids1 + ids2
    gen_df = replayStats(bc, all_roster_ids, logs)

    # 2. Fetch Ranked Grind stats (2s volume)
    print("[Step 2/3] Fetching ranked 2s momentum...")
    ranked_momentum = rankedActivity(bc, all_roster_ids, logs)

    # 3. Fetch H2H specific stats
    print("[Step 3/3] Fetching head-to-head history...")
    h2h_df, h2h_logs = getH2HStats(t1, t2, r1, r2, bc)
    print("Done!\n")

    if gen_df.empty and h2h_df.empty:
        print("No replay data found anywhere! Exiting chat.")
        return

    print(f"Loaded {len(gen_df)} generic player game records!")
    print(f"Loaded {len(h2h_df)} player game records from direct H2H history!")

    # Roster continuity check on H2H data
    confident_h2h = False
    if not h2h_df.empty:
        h2h_players = set(h2h_df['Player'].str.lower().unique())
        expected_players = set([str(p).lower() for p in (r1 + r2)])
        overlap = h2h_players.intersection(expected_players)

        if len(expected_players) > 0 and len(overlap) >= 4:
            confident_h2h = True

        if confident_h2h:
            print("High Confidence: H2H data heavily matches the current rosters!")
        else:
            print("Low Confidence: H2H history features mostly old or different rosters.")

    # Combine available names to help resolver
    av_names = set()
    if not gen_df.empty: av_names.update(gen_df['Player'].dropna().unique())
    if not h2h_df.empty: av_names.update(h2h_df['Player'].dropna().unique())
    player_map = {str(p).lower(): str(p) for p in av_names}

    while True:
        q = input("\nQuery: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break

        player_query, stat, is_over, threshold, num_games = parse_query(q, available_players=av_names)

        if not player_query or not stat or threshold is None or is_over is None:
            print("Could not parse query. Make sure to include Over/Under, a number, a stat (goals, saves, shots, demos), and a player.")
            print("Example: 'Will LJ get over 4 saves in 3 games?'")
            continue

        # Resolve player name
        matched_player = None
        if player_query.lower() in player_map:
            matched_player = player_map[player_query.lower()]
        else:
            for p_lower, p_real in player_map.items():
                if player_query.lower() in p_lower:
                    matched_player = p_real
                    break

        if not matched_player:
            print(f"Could not find a player matching '{player_query}'. Available: {', '.join(av_names)}")
            continue

        if (not gen_df.empty and stat not in gen_df.columns) and (not h2h_df.empty and stat not in h2h_df.columns):
            print(f"Stat '{stat}' is not available. Try one of: Goals, Shots, Saves, Demos")
            continue

        # ─── Gather all stats silently ─────────────────────────
        # H2H stats
        prob_h2h = None
        prob_gen = None
        per_game_avg_h2h = None
        per_game_avg_gen = None
        h2h_games = 0
        gen_games = 0

        if not h2h_df.empty and matched_player in h2h_df['Player'].values:
            p_data = h2h_df[h2h_df["Player"] == matched_player]
            h2h_games = len(p_data)
            if h2h_games > 0 and stat in p_data.columns:
                stat_values = p_data[stat]
                per_game_avg_h2h = stat_values.mean()
                hits = (stat_values > threshold).sum() if is_over else (stat_values < threshold).sum()
                prob_h2h = (hits / h2h_games) * 100

        # Generic form stats
        if not gen_df.empty and matched_player in gen_df['Player'].values:
            p_data = gen_df[gen_df["Player"] == matched_player]
            gen_games = len(p_data)
            if gen_games > 0 and stat in p_data.columns:
                stat_values = p_data[stat]
                per_game_avg_gen = stat_values.mean()
                hits = (stat_values > threshold).sum() if is_over else (stat_values < threshold).sum()
                prob_gen = (hits / gen_games) * 100

        if (not gen_df.empty and stat not in gen_df.columns) and (not h2h_df.empty and stat not in h2h_df.columns):
            print(f"Stat '{stat}' is not available. Try one of: Goals, Shots, Saves, Demos")
            continue

        # Sentiment
        sent_data = get_player_sentiment(matched_player)
        s_score = sent_data["score"]
        s_status = sent_data["status"]

        # Momentum — resolve player name through idMap
        p_canon = matched_player.strip().lower()
        aliases = idMap.get("aliases", {})
        players_table = idMap.get("players", {})
        resolved = aliases.get(p_canon, p_canon)
        player_ids = players_table.get(resolved, [])

        p_momentum = None
        momentum_label = "No data"
        for pid in player_ids:
            mdata = ranked_momentum.get(pid)
            if mdata and mdata.get("games", 0) > 0:
                p_momentum = mdata
                g = mdata["games"]
                momentum_label = f"{'High' if g >= 20 else 'Moderate' if g >= 5 else 'Cold'} ({g} games, {mdata['win_rate']}% WR)"
                break

        # Multi-game projection
        projected_total = None
        best_avg = per_game_avg_h2h if per_game_avg_h2h is not None else per_game_avg_gen
        if best_avg is not None and num_games and num_games > 1:
            projected_total = best_avg * num_games

        # ─── Neural Net or Heuristic ───────────────────────────
        nn_model_used = False
        display_prob = None
        nn_prob = None
        feat_vec = None
        ou_label = "Over" if is_over else "Under"
        games_str = f" in {num_games} games" if num_games and num_games > 1 else ""

        if nn_model:
            nn_model_used = True
            nn_threshold = threshold / num_games if (num_games is not None and num_games > 1) else threshold

            feat_vec = extract_features(
                player_name=matched_player,
                stat_name=stat,
                threshold=nn_threshold,
                h2h_df=h2h_df,
                gen_df=gen_df,
                momentum_data=p_momentum,
                sentiment_data=sent_data,
                confident_h2h=confident_h2h,
                playlist_type=1,
            )
            nn_prob = nn_predict(nn_model, feat_vec)
            display_prob = nn_prob if is_over else (1.0 - nn_prob)
        else:
            # Fallback: use heuristic probability
            best_prob = prob_h2h if prob_h2h is not None else prob_gen
            if best_prob is not None:
                display_prob = best_prob / 100.0

        # ─── Determine Pick ────────────────────────────────────
        if display_prob is not None:
            if display_prob >= 0.5:
                pick = ou_label
                pick_prob = display_prob
            else:
                pick = "Under" if is_over else "Over"
                pick_prob = 1.0 - display_prob
        else:
            pick = "?"
            pick_prob = None

        confidence = "High" if pick_prob and abs(pick_prob - 0.5) > 0.2 else "Medium" if pick_prob and abs(pick_prob - 0.5) > 0.1 else "Low"

        # ─── Print Bet Card ────────────────────────────────────
        print(f"\n{'═' * 50}")
        print(f"  {matched_player} — {ou_label} {threshold} {stat}{games_str}")
        print(f"{'═' * 50}")

        # Pick
        if pick_prob is not None:
            print(f"\n  Pick:       {pick.upper()}")
            print(f"  Chance:     {pick_prob:.1%}")
            print(f"  Confidence: {confidence}")
        else:
            print(f"\n  Pick:       Insufficient data")

        # Why — Stats Reasoning
        print(f"\n  Why?")
        reasons = []

        if per_game_avg_h2h is not None:
            avg_str = f"{per_game_avg_h2h:.1f}"
            if projected_total is not None:
                reasons.append(f"    H2H avg: {avg_str} {stat.lower()}/game ({h2h_games} games) → ~{projected_total:.1f} projected across {num_games}")
            else:
                reasons.append(f"    H2H avg: {avg_str} {stat.lower()}/game ({h2h_games} games)")
            if confident_h2h:
                reasons.append(f"    H2H rosters match current lineups (high confidence)")

        if per_game_avg_gen is not None:
            avg_str = f"{per_game_avg_gen:.1f}"
            if projected_total is not None and per_game_avg_h2h is None:
                reasons.append(f"    Recent avg: {avg_str} {stat.lower()}/game ({gen_games} games) → ~{projected_total:.1f} projected across {num_games}")
            else:
                reasons.append(f"    Recent avg: {avg_str} {stat.lower()}/game ({gen_games} games)")

        if prob_h2h is not None:
            per_game_thresh = threshold / num_games if num_games and num_games > 1 else threshold
            reasons.append(f"    H2H hit rate: {prob_h2h:.0f}% of games had >{per_game_thresh:.1f} {stat.lower()}")

        reasons.append(f"    Ranked 2s momentum: {momentum_label}")
        reasons.append(f"    Reddit sentiment: {s_score:+.2f} ({s_status})")

        for r in reasons:
            print(r)

        print(f"{'─' * 50}")

        # Log the prediction — is_over reflects the NN's pick, not the user's question
        if nn_model_used and feat_vec is not None and nn_prob is not None:
            nn_picks_over = nn_prob >= 0.5
            log_prediction(
                matched_player, stat, threshold, nn_picks_over, num_games,
                nn_prob, feat_vec, team1=t1, team2=t2
            )
        elif not nn_model_used:
            print(f"\n  ⚠ No trained model — run: python train.py")
