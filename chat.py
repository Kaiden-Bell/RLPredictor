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


def log_prediction(player, stat, threshold, is_over, num_games, nn_prob, features):
    """Save prediction metadata to data/prediction_log.csv for future learning."""
    log_path = os.path.join("data", "prediction_log.csv")
    os.makedirs("data", exist_ok=True)
    
    # Header if file is new
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,player,stat,threshold,is_over,num_games,nn_prob,features\n")
            
    # Serialize features (13 floats) as a semicolon-separated string
    feat_str = ";".join([f"{x:.4f}" for x in features])
    
    timestamp = datetime.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{player},{stat},{threshold},{is_over},{num_games or 1},{nn_prob:.4f},{feat_str}\n")


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

        # Display target
        condition_str = f"Over {threshold}" if is_over else f"Under {threshold}"
        games_str = f" in {num_games} games" if num_games else " (per game)"
        print(f"\nTarget: {matched_player} - {condition_str} {stat}{games_str}")

        # H2H calculation (per-game probability)
        prob_h2h = None
        prob_gen = None
        per_game_avg_h2h = None
        per_game_avg_gen = None

        if not h2h_df.empty and matched_player in h2h_df['Player'].values:
            p_data = h2h_df[h2h_df["Player"] == matched_player]
            games_played = len(p_data)
            if games_played > 0:
                stat_values = p_data[stat]
                per_game_avg_h2h = stat_values.mean()
                hits = (stat_values > threshold).sum() if is_over else (stat_values < threshold).sum()
                prob_h2h = (hits / games_played) * 100
                print(f"H2H per-game: {prob_h2h:.1f}% ({hits}/{games_played} games) | Avg {stat}: {per_game_avg_h2h:.2f}/game - Confident: {confident_h2h}")

        # Generic calculation (per-game probability)
        if not gen_df.empty and matched_player in gen_df['Player'].values:
            p_data = gen_df[gen_df["Player"] == matched_player]
            games_played = len(p_data)
            if games_played > 0:
                stat_values = p_data[stat]
                per_game_avg_gen = stat_values.mean()
                hits = (stat_values > threshold).sum() if is_over else (stat_values < threshold).sum()
                prob_gen = (hits / games_played) * 100
                print(f"Generic per-game: {prob_gen:.1f}% ({hits}/{games_played} games) | Avg {stat}: {per_game_avg_gen:.2f}/game")

        # Multi-game projection
        if num_games and num_games > 1:
            best_avg = per_game_avg_h2h if per_game_avg_h2h is not None else per_game_avg_gen
            best_prob = prob_h2h if prob_h2h is not None else prob_gen

            if best_avg is not None and best_prob is not None:
                projected_total = best_avg * num_games

                # Adjusted probability: scale based on projected total vs threshold
                if is_over:
                    if projected_total > threshold:
                        multi_prob = min(95.0, best_prob + (projected_total - threshold) / max(projected_total, 1) * 30)
                    else:
                        multi_prob = max(5.0, best_prob - (threshold - projected_total) / max(threshold, 1) * 30)
                else:
                    if projected_total < threshold:
                        multi_prob = min(95.0, best_prob + (threshold - projected_total) / max(threshold, 1) * 30)
                    else:
                        multi_prob = max(5.0, best_prob - (projected_total - threshold) / max(projected_total, 1) * 30)

                print(f"Multi-game projection ({num_games} games): ~{projected_total:.1f} total {stat} expected | Adj. Probability: ~{multi_prob:.1f}%")

        # Sentiment Analysis
        sent_data = get_player_sentiment(matched_player)
        s_score = sent_data["score"]
        s_status = sent_data["status"]
        s_count = sent_data["count"]
        print(f"Reddit Sentiment: {s_score:+.2f} ({s_status} across {s_count} recent posts/threads)")

        # Momentum Check — resolve player name through idMap properly
        p_canon = matched_player.strip().lower()
        aliases = idMap.get("aliases", {})
        players_table = idMap.get("players", {})
        resolved = aliases.get(p_canon, p_canon)
        player_ids = players_table.get(resolved, [])

        momentum_shown = False
        for pid in player_ids:
            mdata = ranked_momentum.get(pid)
            if mdata and mdata.get("games", 0) > 0:
                g = mdata["games"]
                avg = mdata["avg_score"]
                wr = mdata["win_rate"]
                if g >= 20:
                    label = "High"
                elif g >= 5:
                    label = "Moderate"
                else:
                    label = "Cold"
                print(f"{label} Ranked 2s Momentum: {g} games (last 14d) | Avg Score: {avg} | Win Rate: {wr}%")
                momentum_shown = True
                break
        if not momentum_shown:
            print(f"Ranked 2s Momentum: No recent 2s data found")

        # Neural Net Prediction
        # Resolve momentum data for this player
        p_momentum = None
        for pid in player_ids:
            mdata = ranked_momentum.get(pid)
            if mdata and mdata.get("games", 0) > 0:
                p_momentum = mdata
                break

        if nn_model:
            feat_vec = extract_features(
                player_name=matched_player,
                stat_name=stat,
                threshold=threshold,
                h2h_df=h2h_df,
                gen_df=gen_df,
                momentum_data=p_momentum,
                sentiment_data=sent_data,
                confident_h2h=confident_h2h,
                playlist_type=1,  # inference is always for RLCS/private context
            )
            nn_prob = nn_predict(nn_model, feat_vec)

            # Adjust for over vs under
            display_prob = nn_prob if is_over else (1.0 - nn_prob)
            confidence = "High" if abs(display_prob - 0.5) > 0.2 else "Medium" if abs(display_prob - 0.5) > 0.1 else "Low"

            ou_label = "Over" if is_over else "Under"
            print(f"\nNeural Net Prediction: {display_prob:.1%} chance of {ou_label} {threshold} {stat}")
            print(f"Confidence: {confidence}")

            # Log the prediction
            log_prediction(
                matched_player, stat, threshold, is_over, num_games, 
                nn_prob, feat_vec
            )

            # Smart betting advice based on NN + heuristics
            if display_prob > 0.65:
                print(f"Suggestion: {ou_label} looks strong — {matched_player} has {display_prob:.0%} predicted probability.")
            elif display_prob < 0.35:
                flip_label = "Under" if is_over else "Over"
                print(f"Suggestion: Lean {flip_label} instead — only {display_prob:.0%} chance of {ou_label}.")
            else:
                print(f"Suggestion: This is close to a coin flip — proceed with caution.")
        else:
            # Fallback heuristic advice when no model is trained
            target_prob = prob_h2h if prob_h2h is not None else prob_gen
            if target_prob is not None:
                if is_over and target_prob < 40 and s_score < 0:
                    print(f"\nSuggestion: Lean UNDER — {matched_player}'s hit rate is only {target_prob:.1f}% and sentiment is negative ({s_score:+.2f}).")
                elif is_over and target_prob < 40:
                    print(f"\nSuggestion: Lean UNDER — {matched_player}'s hit rate is only {target_prob:.1f}%.")
                elif not is_over and target_prob < 40 and s_score > 0:
                    print(f"\nSuggestion: Lean OVER — under-hit rate is only {target_prob:.1f}% and sentiment is positive ({s_score:+.2f}).")
                elif not is_over and target_prob < 40:
                    print(f"\nSuggestion: Lean OVER — under-hit rate is only {target_prob:.1f}%.")
            print(f"\nTrain the neural net for better predictions: python3 train.py")
