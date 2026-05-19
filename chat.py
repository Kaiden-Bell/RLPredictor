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
    
    log_path = os.path.join("data", "prediction_log.csv")
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,player,stat,threshold,is_over,num_games,nn_prob,features,team1,team2\n")
            

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

    is_over = None
    if "over" in query_lower or "o/" in query_lower: 
        is_over = True
    elif "under" in query_lower or "u/" in query_lower: 
        is_over = False

    games_match = re.search(r'in\s+(\d+)\s+(?:games?|maps?|rounds?)', query_lower)
    num_games = int(games_match.group(1)) if games_match else None

    all_numbers = [(m.group(1), m.start()) for m in re.finditer(r'(\d+(?:\.\d+)?)', query_lower)]
    threshold = None
    for num_str, pos in all_numbers:
        if games_match and pos == games_match.start(1): continue
        threshold = float(num_str)
        break 

    stat = None
    for s in ["goals", "shots", "saves", "demos", "assists", "score"]:
        if s in query_lower:
            stat = s.capitalize()
            break

    # Collection of stop words
    known = {"over", "under", "for", "will", "get", "in", "the", "a", "an",
             "is", "o/u", "on", "he", "she", "to", "of", "maps", "map",
             "games", "game", "rounds", "round"}
    if stat:
        known.add(stat.lower())
    for num_str, _ in all_numbers:
        known.add(num_str)

    words = [w.strip("?!.,") for w in query.split()]
    player = None

    if available_players:
        sorted_players = sorted(available_players, key=lambda x: len(str(x)), reverse=True)
        for pname in sorted_players:
            pname_str = str(pname).strip()
            if len(pname_str) < 2: continue
            if re.search(r'\b' + re.escape(pname_str.lower()) + r'\b', query_lower):
                player = pname_str.lower()
                break
    
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

    nn_model = load_model()
    if nn_model:
        print("Neural Net model loaded!")
    else:
        print("No trained model found. Run 'python3 train.py' to train.")

    print("Fetching H2H history and recent general games...")
    print("(This may take a few minutes on first run — subsequent runs are cached)\n")
    logs = []

    print("[Step 1/3] Fetching generic recent stats (scrims/private)...")
    ids1 = resolve_ids(r1, idMap)
    ids2 = resolve_ids(r2, idMap)
    all_roster_ids = ids1 + ids2
    gen_df = replayStats(bc, all_roster_ids, logs)

    print("[Step 2/3] Fetching ranked 2s momentum...")
    ranked_momentum = rankedActivity(bc, all_roster_ids, logs)

    print("[Step 3/3] Fetching head-to-head history...")
    h2h_df, h2h_logs = getH2HStats(t1, t2, r1, r2, bc)
    print("Done!\n")

    if gen_df.empty and h2h_df.empty:
        print("No replay data found anywhere! Exiting chat.")
        return

    print(f"Loaded {len(gen_df)} generic player game records!")
    print(f"Loaded {len(h2h_df)} player game records from direct H2H history!")

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
    
    print("Ask a question like: 'Zen o/u 2.5 demos in X games?'")
    print("Type 'q' or 'quit' to exit.\n")

    from utils.player_identity import get_available_players, normalize_player_name
    canonical_players = get_available_players()
    
    present_canonical_ids = set()
    if not gen_df.empty: present_canonical_ids.update(gen_df['canonical_player_id'].dropna().unique())
    if not h2h_df.empty: present_canonical_ids.update(h2h_df['canonical_player_id'].dropna().unique())
    
    alias_lookup = {}
    av_names = set()
    for p in canonical_players:
        cid = p["canonical_player_id"]
        if cid not in present_canonical_ids:
            continue
        cname = p["canonical_name"]
        av_names.add(cname)
        alias_lookup[cid] = p
        alias_lookup[normalize_player_name(cname)] = p
        for alias in p.get("aliases", []):
            alias_lookup[normalize_player_name(alias)] = p

    while True:
        q = input("\nQuery: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break

        player_query, stat, is_over, threshold, num_games = parse_query(q, available_players=av_names)

        if not player_query or not stat or threshold is None or is_over is None:
            print("Could not parse query. Make sure to include Over/Under, a number, a stat (goals, saves, shots, demos), and a player.")
            print("Example: 'LJ o/u 4 saves in X games?'")
            continue

        matched_player_dict = None
        norm_q = normalize_player_name(player_query)
        if norm_q in alias_lookup:
            matched_player_dict = alias_lookup[norm_q]
        else:
            for n_alias, p_dict in alias_lookup.items():
                if norm_q in n_alias or n_alias in norm_q:
                    matched_player_dict = p_dict
                    break

        if not matched_player_dict:
            print(f"Could not find '{player_query}'.")
            print("Available canonical players:")
            for p in canonical_players:
                if p["canonical_player_id"] in present_canonical_ids:
                    aliases_str = f"  aliases: {', '.join(p['aliases'][:3])}" if p['aliases'] else ""
                    print(f"- {p['canonical_name']}{aliases_str}")
            continue

        matched_player_id = matched_player_dict["canonical_player_id"]
        matched_player_name = matched_player_dict["canonical_name"]

        if (not gen_df.empty and stat not in gen_df.columns) and (not h2h_df.empty and stat not in h2h_df.columns):
            print(f"Stat '{stat}' is not available. Try one of: Goals, Shots, Saves, Demos")
            continue
        
        # -----------
        # H2H stats |
        # -----------
        prob_h2h = None
        prob_gen = None
        per_game_avg_h2h = None
        per_game_avg_gen = None
        h2h_games = 0
        gen_games = 0

        if not h2h_df.empty and matched_player_id in h2h_df['canonical_player_id'].values:
            p_data = h2h_df[h2h_df["canonical_player_id"] == matched_player_id]
            h2h_games = len(p_data)
            if h2h_games > 0 and stat in p_data.columns:
                stat_values = p_data[stat]
                per_game_avg_h2h = stat_values.mean()
                hits = (stat_values > threshold).sum() if is_over else (stat_values < threshold).sum()
                prob_h2h = (hits / h2h_games) * 100

        # -------------------
        # Generic Statstics |
        # -------------------
        if not gen_df.empty and matched_player_id in gen_df['canonical_player_id'].values:
            p_data = gen_df[gen_df["canonical_player_id"] == matched_player_id]
            gen_games = len(p_data)
            if gen_games > 0 and stat in p_data.columns:
                stat_values = p_data[stat]
                per_game_avg_gen = stat_values.mean()
                hits = (stat_values > threshold).sum() if is_over else (stat_values < threshold).sum()
                prob_gen = (hits / gen_games) * 100

        if (not gen_df.empty and stat not in gen_df.columns) and (not h2h_df.empty and stat not in h2h_df.columns):
            print(f"Stat '{stat}' is not available. Try one of: Goals, Shots, Saves, Demos")
            continue

        # --------------------
        # Sentiment Analysis |
        # --------------------
        search_names = [matched_player_name] + matched_player_dict.get("aliases", [])
        sent_data = get_player_sentiment(search_names)
        s_score = sent_data["score"]
        s_status = sent_data["status"]

        # -----------------
        # Momentum / Form |
        # -----------------
        players_table = idMap.get("players", {})
        player_ids = players_table.get(matched_player_id, [])

        p_momentum = None
        momentum_label = "No data"
        for pid in player_ids:
            mdata = ranked_momentum.get(pid)
            if mdata and mdata.get("games", 0) > 0:
                p_momentum = mdata
                g = mdata["games"]
                momentum_label = f"{'High' if g >= 20 else 'Moderate' if g >= 5 else 'Cold'} ({g} games, {mdata['win_rate']}% WR)"
                break

        projected_total = None
        best_avg = per_game_avg_h2h if per_game_avg_h2h is not None else per_game_avg_gen
        if best_avg is not None and num_games and num_games > 1:
            projected_total = best_avg * num_games

        # -------------------------
        # Neural Net or Heuristic |
        # -------------------------
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
                player_name=matched_player_name,
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
            best_prob = prob_h2h if prob_h2h is not None else prob_gen
            if best_prob is not None:
                display_prob = best_prob / 100.0

        # ----------------
        # Determine Pick |
        # ----------------
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

        # ----------------
        # Print Bet Card |
        # ----------------
        print(f"\n{'═' * 50}")
        print(f"  {matched_player_name} — {ou_label} {threshold} {stat}{games_str}")
        print(f"{'═' * 50}")

        if pick_prob is not None:
            print(f"\n  Pick:       {pick.upper()}")
            print(f"  Chance:     {pick_prob:.1%}")
            print(f"  Confidence: {confidence}")
        else:
            print(f"\n  Pick:       Insufficient data")

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

        # Log the prediction — is_over reflects the NN's pick, not the user's questioner reflects the NN's pick, not the user's question
        if nn_model_used and feat_vec is not None and nn_prob is not None:
            nn_picks_over = nn_prob >= 0.5
            log_prediction(
                matched_player_name, stat, threshold, nn_picks_over, num_games,
                nn_prob, feat_vec, team1=t1, team2=t2
            )
        elif not nn_model_used:
            print(f"\n  No trained model — run: python train.py")
