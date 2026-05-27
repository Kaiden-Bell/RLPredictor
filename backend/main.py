"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Main CLI interface to scrape brackets, build features, run H2H analysis, or start the predictive chatbot.
- Usage: Executed via command-line (python3 main.py <url> --mode [h2h|features|chat]).
"""

import argparse
import os
import time

from dotenv import load_dotenv
import pandas as pd

from scrapers import (
    scrape_playoffs,
    Ballchasing,
    get_h2h_stats,
    load_player_id_map,
    resolve_ids,
)
from scrapers.h2h_ballchasing import aggregate_players
from chat import run_chat
from stats import build_feat_rows
from utils.database import initialize_database


load_dotenv()


def list_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        Filters scraped matchup frames for concrete opponent rows and prints them.
    Arguments:
        df: Input raw scraped matches DataFrame.
    Returns:
        Pandas DataFrame containing filtered matches list.
    """
    mask = df["team1"].notna() & df["team2"].notna()
    matches = df[mask].reset_index(drop=True).copy()
    if matches.empty:
        print("No concrete matchups yet.")
        return matches

    completed = []
    upcoming = []

    for i, r in matches.iterrows():
        s1 = r.get("team1_score")
        s2 = r.get("team2_score")
        if pd.notna(s1) and pd.notna(s2) and str(s1).strip() != "" and str(s2).strip() != "": completed.append((i, r))
        else: upcoming.append((i, r))

    if completed:
        print("\nCompleted Matches:")
        for i, r in completed:
            sec = r.get("section") or ""
            rnd = r.get("round") or ""
            s1, s2 = r.get("team1_score", ""), r.get("team2_score", "")
            print(f"[{i}] {r['team1']} [{s1}] vs [{s2}] {r['team2']}   | {sec} {rnd}".rstrip())

    if upcoming:
        print("\nUpcoming Matches:")
        for i, r in upcoming:
            sec = r.get("section") or ""
            rnd = r.get("round") or ""
            print(f"[{i}] {r['team1']}  vs  {r['team2']}   | {sec} {rnd}".rstrip())
            
    print("")
    return matches


def choose_match_interactive(matches: pd.DataFrame) -> pd.Series | None:
    """
    Description:
        Prompts user to select a match number from the console list.
    Arguments:
        matches: Filtered matchups DataFrame.
    Returns:
        Optional match row Series or None.
    """
    while True:
        sel = input("Enter a match number (or 'q' to quit): ").strip().lower()
        if sel in {"q", "quit", "exit"}: return None
        if sel.isdigit():
            i = int(sel)
            if 0 <= i < len(matches): return matches.iloc[i]
        print(f"Invalid selection. Choose 0–{len(matches)-1}, or 'q' to quit.")


def preselect_match(matches: pd.DataFrame, match_arg: str) -> pd.Series | None:
    """
    Description:
        Attempts to pre-select a match from list based on index or team name.
    Arguments:
        matches: Filtered matchups DataFrame.
        match_arg: Selector string (index digit or team name fragment).
    Returns:
        Optional match row Series or None.
    """
    s = match_arg.strip()
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(matches): return matches.iloc[i]
        print(f"--match index {i} out of range (0..{len(matches)-1}).")
        return None

    s_low = s.lower()
    mask = matches.apply(
        lambda r: s_low in str(r["team1"]).lower() or s_low in str(r["team2"]).lower(),
        axis=1,
    )
    found = matches[mask]
    if found.empty:
        print(f"--match '{match_arg}' did not match any team names.")
        return None
    if len(found) > 1: print(f"--match '{match_arg}' matched multiple rows; picking the first.")
    return found.iloc[0]


def run_h2h(row: pd.Series, bc: Ballchasing):
    """
    Description:
        Queries direct H2H history and prints aggregated player statistics comparison.
    Arguments:
        row: Selected match row Series.
        bc: Ballchasing client.
    Returns:
        None
    """
    t1, t2 = row["team1"], row["team2"]
    r1, r2 = row["team1_players"], row["team2_players"]

    print(f"\nH2H comparison: {t1} vs {t2}\n")
    stats_raw, logs = get_h2h_stats(t1, t2, r1, r2, bc)
    
    stats = aggregate_players(stats_raw.to_dict('records') if not stats_raw.empty else [])
    
    print(stats if not stats.empty else "No stats found.")
    if logs:
        print("\nLogs:")
        for l in logs[:10]: print("-", l)


def run_features(row: pd.Series, bc: Ballchasing):
    """
    Description:
        Builds team-level prediction vectors for both sides of select match row.
    Arguments:
        row: Selected match row Series.
        bc: Ballchasing client.
    Returns:
        None
    """
    id_map = load_player_id_map()
    logs = []
    r1, r2 = build_feat_rows(bc, row, resolve_ids, id_map, logs)
    out = pd.DataFrame([r1, r2])
    print(out)
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/features_playoffs_selected.csv", index=False)
    print("\nSaved to data/features_playoffs_selected.csv\n")
    if logs:
        print("Logs:")
        for l in logs[:12]: print("-", l)


def main():
    """
    Description:
        Main pipeline orchestrator parsing choices and routing matching tasks.
    Arguments:
        None
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="RL PredictorBot — scrape Liquipedia and fetch stats."
    )
    parser.add_argument(
        "url",
        help="Liquipedia tournament URL (e.g. https://liquipedia.net/rocketleague/Esports_World_Cup/2025)",
    )
    parser.add_argument(
        "--mode",
        choices=["h2h", "features", "chat"],
        default="features",
        help="Choose 'h2h' for head-to-head, 'features' for feature build, or 'chat' for O/U chat.",
    )
    parser.add_argument(
        "--match",
        help="Preselect a match by index (e.g., 0) or team substring (e.g., 'Karmine'). If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--section",
        nargs="+",
        default=None,
        help="Section(s) to scrape: 'group', 'playoff', 'swiss', or 'all'. Examples: --section group playoff, --section all. Default: all.",
    )

    args = parser.parse_args()

    initialize_database()

    bc = Ballchasing()

    sections = args.section
    if sections and "all" in [s.lower() for s in sections]: sections = None

    print(f"\nScraping Liquipedia data from: {args.url}")
    if sections: print(f"Sections: {', '.join(sections)}")
    else: print(f"Sections: all")
    print()
    df = scrape_playoffs(args.url, sections=sections)
    print(df.head())

    matches = list_matches(df)
    if matches.empty: return

    if args.match:
        row = preselect_match(matches, args.match)
        if row is None: row = choose_match_interactive(matches)
    else:
        row = choose_match_interactive(matches)

    if row is None:
        print("Exited.")
        return

    if args.mode == "h2h":
        run_h2h(row, bc)
    elif args.mode == "chat":
        id_map = load_player_id_map()
        run_chat(row, bc, id_map)
    else:
        run_features(row, bc)


if __name__ == "__main__":
    main()
