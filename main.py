# main.py
import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from scrapers import (
    scrape_playoffs,
    Ballchasing,
    getH2HStats,
    load_player_id_map,
    resolve_ids,
)
from stats import buildFeatRows

load_dotenv()  # BALLCHASING_API_KEY from .env


def list_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Return only concrete matchups and print a numbered list."""
    mask = df["team1"].notna() & df["team2"].notna()
    matches = df[mask].reset_index(drop=True).copy()
    if matches.empty:
        print("No concrete matchups yet.")
        return matches

    print("\nAvailable matchups:")
    for i, r in matches.iterrows():
        sec = r.get("section") or ""
        rnd = r.get("round") or ""
        print(f"[{i}] {r['team1']}  vs  {r['team2']}   | {sec} {rnd}".rstrip())
    print("")
    return matches


def choose_match_interactive(matches: pd.DataFrame) -> pd.Series | None:
    """Prompt the user to pick a matchup index."""
    while True:
        sel = input("Enter a match number (or 'q' to quit): ").strip().lower()
        if sel in {"q", "quit", "exit"}:
            return None
        if sel.isdigit():
            i = int(sel)
            if 0 <= i < len(matches):
                return matches.iloc[i]
        print(f"Invalid selection. Choose 0–{len(matches)-1}, or 'q' to quit.")


def preselect_match(matches: pd.DataFrame, match_arg: str) -> pd.Series | None:
    """Select by index or substring (team name fragment)."""
    s = match_arg.strip()
    # index?
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(matches):
            return matches.iloc[i]
        print(f"--match index {i} out of range (0..{len(matches)-1}).")
        return None

    # substring search (case-insensitive in team1/team2)
    s_low = s.lower()
    mask = matches.apply(
        lambda r: s_low in str(r["team1"]).lower() or s_low in str(r["team2"]).lower(),
        axis=1,
    )
    found = matches[mask]
    if found.empty:
        print(f"--match '{match_arg}' did not match any team names.")
        return None
    if len(found) > 1:
        print(f"--match '{match_arg}' matched multiple rows; picking the first.")
    return found.iloc[0]


def run_h2h(row: pd.Series, bc: Ballchasing):
    t1, t2 = row["team1"], row["team2"]
    r1, r2 = row["team1_players"], row["team2_players"]

    print(f"\nH2H comparison: {t1} vs {t2}\n")
    stats_raw, logs = getH2HStats(t1, t2, r1, r2, bc)
    
    from scrapers.h2h_ballchasing import aggregatePlayers
    # Handle the fact that getH2HStats now returns a raw dataframe
    stats = aggregatePlayers(stats_raw.to_dict('records') if not stats_raw.empty else [])
    
    print(stats if not stats.empty else "No stats found.")
    if logs:
        print("\nLogs:")
        for l in logs[:10]:
            print("-", l)


def run_features(row: pd.Series, bc: Ballchasing):
    """Build team-level features for just the chosen matchup (both sides)."""
    idMap = load_player_id_map()
    logs = []
    r1, r2 = buildFeatRows(bc, row, resolve_ids, idMap, logs)
    out = pd.DataFrame([r1, r2])
    print(out)
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/features_playoffs_selected.csv", index=False)
    print("\nSaved to data/features_playoffs_selected.csv\n")
    if logs:
        print("Logs:")
        for l in logs[:12]:
            print("-", l)


def main():
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
    bc = Ballchasing()

    # Resolve sections
    sections = args.section
    if sections and "all" in [s.lower() for s in sections]:
        sections = None  # None = scrape everything

    print(f"\nScraping Liquipedia data from: {args.url}")
    if sections:
        print(f"Sections: {', '.join(sections)}")
    else:
        print(f"Sections: all")
    print()
    df = scrape_playoffs(args.url, sections=sections)
    print(df.head())

    matches = list_matches(df)
    if matches.empty:
        return

    # Preselect if provided; else, prompt.
    if args.match:
        row = preselect_match(matches, args.match)
        if row is None:
            row = choose_match_interactive(matches)
    else:
        row = choose_match_interactive(matches)

    if row is None:
        print("Exited.")
        return

    if args.mode == "h2h":
        run_h2h(row, bc)
    elif args.mode == "chat":
        from chat import run_chat
        idMap = load_player_id_map()
        run_chat(row, bc, idMap)
    else:
        run_features(row, bc)


if __name__ == "__main__":
    main()
