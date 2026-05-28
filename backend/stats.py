"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Stats compilation pipeline fetching recent scrims and ranked activity records.
- Usage: Imported by main.py and chat.py to calculate team stats and momentum features.
"""

import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from utils.database import get_connection
from utils.player_identity import auto_detect_aliases


RECENT_DAYS = 90
MAX_REPLAYS = 150
AGG_KEYS = ["Goals", "Shots", "Saves", "Demos"]
MOMENTUM_DAYS = 14


def progress(msg, current, total, start_time, cached=0):
    """
    Description:
        Prints an inline command line progress bar with ETA calculations.
    Arguments:
        msg: Display message.
        current: Current index.
        total: Total items.
        start_time: Process start epoch time float.
        cached: Count of items loaded from cache.
    Returns:
        None
    """
    elapsed = time.time() - start_time
    if current > 0:
        per_item = elapsed / current
        remaining = per_item * (total - current)
        eta_str = f"ETA ~{remaining:.0f}s" if remaining > 1 else "almost done"
    else:
        eta_str = "calculating..."
    cache_str = f" | {cached} cached" if cached else ""
    sys.stdout.write(f"\r  {msg}: [{current}/{total}] {elapsed:.0f}s elapsed, {eta_str}{cache_str}    ")
    sys.stdout.flush()


def progress_done(msg, total, start_time, cached=0):
    """
    Description:
        Completes the command line progress bar with total stats.
    Arguments:
        msg: Display message.
        total: Total items.
        start_time: Process start epoch time float.
        cached: Count of items loaded from cache.
    Returns:
        None
    """
    elapsed = time.time() - start_time
    cache_str = f" ({cached} from cache)" if cached else ""
    sys.stdout.write(f"\r{msg}: {total} replays in {elapsed:.1f}s{cache_str}\n")
    sys.stdout.flush()


def iso_format(dt_ms_or_iso):
    """
    Description:
        Converts millisecond epoch floats or generic inputs to standard ISO strings.
    Arguments:
        dt_ms_or_iso: Input value.
    Returns:
        String: ISO-formatted timestamp string.
    """
    if isinstance(dt_ms_or_iso, (int, float)):
        return datetime.fromtimestamp(dt_ms_or_iso/1000, tz=timezone.utc).isoformat()
    return str(dt_ms_or_iso)


def get_canonical_name(pid):
    """
    Description:
        Resolves a canonical player display name from their platform ID.
    Arguments:
        pid: Platform ID string.
    Returns:
        String: Canonical player display name.
    """
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT cp.canonical_name FROM player_ids pid "
            "JOIN canonical_players cp ON pid.canonical_player_id = cp.canonical_player_id "
            "WHERE pid.platform_id = ?", (pid,)
        ).fetchone()
        conn.close()
        if row: return row["canonical_name"]
    except Exception:
        pass
    return pid.split(":")[-1][:12] if ":" in pid else pid[:12]


def in_window(date_str, days=RECENT_DAYS):
    """
    Description:
        Checks if a given date string falls inside the active query window.
    Arguments:
        date_str: Date string.
        days: Limit window size in days.
    Returns:
        Boolean: True if date falls within window, False otherwise.
    """
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return True
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)


def pull_replays(bc, player_id, count=MAX_REPLAYS, playlist="private", pro_only=True):
    """
    Description:
        Pulls recent list of replays from Ballchasing API.
    Arguments:
        bc: Ballchasing client.
        player_id: Platform ID to pull for.
        count: Max replay count.
        playlist: Target playlist category (scrims / public, etc.).
        pro_only: Boolean flag to restrict results to lobbies with at least one pro player.
    Returns:
        List of replay dictionaries.
    """
    params = {
        "player-id": player_id,
        "sort-by": "replay-date",
        "sort-dir": "desc",
        "count": min(200, int(count)),
    }
    if playlist: params["playlist"] = playlist
    if pro_only: params["pro"] = "true"
        
    data = bc.list_replays(**params)
    return data.get("list", []) or []


def ranked_activity(bc, player_ids, logs):
    """
    Description:
        Queries Ballchasing API for active player ranked doubles (2v2) momentum statistics.
    Arguments:
        bc: Ballchasing client.
        player_ids: List of platform player IDs.
        logs: Diagnostic logs list.
    Returns:
        Dictionary mapping player platform ID to ranked activity metrics.
    """
    activity = {}
    unique_pids = list(set(player_ids))

    for p_idx, pid in enumerate(unique_pids):
        cname = get_canonical_name(pid)
        print(f"  Ranked 2s momentum: player {p_idx + 1}/{len(unique_pids)} ({cname})")
        info = {"games": 0, "avg_score": 0.0, "win_rate": 0.0}
        try:
            params = {
                "player-id": pid,
                "playlist": "ranked-doubles",
                "sort-by": "replay-date",
                "sort-dir": "desc",
                "count": 50,
                "pro": "true",
            }
            data = bc.list_replays(**params)
            reps = data.get("list", []) or []

            scores, wins, total = [], 0, 0
            t0 = time.time()
            valid_reps = []
            for rep in reps:
                date_str = rep.get("date") or rep.get("created")
                if date_str and not in_window(str(date_str), days=MOMENTUM_DAYS): continue
                rid = rep.get("id")
                if rid: valid_reps.append(rid)

            for r_idx, rid in enumerate(valid_reps):
                progress("Ranked replays", r_idx + 1, len(valid_reps), t0)
                try:
                    detail = bc.get_replay(rid)
                except Exception:
                    continue

                for side in ("blue", "orange"):
                    team = detail.get(side) or {}
                    for pl in team.get("players", []) or []:
                        pl_id = (pl.get("id") or {}).get("id") or ""
                        pl_platform_id = f"{(pl.get('id') or {}).get('platform', '')}:{pl_id}"
                        if pid.lower() in (pl_id.lower(), pl_platform_id.lower()):
                            core = (pl.get("stats") or {}).get("core") or {}
                            scores.append(core.get("score", 0))
                            team_goals = (team.get("stats") or {}).get("core", {}).get("goals", 0)
                            opp_side = "orange" if side == "blue" else "blue"
                            opp_goals = ((detail.get(opp_side) or {}).get("stats") or {}).get("core", {}).get("goals", 0)
                            if team_goals > opp_goals: wins += 1
                            total += 1

            if valid_reps: progress_done("Ranked replays", len(valid_reps), t0)

            if total > 0:
                info["games"] = total
                info["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
                info["win_rate"] = round((wins / total) * 100, 1)

            time.sleep(bc.delay)
        except Exception as e:
            logs.append(f"Ranked 2s fetch failed for {pid}: {e}")

        activity[pid] = info

    return activity


def replay_stats(bc, player_ids, logs):
    """
    Description:
        Retrieves recent generic (private/scrim) replay details for players.
    Arguments:
        bc: Ballchasing client.
        player_ids: List of platform player IDs.
        logs: Diagnostic logs list.
    Returns:
        Pandas DataFrame of stats.
    """
    players = []
    unique_pids = list(set(player_ids))
    for p_idx, pid in enumerate(unique_pids):
        cname = get_canonical_name(pid)
        print(f"  Listing replays: player {p_idx + 1}/{len(unique_pids)} ({cname})")
        try:
            players.extend(pull_replays(bc, pid))
            time.sleep(0.12)
        except Exception as e:
            logs.append(f"List replays failed for {pid}: {e}")

    unique_replays = []
    seen = set()
    for it in players:
        rid = it.get("id")
        if rid and rid not in seen:
            seen.add(rid)
            unique_replays.append(it)

    total_replays = len(unique_replays)
    print(f"  Fetching details for {total_replays} unique replays...")
    t0 = time.time()

    rows = []
    cached_count = 0
    for idx, it in enumerate(unique_replays):
        rid = it.get("id")

        call_start = time.time()
        try:
            detail = bc.get_replay(rid)
        except Exception as e:
            logs.append(f"getReplay {rid} failed: {e}")
            progress("Generic stats", idx + 1, total_replays, t0, cached_count)
            continue
        if time.time() - call_start < 0.01: cached_count += 1

        progress("Generic stats", idx + 1, total_replays, t0, cached_count)

        date_s = iso_format(detail.get("date"))
        if not in_window(date_s): continue

        for side in ("blue", "orange"):
            team = (detail.get(side) or {})
            for pl in team.get("players", []) or []:
                name = pl.get("name") or (pl.get("player") or {}).get("name")
                if not name: continue
                
                plat = (pl.get("id") or {}).get("platform")
                p_id = (pl.get("id") or {}).get("id")
                cid = auto_detect_aliases(name, p_id, plat, date_s)
                
                stats = (pl.get("stats") or {})
                core = stats.get("core") or {}
                demo = stats.get("demo") or {}
                rows.append({
                    "canonical_player_id": cid,
                    "Player": name,
                    "Goals": core.get("goals", 0),
                    "Shots": core.get("shots", 0),
                    "Saves": core.get("saves", 0),
                    "Demos": demo.get("inflicted", 0),
                    "replay_id": detail.get("id"),
                    "Date": date_s,
                })

    progress_done("Generic stats", total_replays, t0, cached_count)
    return pd.DataFrame(rows)


def team_feats(bc, roster_ids, logs):
    """
    Description:
        Aggregates recent generic replay stats for a list of roster platform IDs.
    Arguments:
        bc: Ballchasing client.
        roster_ids: List of resolved player platform IDs.
        logs: Diagnostic logs list.
    Returns:
        Pandas Series containing summed and averaged team features.
    """
    if not roster_ids: return pd.Series({k: 0 for k in AGG_KEYS + ["Shot %", "Games"]})
    dfp = replay_stats(bc, roster_ids, logs)
    if dfp.empty: return pd.Series({k: 0 for k in AGG_KEYS + ["Shot %", "Games"]})

    per_player = dfp.groupby("canonical_player_id", dropna=False).agg(
        Games=("replay_id", "nunique"),
        Goals=("Goals", "sum"),
        Shots=("Shots", "sum"),
        Saves=("Saves", "sum"),
        Demos=("Demos", "sum"),
    ).reset_index()

    totals = per_player[["Goals", "Shots", "Saves", "Demos"]].sum()
    games = per_player["Games"].sum()
    shot_pct = (totals["Goals"]/totals["Shots"]) if totals["Shots"] else 0.0
    out = pd.Series({
        "Games": int(games),
        "Goals": int(totals["Goals"]),
        "Shots": int(totals["Shots"]),
        "Saves": int(totals["Saves"]),
        "Demos": int(totals["Demos"]),
        "Shot %": float(shot_pct),
    })
    return out


def build_feat_rows(bc, matchups, resolve, id_map, logs):
    """
    Description:
        Compiles high-level training and prediction team stats vectors from playoff matchups.
    Arguments:
        bc: Ballchasing client.
        matchups: Match data dict or row series.
        resolve: ID resolver function.
        id_map: Active player platform mappings.
        logs: Diagnostic logs list.
    Returns:
        Tuple of two Pandas Series representing team1 and team2 feature rows.
    """
    t1, t2 = matchups["team1"], matchups["team2"]
    r1, r2 = matchups["team1_players"], matchups["team2_players"]
    ids1 = resolve(r1, id_map)
    ids2 = resolve(r2, id_map)

    f1 = team_feats(bc, ids1, logs)
    f2 = team_feats(bc, ids2, logs)

    left = pd.Series({
        "team": t1,
        "opponent": t2,
        "section": matchups.get("section"),
        "round": matchups.get("round"),
        "best_of": matchups.get("best_of", 7),
        "side": "team1",
    })
    right = left.copy()
    right["team"], right["opponent"], right["side"] = t2, t1, "team2"

    row1 = pd.concat([left, f1])
    row2 = pd.concat([right, f2])
    return row1, row2
