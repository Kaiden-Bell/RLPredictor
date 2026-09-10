# stats_pipeline.py (patched)

import sys
import time
import pandas as pd
from datetime import datetime, timedelta, timezone


def _progress(msg, current, total, start_time, cached=0):
    """Print an inline progress indicator with ETA."""
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


def _progress_done(msg, total, start_time, cached=0):
    """Finish a progress line."""
    elapsed = time.time() - start_time
    cache_str = f" ({cached} from cache)" if cached else ""
    sys.stdout.write(f"\r  {msg}: {total} replays in {elapsed:.1f}s{cache_str}                    \n")
    sys.stdout.flush()

RECENT_DAYS = 90
MAX_REPLAYS = 150
AGG_KEYS = ["Goals", "Shots", "Saves", "Demos"]

def _iso(dt_ms_or_iso):
    if isinstance(dt_ms_or_iso, (int, float)):
        return datetime.fromtimestamp(dt_ms_or_iso/1000, tz=timezone.utc).isoformat()
    return str(dt_ms_or_iso)

def _in_window(dateStr, days=RECENT_DAYS):
    try:
        dt = datetime.fromisoformat(dateStr.replace("Z", "+00:00"))
    except Exception:
        return True
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)

def pullReplays(bc, playerID, count=MAX_REPLAYS, playlist="private"):
    params = {
        "player-id": playerID,
        "sort-by": "replay-date",
        "sort-dir": "desc",
        "count": min(200, int(count)),
    }
    if playlist:
        params["playlist"] = playlist
        
    data = bc.listReplays(**params)
    return data.get("list", []) or []

MOMENTUM_DAYS = 14  # only count recent ranked 2s for momentum

def rankedActivity(bc, playerIDs, logs):
    """
    Returns a dict mapping PlayerID -> {games, avg_score, win_rate}.
    Provides a richer "momentum" input vector for the Neural Network
    by only considering ranked-doubles games from the last MOMENTUM_DAYS.
    """
    activity = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=MOMENTUM_DAYS)
    unique_pids = list(set(playerIDs))

    for p_idx, pid in enumerate(unique_pids):
        short_id = pid.split(":")[-1][:12] if ":" in pid else pid[:12]
        print(f"  Ranked 2s momentum: player {p_idx + 1}/{len(unique_pids)} ({short_id})")
        info = {"games": 0, "avg_score": 0.0, "win_rate": 0.0}
        try:
            params = {
                "player-id": pid,
                "playlist": "ranked-doubles",
                "sort-by": "replay-date",
                "sort-dir": "desc",
                "count": 50,
            }
            data = bc.listReplays(**params)
            reps = data.get("list", []) or []

            scores, wins, total = [], 0, 0
            t0 = time.time()
            valid_reps = []
            for rep in reps:
                date_str = rep.get("date") or rep.get("created")
                if date_str and not _in_window(str(date_str), days=MOMENTUM_DAYS):
                    continue
                rid = rep.get("id")
                if rid:
                    valid_reps.append(rid)

            for r_idx, rid in enumerate(valid_reps):
                _progress("Ranked replays", r_idx + 1, len(valid_reps), t0)
                try:
                    detail = bc.getReplay(rid)
                except Exception:
                    continue

                # Find this player's stats in the replay
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
                            if team_goals > opp_goals:
                                wins += 1
                            total += 1

            if valid_reps:
                _progress_done("Ranked replays", len(valid_reps), t0)

            if total > 0:
                info["games"] = total
                info["avg_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
                info["win_rate"] = round((wins / total) * 100, 1)

            time.sleep(bc.delay)
        except Exception as e:
            logs.append(f"Ranked 2s fetch failed for {pid}: {e}")

        activity[pid] = info

    return activity

def replayStats(bc, playerIDs, logs):
    players = []
    unique_pids = list(set(playerIDs))
    for p_idx, pid in enumerate(unique_pids):
        short_id = pid.split(":")[-1][:12] if ":" in pid else pid[:12]
        print(f"  Listing replays: player {p_idx + 1}/{len(unique_pids)} ({short_id})")
        try:
            players.extend(pullReplays(bc, pid))
            time.sleep(0.12)
        except Exception as e:
            logs.append(f"List replays failed for {pid}: {e}")

    # Deduplicate
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
            detail = bc.getReplay(rid)
        except Exception as e:
            logs.append(f"getReplay {rid} failed: {e}")
            _progress("Generic stats", idx + 1, total_replays, t0, cached_count)
            continue
        # If the call returned in <10ms, it was a cache hit
        if time.time() - call_start < 0.01:
            cached_count += 1

        _progress("Generic stats", idx + 1, total_replays, t0, cached_count)

        date_s = _iso(detail.get("date"))
        if not _in_window(date_s):
            continue

        for side in ("blue", "orange"):
            team = (detail.get(side) or {})
            for pl in team.get("players", []) or []:
                name  = pl.get("name") or (pl.get("player") or {}).get("name")
                stats = (pl.get("stats") or {})
                core  = stats.get("core") or {}
                demo  = stats.get("demo") or {}
                rows.append({
                    "Player": name,
                    "Goals": core.get("goals", 0),
                    "Shots": core.get("shots", 0),
                    "Saves": core.get("saves", 0),
                    "Demos": demo.get("inflicted", 0),
                    "replay_id": detail.get("id"), 
                    "Date": date_s,
                })

    _progress_done("Generic stats", total_replays, t0, cached_count)
    return pd.DataFrame(rows)

def teamFeats(bc, rosterIDs, logs):
    if not rosterIDs:
        return pd.Series({k: 0 for k in AGG_KEYS + ["Shot %", "Games"]})
    dfp = replayStats(bc, rosterIDs, logs)
    if dfp.empty:
        return pd.Series({k: 0 for k in AGG_KEYS + ["Shot %", "Games"]})

    perPlayer = dfp.groupby("Player", dropna=False).agg(
        Games=("replay_id", "nunique"),
        Goals=("Goals","sum"),
        Shots=("Shots","sum"),
        Saves=("Saves","sum"),
        Demos=("Demos","sum"),
    ).reset_index()

    totals = perPlayer[["Goals","Shots","Saves","Demos"]].sum()
    games = perPlayer["Games"].sum()
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

def buildFeatRows(bc, matchups, resolve, idMap, logs):
    t1, t2 = matchups["team1"], matchups["team2"]
    r1, r2 = matchups["team1_players"], matchups["team2_players"] 
    ids1 = resolve(r1, idMap)
    ids2 = resolve(r2, idMap)

    f1 = teamFeats(bc, ids1, logs)
    f2 = teamFeats(bc, ids2, logs)

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
