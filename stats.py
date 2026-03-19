# stats_pipeline.py (patched)

import time
import pandas as pd
from datetime import datetime, timedelta, timezone

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

    for pid in set(playerIDs):
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
            for rep in reps:
                # Filter by recency using the replay date
                date_str = rep.get("date") or rep.get("created")
                if date_str and not _in_window(str(date_str), days=MOMENTUM_DAYS):
                    continue

                rid = rep.get("id")
                if not rid:
                    continue

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
                            # Check if this player's side won
                            team_goals = (team.get("stats") or {}).get("core", {}).get("goals", 0)
                            opp_side = "orange" if side == "blue" else "blue"
                            opp_goals = ((detail.get(opp_side) or {}).get("stats") or {}).get("core", {}).get("goals", 0)
                            if team_goals > opp_goals:
                                wins += 1
                            total += 1

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
    for pid in set(playerIDs):
        try:
            players.extend(pullReplays(bc, pid))
            time.sleep(0.12)
        except Exception as e:
            logs.append(f"List replays failed for {pid}: {e}")

    rows = []
    seen = set()
    for it in players:
        rid = it.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        try:
            detail = bc.getReplay(rid)
        except Exception as e:
            logs.append(f"getReplay {rid} failed: {e}")
            continue
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
