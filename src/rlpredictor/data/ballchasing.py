"""Ballchasing.com enrichment: per-player series stats from the group linked on Liquipedia.

Each Liquipedia match carries `|ballchasing=<group id>`. One `GET /api/groups/{id}` call
returns cumulative per-player stats for the whole series. Calls are budgeted because
regular Ballchasing accounts are limited to ~500 calls/hour; re-running continues where
the last run stopped.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable

from rlpredictor import config
from rlpredictor.data.http import HttpError, ThrottledClient
from rlpredictor.data.identity import match_player, remember

log = logging.getLogger(__name__)


class Ballchasing:
    def __init__(self, conn: sqlite3.Connection | None, api_key: str | None = None):
        key = api_key or config.ballchasing_api_key()
        if not key:
            raise config.ConfigError(
                "BALLCHASING_API_KEY is not set. Get a token at https://ballchasing.com/upload "
                "and add it to .env.")
        self.http = ThrottledClient(conn, headers={"Authorization": key},
                                    min_interval=config.BALLCHASING_MIN_INTERVAL)

    def ping(self) -> dict:
        return self.http.get_json(f"{config.BALLCHASING_API}/", cache=False, use_cache=False)

    def group(self, group_id: str) -> dict:
        # Finished-series groups are immutable in practice; cache for 30 days.
        return self.http.get_json(f"{config.BALLCHASING_API}/groups/{group_id}", ttl=30 * 86400)


@dataclass
class EnrichReport:
    attempted: int = 0
    ok: int = 0
    unmatched_players: int = 0
    missing: int = 0
    errors: int = 0


def _core(p: dict) -> tuple[int, dict]:
    cum = p.get("cumulative") or {}
    core = cum.get("core") or {}
    demo = cum.get("demo") or {}
    games = int(cum.get("games") or 0)
    return games, {
        "goals": core.get("goals"), "assists": core.get("assists"), "saves": core.get("saves"),
        "shots": core.get("shots"), "score": core.get("score"), "demos": demo.get("inflicted"),
    }


def store_group(conn: sqlite3.Connection, series_id: str, group: dict,
                rosters: dict[int, dict[str, str]]) -> tuple[int, int]:
    """Write player_series_stats for one series. rosters: {team_id: {player_id: name}}.
    Returns (matched, unmatched)."""
    candidates = {pid: name for r in rosters.values() for pid, name in r.items()}
    team_of = {pid: tid for tid, r in rosters.items() for pid in r}
    matched = unmatched = 0
    conn.execute("DELETE FROM player_series_stats WHERE series_id = ?", (series_id,))
    for p in group.get("players") or []:
        platform_id = f"{p['platform']}:{p['id']}" if p.get("platform") and p.get("id") else None
        pid = match_player(conn, platform_id, p.get("name", ""), candidates)
        if pid is None:
            unmatched += 1
            continue
        remember(conn, platform_id, pid, p.get("name", ""))
        games, st = _core(p)
        conn.execute(
            """INSERT OR REPLACE INTO player_series_stats
               (series_id, player_id, team_id, games, goals, assists, saves, shots, score, demos)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (series_id, pid, team_of[pid], games, st["goals"], st["assists"], st["saves"],
             st["shots"], st["score"], st["demos"]))
        matched += 1
    return matched, unmatched


def _rosters_for(conn, event_root: str, team_ids: tuple[int, int]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for tid in team_ids:
        rows = conn.execute(
            """SELECT er.player_id, p.name FROM event_rosters er JOIN players p ON p.id = er.player_id
               WHERE er.event_root = ? AND er.team_id = ?""", (event_root, tid)).fetchall()
        if rows:
            out[tid] = {r[0]: r[1] for r in rows}
    return out


def enrich(conn: sqlite3.Connection, bc: Ballchasing, budget: int = 400,
           team_ids: list[int] | None = None,
           progress: Callable[[str], None] = lambda s: None) -> EnrichReport:
    """Fetch Ballchasing groups for finished series that have both rosters listed.
    Newest first; series involving `team_ids` first."""
    prio = ""
    params: list = []
    if team_ids:
        marks = ",".join("?" * len(team_ids))
        prio = f"(s.team1_id IN ({marks}) OR s.team2_id IN ({marks})) DESC, "
        params = list(team_ids) * 2
    rows = conn.execute(
        f"""SELECT s.id, s.event_root, s.team1_id, s.team2_id, s.bc_group FROM series s
            LEFT JOIN bc_enrichment e ON e.series_id = s.id
            WHERE s.finished = 1 AND s.bc_group IS NOT NULL AND e.series_id IS NULL
              AND EXISTS (SELECT 1 FROM event_rosters r WHERE r.event_root = s.event_root AND r.team_id = s.team1_id)
              AND EXISTS (SELECT 1 FROM event_rosters r WHERE r.event_root = s.event_root AND r.team_id = s.team2_id)
            ORDER BY {prio} s.date DESC LIMIT ?""", params + [budget]).fetchall()

    rep = EnrichReport()
    for i, r in enumerate(rows, 1):
        rep.attempted += 1
        status, detail = "ok", None
        try:
            g = bc.group(r["bc_group"])
            m, u = store_group(conn, r["id"], g, _rosters_for(conn, r["event_root"],
                                                              (r["team1_id"], r["team2_id"])))
            rep.unmatched_players += u
            if m == 0:
                status, detail = "unmatched", f"{u} players unmatched"
            else:
                rep.ok += 1
        except HttpError as e:
            if e.status == 404:
                status, detail = "missing", "group not found"
                rep.missing += 1
            else:
                status, detail = "error", str(e)[:200]
                rep.errors += 1
                log.warning("ballchasing %s: %s", r["bc_group"], e)
                if e.status in (401, 403):
                    raise
        conn.execute("INSERT OR REPLACE INTO bc_enrichment (series_id, status, detail, fetched_at) "
                     "VALUES (?,?,?,?)", (r["id"], status, detail, time.time()))
        if i % 20 == 0:
            conn.commit()
            progress(f"ballchasing {i}/{len(rows)} (ok {rep.ok}, missing {rep.missing})")
    conn.commit()
    return rep
