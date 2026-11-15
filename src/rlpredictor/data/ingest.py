"""Ingestion: Liquipedia pages -> series/games/rosters in SQLite.

Incremental: page revision ids are stored, and only pages whose revision changed are
re-fetched and re-parsed. Team names (template short names like "nip", old names like
"rstv") are resolved to canonical team pages via Liquipedia's own team templates.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable

from rlpredictor import config
from rlpredictor.data import db
from rlpredictor.data.liquipedia import Liquipedia
from rlpredictor.data.wikitext import (
    PageInfo,
    RosterEntry,
    SeriesResult,
    SquadMember,
    parse_matches,
    parse_page_info,
    parse_rosters,
    parse_squad,
    player_key,
    region_from_title,
    team_key,
)

log = logging.getLogger(__name__)

EXCLUDE = re.compile(
    r"/(1v1|2v2|Rankings|Statistics|Broadcast Talent|Media|Showmatch|[^/]*Rivalry[^/]*)(/|$)", re.I)
NON_COMPETITIVE = re.compile(r"show\s*match|all[- ]?star|rivalry|exhibition", re.I)


@dataclass
class IngestReport:
    pages_listed: int = 0
    pages_fetched: int = 0
    series_upserted: int = 0
    rosters: int = 0
    unresolved: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class ParsedPage:
    title: str
    revid: int | None
    info: PageInfo
    rosters: list[RosterEntry]
    matches: list[SeriesResult]


def season_prefixes(seasons: list[str]) -> list[str]:
    return [p.format(season=s) for s in seasons for p in config.DEFAULT_SEASON_PREFIXES]


def parse_page(title: str, revid: int | None, text: str) -> ParsedPage:
    info = parse_page_info(title, text)
    return ParsedPage(title, revid, info, parse_rosters(text),
                      parse_matches(text, info.page_date or info.start_date))


# ---- team resolution -------------------------------------------------------
def team_names(parsed: list[ParsedPage]) -> set[str]:
    names: set[str] = set()
    for p in parsed:
        for r in p.rosters:
            names.add(r.team_link)
        for m in p.matches:
            names.update((m.team1, m.team2))
    return names


def resolve_teams(conn: sqlite3.Connection, lp: Liquipedia | None, names: set[str],
                  progress: Callable[[str], None] = lambda s: None) -> None:
    """Fill team_resolution for names not seen before."""
    known = {r[0] for r in conn.execute("SELECT raw_key FROM team_resolution")}
    todo: dict[str, str] = {}
    for n in names:
        k = team_key(n)
        if k not in known and k not in todo:
            todo[k] = n
    if not todo:
        return
    if lp is None:
        for k in todo:
            conn.execute("INSERT OR IGNORE INTO team_resolution (raw_key, page) VALUES (?, NULL)", (k,))
        return
    progress(f"resolving {len(todo)} new team names via Liquipedia team templates "
             f"(~{(len(todo) // 250 + 1) * 30}s)")
    resolved = lp.resolve_team_pages(list(todo.values()))
    for k, raw in todo.items():
        conn.execute("INSERT OR REPLACE INTO team_resolution (raw_key, page) VALUES (?, ?)",
                     (k, resolved.get(raw)))
    conn.commit()


def canonical_team(conn: sqlite3.Connection, raw: str, region: str | None = None) -> int:
    """Team id for a raw template/link name, creating the team if needed."""
    rk = team_key(raw)
    row = conn.execute("SELECT page FROM team_resolution WHERE raw_key = ?", (rk,)).fetchone()
    page = row[0] if row and row[0] else raw
    key = team_key(page)
    tid = db.get_or_create_team(conn, key, page, region)
    if rk != key:
        conn.execute("INSERT OR IGNORE INTO team_aliases (alias, team_id) VALUES (?, ?)", (rk, tid))
    if region and region != "International":
        conn.execute("UPDATE teams SET region = ? WHERE id = ?", (region, tid))
    return tid


# ---- writing ---------------------------------------------------------------
def _event_root(conn: sqlite3.Connection, title: str) -> str:
    parts = title.split("/")
    for i in range(len(parts), 0, -1):
        cand = "/".join(parts[:i])
        if cand != title and config.STANDALONE_ROOTS.match(cand):
            break
        row = conn.execute("SELECT has_infobox FROM pages WHERE title = ?", (cand,)).fetchone()
        if row and row[0]:
            return cand
    return title


def _upsert_page(conn, title: str, revid: int | None, info: PageInfo) -> None:
    sd = info.start_date or info.page_date
    conn.execute(
        """INSERT INTO pages (title, has_infobox, revid, name, tier, region, start_date, end_date, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(title) DO UPDATE SET has_infobox=excluded.has_infobox, revid=excluded.revid,
             name=excluded.name, tier=excluded.tier, region=excluded.region,
             start_date=excluded.start_date, end_date=excluded.end_date, fetched_at=excluded.fetched_at""",
        (title, int(info.has_infobox), revid, info.name, info.tier, region_from_title(title),
         sd and str(sd), info.end_date and str(info.end_date), time.time()),
    )


def _player(conn, raw: str) -> str:
    pid = player_key(raw)
    conn.execute("INSERT OR IGNORE INTO players (id, name) VALUES (?, ?)", (pid, raw))
    return pid


def write_page(conn: sqlite3.Connection, p: ParsedPage, report: IngestReport) -> None:
    title = p.title
    _upsert_page(conn, title, p.revid, p.info)
    root = _event_root(conn, title)
    conn.execute("UPDATE pages SET event_root = ? WHERE title = ?", (root, title))
    root_row = conn.execute("SELECT tier, start_date FROM pages WHERE title = ?", (root,)).fetchone()
    tier = p.info.tier or (root_row["tier"] if root_row else None)
    region = region_from_title(title)

    if p.rosters:
        conn.execute("DELETE FROM event_rosters WHERE event_root = ?", (root,))
        for r in p.rosters:
            tid = canonical_team(conn, r.team_link, region)
            for slot, pname in enumerate(r.players + r.subs, start=1):
                conn.execute(
                    "INSERT OR IGNORE INTO event_rosters (event_root, team_id, player_id, slot) "
                    "VALUES (?,?,?,?)", (root, tid, _player(conn, pname), slot))
            report.rosters += 1

    roster_team_ids = {r[0] for r in conn.execute(
        "SELECT team_id FROM event_rosters WHERE event_root = ?", (root,))}

    fallback = None
    if root_row and root_row["start_date"]:
        fallback = date.fromisoformat(root_row["start_date"])

    seen: set[str] = set()
    for m in p.matches:
        if m.section and NON_COMPETITIVE.search(m.section):
            continue
        t1, t2 = canonical_team(conn, m.team1), canonical_team(conn, m.team2)
        if t1 == t2:
            continue
        for raw, tid in ((m.team1, t1), (m.team2, t2)):
            if roster_team_ids and tid not in roster_team_ids:
                report.unresolved.add((title, raw))
                conn.execute("INSERT OR IGNORE INTO unresolved_teams (page, raw_name) VALUES (?,?)",
                             (title, raw))
        sid = f"{title}#{m.match_key}"
        seen.add(sid)
        d = m.date or fallback
        conn.execute(
            """INSERT INTO series (id, page, event_root, date, tier, region, section, best_of,
                 team1_id, team2_id, score1, score2, finished, walkover, bc_group)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET event_root=excluded.event_root, date=excluded.date,
                 tier=excluded.tier, region=excluded.region, section=excluded.section,
                 best_of=excluded.best_of, team1_id=excluded.team1_id, team2_id=excluded.team2_id,
                 score1=excluded.score1, score2=excluded.score2, finished=excluded.finished,
                 walkover=excluded.walkover, bc_group=excluded.bc_group""",
            (sid, title, root, d and str(d), tier, region, m.section, m.best_of,
             t1, t2, m.score1, m.score2, int(m.finished), int(m.walkover), m.ballchasing),
        )
        conn.execute("DELETE FROM games WHERE series_id = ?", (sid,))
        for g in m.games:
            conn.execute(
                "INSERT INTO games (series_id, n, score1, score2, winner, ot) VALUES (?,?,?,?,?,?)",
                (sid, g.n, g.score1, g.score2, g.winner, int(g.ot)))
        report.series_upserted += 1

    for (sid,) in conn.execute("SELECT id FROM series WHERE page = ?", (title,)).fetchall():
        if sid not in seen:
            conn.execute("DELETE FROM series WHERE id = ?", (sid,))


def store_pages(conn: sqlite3.Connection, parsed: list[ParsedPage], lp: Liquipedia | None,
                report: IngestReport, progress: Callable[[str], None] = lambda s: None) -> None:
    parsed = sorted(parsed, key=lambda p: (p.title.count("/"), p.title))
    # Page infos first so event roots (infobox pages) are known for every child.
    for p in parsed:
        _upsert_page(conn, p.title, None, p.info)
    resolve_teams(conn, lp, team_names(parsed), progress)
    conn.executemany("DELETE FROM unresolved_teams WHERE page = ?", [(p.title,) for p in parsed])
    for i, p in enumerate(parsed, 1):
        write_page(conn, p, report)
        report.pages_fetched += 1
        if i % 50 == 0:
            conn.commit()
            progress(f"stored {i}/{len(parsed)} pages")
    conn.commit()


def ingest(conn: sqlite3.Connection, lp: Liquipedia, seasons: list[str], force: bool = False,
           progress: Callable[[str], None] = lambda s: None) -> IngestReport:
    report = IngestReport()
    titles: list[str] = []
    for prefix in season_prefixes(seasons):
        listed = [t for t in lp.list_pages(prefix) if not EXCLUDE.search(t)]
        progress(f"{prefix}: {len(listed)} pages")
        titles += listed
    report.pages_listed = len(titles)

    if force:
        changed = titles
    else:
        stored = {r["title"]: r["revid"] for r in conn.execute("SELECT title, revid FROM pages")}
        revs = lp.revisions(titles)
        changed = [t for t in titles if revs.get(t) != stored.get(t)]
    progress(f"{len(changed)} of {len(titles)} pages changed since last ingest")
    if not changed:
        return report

    parsed: list[ParsedPage] = []
    for i, page in enumerate(lp.fetch(changed), 1):
        if page.text:
            parsed.append(parse_page(page.title, page.revid, page.text))
        if i % 50 == 0:
            progress(f"fetched+parsed {i}/{len(changed)} pages")
    store_pages(conn, parsed, lp, report, progress)
    return report


# ---- current squads --------------------------------------------------------
@dataclass
class SquadReport:
    teams_checked: int = 0
    teams_fetched: int = 0
    with_squad: int = 0
    changed: list[str] = field(default_factory=list)


def _id_shape(pid: str) -> tuple[str, str]:
    """(alphanumeric core, disambiguator) — "atow." -> ("atow", ""),
    "diaz (brazilian player)" -> ("diaz", "brazilian player")."""
    m = re.match(r"^(.*?)\s*\(([^)]*)\)?\s*$", pid)
    core, dis = (m.group(1), m.group(2) or "") if m else (pid, "")
    return re.sub(r"[^\w]", "", core), dis.strip()


def resolve_squad_player(conn: sqlite3.Connection, team_id: int, link: str, name: str) -> str:
    """Player id for a squad member, reusing the id that already carries their history.

    Squad links are Liquipedia page names ("Atow"), while event TeamCards often store the
    name as written ("Atow."), so the same player can exist under several ids. Among ids
    with the same alphanumeric core (and the same disambiguator, if the link has one),
    prefer one that played for this team, then the one with the most event appearances."""
    key = player_key(link)
    core, dis = _id_shape(key)
    if not core:
        return key
    cands = []
    for (pid,) in conn.execute("SELECT id FROM players WHERE id LIKE ?", (f"%{core[:3]}%",)):
        c2, d2 = _id_shape(pid)
        if c2 == core and (d2 == dis or (not dis and not d2)):
            cands.append(pid)
    if not cands:
        return key

    def rank(pid):
        here = conn.execute("SELECT COUNT(*) FROM event_rosters WHERE player_id = ? AND team_id = ?",
                            (pid, team_id)).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM event_rosters WHERE player_id = ?",
                             (pid,)).fetchone()[0]
        return (here > 0, total, pid == key)
    return max(cands, key=rank)


def store_squad(conn: sqlite3.Connection, team_id: int, members: list[SquadMember]) -> bool:
    """Replace a team's active squad. Starters get slots 1..3, subs 4+. Returns True if the
    set of players changed."""
    before = {r[0] for r in conn.execute("SELECT player_id FROM team_squads WHERE team_id = ?",
                                         (team_id,))}
    conn.execute("DELETE FROM team_squads WHERE team_id = ?", (team_id,))
    ordered = [m for m in members if not m.is_sub] + [m for m in members if m.is_sub]
    after = set()
    for slot, m in enumerate(ordered, start=1):
        pid = resolve_squad_player(conn, team_id, m.link, m.name)
        conn.execute("INSERT OR IGNORE INTO players (id, name) VALUES (?, ?)", (pid, m.name))
        conn.execute(
            "INSERT OR IGNORE INTO team_squads (team_id, player_id, slot, role, join_date) "
            "VALUES (?,?,?,?,?)", (team_id, pid, slot, m.role or None,
                                   m.join_date and str(m.join_date)))
        after.add(pid)
    conn.execute("INSERT OR REPLACE INTO team_squad_fetch (team_id, fetched_at, n_players) "
                 "VALUES (?,?,?)", (team_id, time.time(), len(ordered)))
    return before != after


def ingest_squads(conn: sqlite3.Connection, lp: Liquipedia, team_ids: list[int] | None = None,
                  days: int = 180, force: bool = False, max_age: float = 12 * 3600,
                  progress: Callable[[str], None] = lambda s: None) -> SquadReport:
    """Refresh active squads for `team_ids`, or for every team that played in the last
    `days` days. Teams checked within `max_age` seconds are skipped unless `force`."""
    if team_ids:
        marks = ",".join("?" * len(team_ids))
        rows = conn.execute(f"SELECT id, name FROM teams WHERE id IN ({marks})", team_ids).fetchall()
    else:
        since = str(date.today() - timedelta(days=days))
        rows = conn.execute(
            """SELECT DISTINCT t.id, t.name FROM teams t
               JOIN series s ON (s.team1_id = t.id OR s.team2_id = t.id)
               WHERE s.date >= ?""", (since,)).fetchall()
    rep = SquadReport(teams_checked=len(rows))
    if not force:
        fresh = {r[0] for r in conn.execute("SELECT team_id FROM team_squad_fetch WHERE fetched_at > ?",
                                            (time.time() - max_age,))}
        rows = [r for r in rows if r[0] not in fresh]
    if not rows:
        return rep
    progress(f"fetching active squads for {len(rows)} teams "
             f"(~{(len(rows) // 40 + 1) * 30}s, Liquipedia expandtemplates throttle)")
    expanded = lp.active_squads([r[1] for r in rows], cache=not force)
    for tid, name in rows:
        text = expanded.get(name)
        if text is None or 'class="error"' in text or "Lua error" in text:
            continue  # keep the stored squad rather than wiping it on a failed expansion
        rep.teams_fetched += 1
        members = parse_squad(text)
        if members:
            rep.with_squad += 1
        if store_squad(conn, tid, members):
            rep.changed.append(name)
    conn.commit()
    return rep
