"""SQLite storage: schema and small query helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rlpredictor import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    url TEXT,
    body TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    ttl REAL                      -- seconds; NULL = never expires
);

CREATE TABLE IF NOT EXISTS pages (
    title TEXT PRIMARY KEY,
    event_root TEXT,              -- nearest ancestor page with an Infobox league
    has_infobox INTEGER NOT NULL DEFAULT 0,
    revid INTEGER,
    name TEXT,
    tier INTEGER,
    region TEXT,
    start_date TEXT,
    end_date TEXT,
    fetched_at REAL
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,     -- team_key() of the canonical name
    name TEXT NOT NULL,
    region TEXT
);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias TEXT PRIMARY KEY,       -- team_key() of an alternative name
    team_id INTEGER NOT NULL REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS players (
    id TEXT PRIMARY KEY,          -- player_key() of the Liquipedia page name
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_platform_ids (
    platform_id TEXT PRIMARY KEY, -- e.g. steam:7656...
    player_id TEXT NOT NULL REFERENCES players(id),
    bc_name TEXT
);

CREATE TABLE IF NOT EXISTS event_rosters (
    event_root TEXT NOT NULL,
    team_id INTEGER NOT NULL REFERENCES teams(id),
    player_id TEXT NOT NULL REFERENCES players(id),
    slot INTEGER NOT NULL,        -- 1..3 starters, 4+ subs
    PRIMARY KEY (event_root, team_id, player_id)
);

CREATE TABLE IF NOT EXISTS series (
    id TEXT PRIMARY KEY,          -- "<page>#<match key>"
    page TEXT NOT NULL,
    event_root TEXT,
    date TEXT,
    tier INTEGER,
    region TEXT,
    section TEXT,
    best_of INTEGER,
    team1_id INTEGER NOT NULL REFERENCES teams(id),
    team2_id INTEGER NOT NULL REFERENCES teams(id),
    score1 INTEGER,
    score2 INTEGER,
    finished INTEGER NOT NULL,
    walkover INTEGER NOT NULL DEFAULT 0,
    bc_group TEXT
);
CREATE INDEX IF NOT EXISTS idx_series_date ON series(date);
CREATE INDEX IF NOT EXISTS idx_series_teams ON series(team1_id, team2_id);

CREATE TABLE IF NOT EXISTS games (
    series_id TEXT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    n INTEGER NOT NULL,
    score1 INTEGER,
    score2 INTEGER,
    winner INTEGER,
    ot INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (series_id, n)
);

CREATE TABLE IF NOT EXISTS player_series_stats (
    series_id TEXT NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    player_id TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    games INTEGER,
    goals REAL, assists REAL, saves REAL, shots REAL, score REAL, demos REAL,
    PRIMARY KEY (series_id, player_id)
);

CREATE TABLE IF NOT EXISTS bc_enrichment (
    series_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,         -- ok | missing | error | unmatched
    detail TEXT,
    fetched_at REAL
);

CREATE TABLE IF NOT EXISTS team_resolution (
    raw_key TEXT PRIMARY KEY,     -- team_key() of a template/link name as written
    page TEXT                     -- Liquipedia team page it resolves to (NULL = unknown)
);

-- Current lineup from the team page's active squad (Liquipedia transfer data).
CREATE TABLE IF NOT EXISTS team_squads (
    team_id INTEGER NOT NULL REFERENCES teams(id),
    player_id TEXT NOT NULL REFERENCES players(id),
    slot INTEGER NOT NULL,        -- 1..3 starters, 4+ subs
    role TEXT,
    join_date TEXT,
    PRIMARY KEY (team_id, player_id)
);

CREATE TABLE IF NOT EXISTS team_squad_fetch (
    team_id INTEGER PRIMARY KEY REFERENCES teams(id),
    fetched_at REAL NOT NULL,
    n_players INTEGER NOT NULL
);

-- r/RocketLeagueEsports posts/comments, logged forward with their own timestamps.
CREATE TABLE IF NOT EXISTS reddit_posts (
    id TEXT PRIMARY KEY,          -- t3_<id> post, t1_<id> comment
    kind TEXT NOT NULL,
    created_utc REAL NOT NULL,
    title TEXT,
    body TEXT,
    compound REAL NOT NULL,       -- VADER compound score
    permalink TEXT,
    fetched_at REAL
);
CREATE INDEX IF NOT EXISTS idx_reddit_created ON reddit_posts(created_utc);

CREATE TABLE IF NOT EXISTS reddit_post_teams (
    post_id TEXT NOT NULL REFERENCES reddit_posts(id),
    team_id INTEGER NOT NULL REFERENCES teams(id),
    terms TEXT,                   -- which names matched
    PRIMARY KEY (post_id, team_id)
);
CREATE INDEX IF NOT EXISTS idx_reddit_team ON reddit_post_teams(team_id);

CREATE TABLE IF NOT EXISTS unresolved_teams (
    page TEXT NOT NULL,
    raw_name TEXT NOT NULL,
    PRIMARY KEY (page, raw_name)
);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


# ---- team helpers ----------------------------------------------------------
def get_or_create_team(conn: sqlite3.Connection, key: str, name: str, region: str | None = None) -> int:
    row = conn.execute(
        "SELECT team_id FROM team_aliases WHERE alias = ?", (key,)
    ).fetchone()
    if row:
        return row[0]
    row = conn.execute("SELECT id FROM teams WHERE key = ?", (key,)).fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO teams (key, name, region) VALUES (?,?,?)", (key, name, region))
    return cur.lastrowid


def find_team(conn: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    """Resolve a user-typed team name. Exact key/alias first, then substring by recency."""
    from rlpredictor.data.wikitext import team_key

    k = team_key(query)
    row = conn.execute(
        "SELECT t.* FROM teams t LEFT JOIN team_aliases a ON a.team_id = t.id "
        "WHERE t.key = ? OR a.alias = ? LIMIT 1", (k, k)
    ).fetchone()
    if row:
        return [row]
    return conn.execute(
        """
        SELECT t.*, MAX(s.date) AS last_played, COUNT(s.id) AS n_series
        FROM teams t LEFT JOIN series s ON (s.team1_id = t.id OR s.team2_id = t.id)
        WHERE t.key LIKE ? OR t.id IN (SELECT team_id FROM team_aliases WHERE alias LIKE ?)
        GROUP BY t.id ORDER BY last_played DESC NULLS LAST, n_series DESC LIMIT 10
        """,
        (f"%{k}%", f"%{k}%"),
    ).fetchall()
