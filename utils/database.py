# utils/database.py
# SQLite caching layer for RLPredictor.

import sqlite3
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictor.db"

# --------------------
# Connection helpers |
# --------------------

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Returns a connection to the SQLite database with WAL mode enabled."""
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path = DB_PATH):
    """Creates all tables and indexes if they do not already exist."""
    conn = get_connection(db_path)
    c = conn.cursor()

    # ---------------------------------------------------------------------
    # 1. API cache — general-purpose cache for ANY Ballchasing API call.  |
    #    This is the direct replacement for the MD5-keyed .bc_cache.json. |
    #    Keys are the same MD5 hashes the old code used.                  |
    # ---------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key   TEXT PRIMARY KEY,
            endpoint    TEXT,
            raw_json    TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # 2. Replays — dedicated table for replay details only.  |
    #    Extracted from api_cache for fast indexed lookups.  |
    #    This is what features.py and train.py iterate over. |
    # --------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS replays (
            replay_id       TEXT PRIMARY KEY,
            date            TEXT,
            playlist_id     TEXT,
            playlist_name   TEXT,
            raw_json        TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_replay_date     ON replays(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_replay_playlist ON replays(playlist_id)")

    # -------------------------------------------------------------------------
    # 3. Player stats — one row per player per replay.                        |
    #    Mirrors the extractStats() / replayStats() / _extract_player_stats() |
    #    output across h2h_ballchasing.py, stats.py, and features.py.         |
    # -------------------------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id   TEXT    NOT NULL,
            player_name TEXT    NOT NULL,
            side        TEXT,
            goals       INTEGER DEFAULT 0,
            shots       INTEGER DEFAULT 0,
            saves       INTEGER DEFAULT 0,
            demos       INTEGER DEFAULT 0,
            score       INTEGER DEFAULT 0,
            shot_pct    REAL    DEFAULT 0.0,
            date        TEXT,
            FOREIGN KEY (replay_id) REFERENCES replays(replay_id)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_ps_player  ON player_stats(player_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ps_replay  ON player_stats(replay_id)")

    # ----------------------------------------
    # 4. Player IDs — replaces data/ids.json |
    # ----------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS player_ids (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name   TEXT NOT NULL,
            platform_id   TEXT NOT NULL,
            UNIQUE(player_name, platform_id)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_pid_name ON player_ids(player_name)")

    # -------------------
    # 5. Player aliases |
    # -------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS player_aliases (
            alias           TEXT PRIMARY KEY,
            canonical_name  TEXT NOT NULL
        )
    """)

    # -----------------------------------------------------------
    # 6. Tournaments — metadata for each scraped Liquipedia URL |
    # -----------------------------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS tournaments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT UNIQUE NOT NULL,
            name        TEXT,
            scraped_at  TEXT NOT NULL
        )
    """)

    # ------------------------------------
    # 7. Matchups — scraped bracket data |
    # ------------------------------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS matchups (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            tournament_id   INTEGER NOT NULL,
            section         TEXT,
            round           TEXT,
            best_of         INTEGER DEFAULT 7,
            team1           TEXT,
            team2           TEXT,
            team1_url       TEXT,
            team2_url       TEXT,
            team1_players   TEXT,
            team2_players   TEXT,
            FOREIGN KEY (tournament_id) REFERENCES tournaments(id)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_mu_tourney ON matchups(tournament_id)")

    # -----------------
    # 8. Roster cache |
    # -----------------
    c.execute("""
        CREATE TABLE IF NOT EXISTS roster_cache (
            team_url    TEXT PRIMARY KEY,
            players     TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# --------------------------------------------------------------------
# General API cache operations  (replaces .bc_cache.json read/write) |
# --------------------------------------------------------------------

def cache_api_response(cache_key: str, endpoint: str, raw_json: str,
                       conn: Optional[sqlite3.Connection] = None):
    """
        Feat: Insert or update a generic API response in the cache.
        Arguments: 
            cache_key : Str
            endpoint : Str
            raw_json : Str
            conn : Optional[sqlite3.Connection]
        Returns: None
    """
    own = conn is None
    if own:
        conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO api_cache (cache_key, endpoint, raw_json, cached_at) VALUES (?, ?, ?, ?)",
        (cache_key, endpoint, raw_json, now),
    )
    if own:
        conn.commit()
        conn.close()


def get_cached_api_response(cache_key: str,
                            conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    """
        Feat: Return the parsed JSON for a cached API response, or None.
        Arguments: 
            cache_key : Str
            conn : Optional[sqlite3.Connection]
        Returns: Optional[Dict]
    """
    own = conn is None
    if own:
        conn = get_connection()
    row = conn.execute(
        "SELECT raw_json FROM api_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    if own:
        conn.close()
    return json.loads(row["raw_json"]) if row else None


# ---------------------------------------------------------------------------
# Replay cache operations
# ---------------------------------------------------------------------------

def cache_replay(replay_id: str, date: str, playlist_id: str,
                 playlist_name: str, raw_json: str,
                 conn: Optional[sqlite3.Connection] = None):
    """
        Feat: Insert or update a replay in the dedicated replay table.
        Arguments: 
            replay_id : Str
            date : Str
            playlist_id : Str
            playlist_name : Str
            raw_json : Str
            conn : Optional[sqlite3.Connection]
        Returns: None
    """
    own = conn is None
    if own:
        conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO replays
           (replay_id, date, playlist_id, playlist_name, raw_json)
           VALUES (?, ?, ?, ?, ?)""",
        (replay_id, date, playlist_id, playlist_name, raw_json),
    )
    if own:
        conn.commit()
        conn.close()


def get_cached_replay(replay_id: str,
                      conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    """
        Feat: Return the parsed JSON for a cached replay, or None.
        Arguments: 
            replay_id : Str
            conn : Optional[sqlite3.Connection]
        Returns: Optional[Dict]
    """
    own = conn is None
    if own:
        conn = get_connection()
    row = conn.execute(
        "SELECT raw_json FROM replays WHERE replay_id = ?", (replay_id,)
    ).fetchone()
    if own:
        conn.close()
    return json.loads(row["raw_json"]) if row else None


def get_all_replay_details(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """
        Feat: Return ALL cached replay detail dicts. Used by features.py / train.py.
        Arguments: conn : Optional[sqlite3.Connection]
        Returns: List[dict]
    """
    own = conn is None
    if own:
        conn = get_connection()
    rows = conn.execute("SELECT raw_json FROM replays").fetchall()
    if own:
        conn.close()
    return [json.loads(r["raw_json"]) for r in rows]


def cache_player_stats(replay_id: str, player_name: str, side: str,
                       goals: int, shots: int, saves: int, demos: int,
                       score: int, shot_pct: float, date: str,
                       conn: Optional[sqlite3.Connection] = None):
    """
    Feat: Insert a single player-stat row.
    Arguments: 
        replay_id : Str
        player_name : Str
        side : Str
        goals : Int
        shots : Int
        saves : Int
        demos : Int
        score : Int
        shot_pct : Float
        date : Str
        conn : Optional[sqlite3.Connection]
    Returns: None
    """
    own = conn is None
    if own:
        conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO player_stats
            (replay_id, player_name, side, goals, shots, saves, demos, score, shot_pct, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (replay_id, player_name, side, goals, shots, saves, demos, score, shot_pct, date))
    if own:
        conn.commit()
        conn.close()


# ---------------------------------------------------------------------------
# Player ID / alias operations  (replaces ids.json read/write)
# ---------------------------------------------------------------------------

def load_player_id_map_db(conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    Feat: Reconstruct the same {aliases: {}, players: {}} dict from the DB.
    Arguments: conn : Optional[sqlite3.Connection]
    Returns: Dict
    """
    own = conn is None
    if own:
        conn = get_connection()

    aliases = {}
    for row in conn.execute("SELECT alias, canonical_name FROM player_aliases"):
        aliases[row["alias"]] = row["canonical_name"]

    players: dict[str, list[str]] = {}
    for row in conn.execute("SELECT player_name, platform_id FROM player_ids"):
        players.setdefault(row["player_name"], []).append(row["platform_id"])

    if own:
        conn.close()
    return {"aliases": aliases, "players": players}


def import_ids_json(json_path: str, conn: Optional[sqlite3.Connection] = None):
    """
    Feat: One-time migration: load an ids.json file into the DB tables.
    Arguments: 
        json_path : Str
        conn : Optional[sqlite3.Connection]
    Returns: None
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    own = conn is None
    if own:
        conn = get_connection()

    for alias, canonical in (data.get("aliases") or {}).items():
        conn.execute(
            "INSERT OR REPLACE INTO player_aliases (alias, canonical_name) VALUES (?, ?)",
            (alias.strip().lower(), canonical),
        )

    for name, ids in (data.get("players") or {}).items():
        id_list = ids if isinstance(ids, list) else [ids]
        for pid in id_list:
            conn.execute(
                "INSERT OR IGNORE INTO player_ids (player_name, platform_id) VALUES (?, ?)",
                (name.strip().lower(), pid),
            )

    conn.commit()
    if own:
        conn.close()

# ------------------------------------
# Migration: .bc_cache.json → SQLite |
# ------------------------------------

def migrate_json_cache(cache_path: str = ".bc_cache.json", conn: Optional[sqlite3.Connection] = None) -> int:
    """
        Feat: Migrate the existing .bc_cache.json into the SQLite database.
        Arguments: 
            cache_path : Str
            conn : Optional[sqlite3.Connection]
        Returns: Int
    """
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return 0

    print(f"  Loading {cache_path} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)...")
    with open(cache_file, "r") as f:
        cache = json.load(f)

    own = conn is None
    if own:
        conn = get_connection()

    count = 0
    replay_count = 0
    now = datetime.now(timezone.utc).isoformat()

    for key, data in cache.items():
        raw = json.dumps(data)

        conn.execute(
            "INSERT OR IGNORE INTO api_cache (cache_key, endpoint, raw_json, cached_at) VALUES (?, ?, ?, ?)",
            (key, "", raw, now),
        )

        if isinstance(data, dict) and "blue" in data and "orange" in data:
            rid = data.get("id", "")
            if rid:
                date_val = str(data.get("date", ""))
                playlist_id = str(data.get("playlist_id", ""))
                playlist_name = str(data.get("playlist_name", ""))
                conn.execute(
                    """INSERT OR IGNORE INTO replays
                       (replay_id, date, playlist_id, playlist_name, raw_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (rid, date_val, playlist_id, playlist_name, raw),
                )

                for side in ("blue", "orange"):
                    team = data.get(side) or {}
                    for pl in team.get("players", []) or []:
                        name = pl.get("name") or (pl.get("player") or {}).get("name")
                        if not name:
                            continue
                        stats = pl.get("stats") or {}
                        core = stats.get("core") or {}
                        demo = stats.get("demo") or {}
                        goals = core.get("goals", 0)
                        shots = core.get("shots", 0)
                        conn.execute("""
                            INSERT OR IGNORE INTO player_stats
                                (replay_id, player_name, side, goals, shots, saves, demos, score, shot_pct, date)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            rid, name, side,
                            goals, shots,
                            core.get("saves", 0),
                            demo.get("inflicted", 0),
                            core.get("score", 0),
                            (goals / shots) if shots else 0.0,
                            date_val,
                        ))

                replay_count += 1
        count += 1
        if count % 1000 == 0:
            conn.commit()  # commit to avoid holding too much in mem
            print(f"    ...{count} entries migrated ({replay_count} replays)")

    conn.commit()
    if own:
        conn.close()

    print(f"  Migration complete: {count} cache entries, {replay_count} replays extracted")
    return count

# ---------------------------------
# Tournament & matchup operations |
# ---------------------------------

def save_tournament(url: str, name: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> int:
    """ 
        Feat: Insert a tournament and return its ID (or the existing one).
        Arguments: url : Str
        Returns: Int
    """
    own = conn is None
    if own:
        conn = get_connection()

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO tournaments (url, name, scraped_at) VALUES (?, ?, ?)",
        (url, name, now),
    )
    row = conn.execute("SELECT id FROM tournaments WHERE url = ?", (url,)).fetchone()
    tid = row["id"]

    if own:
        conn.commit()
        conn.close()
    return tid


def save_matchups(tournament_id: int, rows: list[dict], conn: Optional[sqlite3.Connection] = None):
    """ 
        Feat: Bulk-insert matchup rows for a tournament.
        Arguments: tournament_id : Int
        Returns: None
    """
    own = conn is None
    if own:
        conn = get_connection()

    conn.execute("DELETE FROM matchups WHERE tournament_id = ?", (tournament_id,))

    for r in rows:
        conn.execute("""
            INSERT INTO matchups
                (tournament_id, section, round, best_of,
                 team1, team2, team1_url, team2_url,
                 team1_players, team2_players)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tournament_id,
            r.get("section"), r.get("round"), r.get("best_of", 7),
            r.get("team1"), r.get("team2"),
            r.get("team1_url"), r.get("team2_url"),
            json.dumps(r.get("team1_players", [])),
            json.dumps(r.get("team2_players", [])),
        ))

    conn.commit()
    if own:
        conn.close()


# -------------------------
# Roster cache operations |
# -------------------------

def cache_roster(team_url: str, players: list[str], conn: Optional[sqlite3.Connection] = None):
    """ 
        Feat: Cache a team's roster.
        Arguments: team_url : Str
        Returns: None
    """
    own = conn is None
    if own:
        conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO roster_cache (team_url, players, cached_at) VALUES (?, ?, ?)",
        (team_url, json.dumps(players), now),
    )
    if own:
        conn.commit()
        conn.close()


def get_cached_roster(team_url: str, conn: Optional[sqlite3.Connection] = None) -> Optional[list[str]]:
    """ 
        Feat: Return cached roster list or None.
        Arguments: team_url : Str
        Returns: Optional[list[str]]
    """
    own = conn is None
    if own:
        conn = get_connection()
    row = conn.execute(
        "SELECT players FROM roster_cache WHERE team_url = ?", (team_url,)
    ).fetchone()
    if own:
        conn.close()
    return json.loads(row["players"]) if row else None


# -----------------------------------------------------------------
# Import / Export — for the future webserver data-sharing feature |
# -----------------------------------------------------------------

def export_database_file(destination_path: str) -> bool:
    """ 
        Feat: Copies the current SQLite database to a destination (e.g., for user download).
        Arguments: destination_path : Str
        Returns: bool
    """
    if DB_PATH.exists():
        shutil.copy2(DB_PATH, destination_path)
        return True
    return False


def import_database_file(source_path: str) -> bool:
    """ 
        Feat: Replaces the current SQLite database with an uploaded one.
        Arguments: source_path : Str
        Returns: bool
    """
    if not os.path.exists(source_path):
        return False
    if DB_PATH.exists():
        shutil.copy2(DB_PATH, DB_PATH.with_suffix(".db.bak"))
    shutil.copy2(source_path, DB_PATH)
    return True


# -----------------------------------------------
# CLI entry point — initialise, migrate, verify |
# -----------------------------------------------

if __name__ == "__main__":
    import sys

    print(f"Initializing database at {DB_PATH}")
    initialize_database()

    # Auto-migrate ids.json if it exists and the DB is empty
    ids_json = DB_PATH.parent / "ids.json"
    if ids_json.exists():
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
        if count == 0:
            print(f"Migrating {ids_json} → player_ids + player_aliases ...")
            import_ids_json(str(ids_json), conn)
            pid_count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
            print(f"  Done — {pid_count} player IDs imported.")
        else:
            print(f"player_ids already has {count} rows — skipping ids.json import.")
        conn.close()

    # Auto-migrate .bc_cache.json if it exists and api_cache is empty
    cache_json = Path(__file__).resolve().parents[1] / ".bc_cache.json"
    if cache_json.exists():
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
        if count == 0:
            print(f"\nMigrating {cache_json.name} → api_cache + replays + player_stats ...")
            migrate_json_cache(str(cache_json), conn)
        else:
            print(f"api_cache already has {count} rows — skipping .bc_cache.json import.")
        conn.close()

    conn = get_connection()
    tables = ["api_cache", "replays", "player_stats", "player_ids", "player_aliases",
              "tournaments", "matchups", "roster_cache"]
    print(f"\n{'═' * 40}")
    print(f"  Database Summary")
    print(f"{'═' * 40}")
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:20s} {n:>8,} rows")
    print(f"{'═' * 40}")
    conn.close()
    print("Database ready.")
