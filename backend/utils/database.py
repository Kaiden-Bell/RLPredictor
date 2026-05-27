"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: SQLite caching database utilities for Ballchasing API data, Liquipedia tournaments, and team rosters.
- Usage: Imported across the application to handle fast read/write caching and WAL mode transactions.
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import utils.player_identity as player_identity


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictor.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Description:
        Returns a SQLite database connection with WAL mode enabled.
    Arguments:
        db_path: Absolute Path object to the SQLite db file.
    Returns:
        sqlite3.Connection object.
    """
    os.makedirs(db_path.parent, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: Path = DB_PATH):
    """
    Description:
        Initializes the SQLite schema tables and index constraints.
    Arguments:
        db_path: Absolute Path object to the target database.
    Returns:
        None
    """
    conn = get_connection(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key   TEXT PRIMARY KEY,
            endpoint    TEXT,
            raw_json    TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)

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

    c.execute("""
        CREATE TABLE IF NOT EXISTS player_stats (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            replay_id           TEXT    NOT NULL,
            canonical_player_id TEXT NOT NULL,
            canonical_name      TEXT NOT NULL,
            display_name_seen   TEXT NOT NULL,
            platform_player_id  TEXT,
            platform            TEXT,
            side                TEXT,
            goals               INTEGER DEFAULT 0,
            shots               INTEGER DEFAULT 0,
            saves               INTEGER DEFAULT 0,
            demos               INTEGER DEFAULT 0,
            score               INTEGER DEFAULT 0,
            shot_pct            REAL    DEFAULT 0.0,
            date                TEXT,
            FOREIGN KEY (replay_id) REFERENCES replays(replay_id)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_ps_player  ON player_stats(canonical_player_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_ps_replay  ON player_stats(replay_id)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS canonical_players (
            canonical_player_id TEXT PRIMARY KEY,
            canonical_name      TEXT NOT NULL,
            first_seen          TEXT,
            last_seen           TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS player_ids (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_player_id TEXT NOT NULL,
            platform_id         TEXT NOT NULL,
            UNIQUE(canonical_player_id, platform_id),
            FOREIGN KEY (canonical_player_id) REFERENCES canonical_players(canonical_player_id)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_pid_canon ON player_ids(canonical_player_id)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS player_aliases (
            alias               TEXT PRIMARY KEY,
            canonical_player_id TEXT NOT NULL,
            FOREIGN KEY (canonical_player_id) REFERENCES canonical_players(canonical_player_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS tournaments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT UNIQUE NOT NULL,
            name        TEXT,
            scraped_at  TEXT NOT NULL
        )
    """)

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

    c.execute("""
        CREATE TABLE IF NOT EXISTS roster_cache (
            team_url    TEXT PRIMARY KEY,
            players     TEXT NOT NULL,
            cached_at   TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def cache_api_response(cache_key: str, endpoint: str, raw_json: str, conn: Optional[sqlite3.Connection] = None):
    """
    Description:
        Caches a raw JSON API response mapped to an MD5 key.
    Arguments:
        cache_key: MD5 hashed endpoint key string.
        endpoint: API path string.
        raw_json: Raw JSON string value.
        conn: Optional DB connection.
    Returns:
        None
    """
    own = conn is None
    if own: conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO api_cache (cache_key, endpoint, raw_json, cached_at) VALUES (?, ?, ?, ?)",
        (cache_key, endpoint, raw_json, now),
    )
    if own:
        conn.commit()
        conn.close()


def get_cached_api_response(cache_key: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    """
    Description:
        Retrieves parsed cached JSON dictionary response.
    Arguments:
        cache_key: MD5 query key string.
        conn: Optional DB connection.
    Returns:
        Optional parsed JSON dictionary or None.
    """
    own = conn is None
    if own: conn = get_connection()
    row = conn.execute(
        "SELECT raw_json FROM api_cache WHERE cache_key = ?", (cache_key,)
    ).fetchone()
    if own: conn.close()
    return json.loads(row["raw_json"]) if row else None


def cache_replay(replay_id: str, date: str, playlist_id: str, playlist_name: str, raw_json: str, conn: Optional[sqlite3.Connection] = None):
    """
    Description:
        Saves full replay structures into the local SQLite DB replays table.
    Arguments:
        replay_id: Ballchasing Replay ID.
        date: Replay date string.
        playlist_id: Ballchasing playlist ID.
        playlist_name: Human readable playlist name.
        raw_json: Full detail JSON string.
        conn: Optional DB connection.
    Returns:
        None
    """
    own = conn is None
    if own: conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO replays
           (replay_id, date, playlist_id, playlist_name, raw_json)
           VALUES (?, ?, ?, ?, ?)""",
        (replay_id, date, playlist_id, playlist_name, raw_json),
    )
    if own:
        conn.commit()
        conn.close()


def get_cached_replay(replay_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    """
    Description:
        Retrieves parsed cached JSON replay details.
    Arguments:
        replay_id: Replay ID string.
        conn: Optional DB connection.
    Returns:
        Replay detail parsed JSON dictionary or None.
    """
    own = conn is None
    if own: conn = get_connection()
    row = conn.execute(
        "SELECT raw_json FROM replays WHERE replay_id = ?", (replay_id,)
    ).fetchone()
    if own: conn.close()
    return json.loads(row["raw_json"]) if row else None


def get_all_replay_details(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """
    Description:
        Pulls all cached replay details from SQLite database.
    Arguments:
        conn: Optional DB connection.
    Returns:
        List of parsed replay details.
    """
    own = conn is None
    if own: conn = get_connection()
    rows = conn.execute("SELECT raw_json FROM replays").fetchall()
    if own: conn.close()
    return [json.loads(r["raw_json"]) for r in rows]


def cache_player_stats(replay_id: str, canonical_player_id: str, canonical_name: str,
                       display_name_seen: str, platform_player_id: str, platform: str,
                       side: str, goals: int, shots: int, saves: int, demos: int,
                       score: int, shot_pct: float, date: str, conn: Optional[sqlite3.Connection] = None):
    """
    Description:
        Records stats details for a player in a given game series.
    Arguments:
        replay_id: Replay ID.
        canonical_player_id: Canonical player ID.
        canonical_name: Canonical player name.
        display_name_seen: Raw display name.
        platform_player_id: Unique platform ID.
        platform: Platform name.
        side: Team side (blue/orange).
        goals: Goals count.
        shots: Shots count.
        saves: Saves count.
        demos: Inflicted demos.
        score: Replay score.
        shot_pct: Ratio of goals to shots.
        date: Game date string.
        conn: Optional DB connection.
    Returns:
        None
    """
    own = conn is None
    if own: conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO player_stats
            (replay_id, canonical_player_id, canonical_name, display_name_seen, platform_player_id, platform, side, goals, shots, saves, demos, score, shot_pct, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (replay_id, canonical_player_id, canonical_name, display_name_seen, platform_player_id, platform, side, goals, shots, saves, demos, score, shot_pct, date))
    if own:
        conn.commit()
        conn.close()


def load_player_id_map_db(conn: Optional[sqlite3.Connection] = None) -> dict:
    """
    Description:
        Constructs maps containing player IDs and active aliases from database tables.
    Arguments:
        conn: Optional DB connection.
    Returns:
        Dictionary detailing {"aliases": {}, "players": {}}.
    """
    own = conn is None
    if own: conn = get_connection()

    aliases = {}
    for row in conn.execute("SELECT alias, canonical_player_id FROM player_aliases"):
        aliases[row["alias"]] = row["canonical_player_id"]

    players = {}
    for row in conn.execute("SELECT canonical_player_id, platform_id FROM player_ids"):
        players.setdefault(row["canonical_player_id"], []).append(row["platform_id"])

    if own: conn.close()
    return {"aliases": aliases, "players": players}


def import_ids_json(json_path: str, conn: Optional[sqlite3.Connection] = None):
    """
    Description:
        Performs database imports migrating ids.json contents into SQLite.
    Arguments:
        json_path: Filepath to target JSON.
        conn: Optional DB connection.
    Returns:
        None
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    own = conn is None
    if own: conn = get_connection()

    for alias, canonical in (data.get("aliases") or {}).items():
        n_alias = player_identity.normalize_player_name(alias)
        c_id = player_identity.normalize_player_name(canonical)
        conn.execute("INSERT OR IGNORE INTO canonical_players (canonical_player_id, canonical_name) VALUES (?, ?)", (c_id, canonical))
        conn.execute(
            "INSERT OR REPLACE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)",
            (n_alias, c_id),
        )

    for name, ids in (data.get("players") or {}).items():
        c_id = player_identity.normalize_player_name(name)
        conn.execute("INSERT OR IGNORE INTO canonical_players (canonical_player_id, canonical_name) VALUES (?, ?)", (c_id, name))
        id_list = ids if isinstance(ids, list) else [ids]
        for pid in id_list:
            conn.execute(
                "INSERT OR IGNORE INTO player_ids (canonical_player_id, platform_id) VALUES (?, ?)",
                (c_id, pid),
            )

    conn.commit()
    if own: conn.close()


def migrate_json_cache(cache_path: str = ".bc_cache.json", conn: Optional[sqlite3.Connection] = None) -> int:
    """
    Description:
        Migrates legacy JSON cache file records directly into SQLite DB tables.
    Arguments:
        cache_path: Path to target JSON cache.
        conn: Optional DB connection.
    Returns:
        Integer: Count of migrated cache files.
    """
    cache_file = Path(cache_path)
    if not cache_file.exists(): return 0

    print(f"  Loading {cache_path} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)...")
    with open(cache_file, "r") as f:
        cache = json.load(f)

    own = conn is None
    if own: conn = get_connection()

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
                        if not name: continue
                            
                        plat = (pl.get("id") or {}).get("platform")
                        p_id = (pl.get("id") or {}).get("id")
                        
                        cid = player_identity.auto_detect_aliases(name, p_id, plat, date_val, conn=conn)
                        c_name = conn.execute("SELECT canonical_name FROM canonical_players WHERE canonical_player_id = ?", (cid,)).fetchone()["canonical_name"]

                        stats = pl.get("stats") or {}
                        core = stats.get("core") or {}
                        demo = stats.get("demo") or {}
                        goals = core.get("goals", 0)
                        shots = core.get("shots", 0)
                        conn.execute("""
                            INSERT OR IGNORE INTO player_stats
                                (replay_id, canonical_player_id, canonical_name, display_name_seen, platform_player_id, platform, side, goals, shots, saves, demos, score, shot_pct, date)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            rid, cid, c_name, name, p_id, plat, side,
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
            conn.commit()
            print(f"    ...{count} entries migrated ({replay_count} replays)")

    conn.commit()
    if own: conn.close()

    print(f"  Migration complete: {count} cache entries, {replay_count} replays extracted")
    return count


def save_tournament(url: str, name: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> int:
    """
    Description:
        Records a tournament scraper URL reference, returning its ID.
    Arguments:
        url: target Liquipedia URL.
        name: human readable tournament name.
        conn: Optional DB connection.
    Returns:
        Integer: Unique ID.
    """
    own = conn is None
    if own: conn = get_connection()

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
    Description:
        Saves a series of matchup records for a tournament ID.
    Arguments:
        tournament_id: Tournament DB ID integer.
        rows: List of matchup dictionaries.
        conn: Optional DB connection.
    Returns:
        None
    """
    own = conn is None
    if own: conn = get_connection()
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
    if own: conn.close()


def cache_roster(team_url: str, players: list[str], conn: Optional[sqlite3.Connection] = None):
    """
    Description:
        Saves roster lists of a team for subsequent quick lookups.
    Arguments:
        team_url: Team profile URL string.
        players: List of active roster player name strings.
        conn: Optional DB connection.
    Returns:
        None
    """
    own = conn is None
    if own: conn = get_connection()
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
    Description:
        Retrieves roster details for a cached team URL.
    Arguments:
        team_url: Team URL string.
        conn: Optional DB connection.
    Returns:
        List of active players or None.
    """
    own = conn is None
    if own: conn = get_connection()
    row = conn.execute(
        "SELECT players FROM roster_cache WHERE team_url = ?", (team_url,)
    ).fetchone()
    if own: conn.close()
    return json.loads(row["players"]) if row else None


def export_database_file(destination_path: str) -> bool:
    """
    Description:
        Creates a copy of the SQLite database to a given target filepath.
    Arguments:
        destination_path: Target path string.
    Returns:
        Boolean: True if successful, False otherwise.
    """
    if DB_PATH.exists():
        shutil.copy2(DB_PATH, destination_path)
        return True
    return False


def import_database_file(source_path: str) -> bool:
    """
    Description:
        Overwrites active predictor database with external DB inputs.
    Arguments:
        source_path: 외부 database filepath.
    Returns:
        Boolean: True if successful, False otherwise.
    """
    if not os.path.exists(source_path): return False
    if DB_PATH.exists(): shutil.copy2(DB_PATH, DB_PATH.with_suffix(".db.bak"))
    shutil.copy2(source_path, DB_PATH)
    return True


if __name__ == "__main__":
    print(f"Initializing database at {DB_PATH}")
    initialize_database()

    ids_json = DB_PATH.parent / "ids.json"
    if ids_json.exists():
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
        if count == 0:
            print(f"Migrating {ids_json} -> player_ids + player_aliases ...")
            import_ids_json(str(ids_json), conn)
            pid_count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
            print(f"  Done — {pid_count} player IDs imported.")
        else:
            print(f"player_ids already has {count} rows — skipping ids.json import.")
        conn.close()

    cache_json = DB_PATH.parent / ".bc_cache.json"
    if cache_json.exists():
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0]
        if count == 0:
            print(f"\nMigrating {cache_json.name} -> api_cache + replays + player_stats ...")
            migrate_json_cache(str(cache_json), conn)
        else:
            print(f"api_cache already has {count} rows — skipping .bc_cache.json import.")
        conn.close()

    conn = get_connection()
    tables = ["api_cache", "replays", "player_stats", "player_ids", "player_aliases",
              "tournaments", "matchups", "roster_cache"]
    print(f"\n{'=' * 40}")
    print(f"  Database Summary")
    print(f"{'=' * 40}")
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:20s} {n:>8,} rows")
    print(f"{'=' * 40}")
    conn.close()
    print("Database ready.")
