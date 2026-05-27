"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Web scraper for past Liquipedia head-to-head match history and Ballchasing API wrapper.
- Usage: Imported by main.py and chat.py to fetch H2H stats and replay details.
"""

import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlencode

from bs4 import BeautifulSoup
import pandas as pd
import requests

from utils.database import (
    initialize_database,
    get_connection,
    get_cached_api_response,
    cache_api_response,
    cache_replay,
    cache_player_stats,
    load_player_id_map_db,
    import_ids_json,
)
from utils.player_identity import normalize_player_name, auto_detect_aliases


ID_FILE = Path(__file__).resolve().parents[1] / "data" / "ids.json"
PLAYER_ID_RE = re.compile(r"^(steam|epic|xbox|ps|psn|ps4|ps5):", re.I)

LP_BASE = "https://liquipedia.net"
LP_RL = f"{LP_BASE}/rocketleague"
BC_API = "https://ballchasing.com/api"

HEADERS = {
    "User-Agent": "RL-PredictorBot/1.0 (https://example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


def canon(s: str) -> str:
    """
    Description:
        Standardizes string formats for uniform normalization.
    Arguments:
        s: Raw string.
    Returns:
        String: Standardized lowercased normalized string.
    """
    if not s: return ""
    s = unicodedata.normalize("NFKC", s).replace("\u200b", "")
    return " ".join(s.strip().split()).lower()


def load_player_id_map(path: Path = ID_FILE) -> dict:
    """
    Description:
        Loads the player platform/alias ID mapping dictionary.
    Arguments:
        path: Path object to ids.json file.
    Returns:
        Dictionary mapping players and aliases to platform IDs.
    """
    try:
        initialize_database()
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
        if count == 0 and path.exists(): import_ids_json(str(path), conn)
        data = load_player_id_map_db(conn)
        conn.close()
        return data
    except Exception:
        if not path.exists(): return {"aliases": {}, "players": {}}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        aliases = {canon(k): v for k, v in (data.get("aliases") or {}).items()}
        players = {}
        for k, v in (data.get("players") or {}).items():
            key = canon(k)
            ids = v if isinstance(v, list) else [v]
            clean = [pid for pid in ids if isinstance(pid, str) and PLAYER_ID_RE.search(pid)]
            if clean: players[key] = clean
        return {"aliases": aliases, "players": players}


def resolve_ids(names, idmap) -> list[str]:
    """
    Description:
        Resolves a list of player names into their respective platform IDs.
    Arguments:
        names: List of player names.
        idmap: Dictionary containing the player ID maps.
    Returns:
        List of resolved platform IDs.
    """
    if not names: return []
    aliases = idmap.get("aliases", {})
    table = idmap.get("players", {})
    out = []
    for name in names:
        if not name: continue
        c = normalize_player_name(name)
        if c in aliases: c = normalize_player_name(aliases[c])
        ids = table.get(c)
        if ids: out.extend(ids)
    seen, uniq = set(), []
    for pid in out:
        if pid not in seen: uniq.append(pid); seen.add(pid)
    return uniq


def fetch_soup(url, session=None):
    """
    Description:
        Requests a page URL and parses it utilizing BeautifulSoup.
    Arguments:
        url: Request target URL string.
        session: Requests Session object.
    Returns:
        BeautifulSoup object parsed from HTML response.
    """
    sess = session or requests.Session()
    r = sess.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def build_h2h(t1, t2):
    """
    Description:
        Generates the Liquipedia head-to-head lookup query URL.
    Arguments:
        t1: Team 1 name.
        t2: Team 2 name.
    Returns:
        String: Liquipedia Head-to-Head query URL.
    """
    params = {
        "Headtohead[team1]": t1,
        "Headtohead[team2]": t2,
        "RunQuery": "Run",
        "pfRunQueryFormName": "Head2head"
    }
    return f"{LP_RL}/Special:RunQuery/Head2head?{urlencode(params)}"


def parse_h2h(t1, t2):
    """
    Description:
        Queries Liquipedia for previous H2H match history details.
    Arguments:
        t1: Team 1 name.
        t2: Team 2 name.
    Returns:
        List of dictionaries with past series details.
    """
    url = build_h2h(t1, t2)
    s = fetch_soup(url)
    rows = []

    for tr in s.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2: continue
        
        a = tr.select_one("a[href*='/rocketleague/']")
        if not a: continue
        href = a.get("href")
        if not href: continue
        ml = href if href.startswith("http") else (LP_BASE + href)
        date = (tds[0].get_text(" ", strip=True) if tds else " ")[:32]
        score = tr.get_text(" ", strip=True)
        rows.append({"date": date, "matchLink": ml, "score": score})
    return rows


BC_ID_RE = re.compile(r"(?:ballchasing\.com/(?:replay|group)/)([A-Za-z0-9-]+)")


def extract_ballchasing(url, session):
    """
    Description:
        Scrapes a page for Ballchasing replay or group IDs.
    Arguments:
        url: target URL string.
        session: Requests Session object.
    Returns:
        List of tuples: (type, ID) found on the page.
    """
    s = fetch_soup(url, session=session)
    out = []

    for a in s.select("a[href*='ballchasing.com']"):
        href = a.get("href") or "" 
        m = BC_ID_RE.search(href)
        if m:
            rid = m.group(1)
            tt = "group" if "/group/" in href else "replay"
            out.append((tt, rid))
    return out


class Ballchasing:
    """
    Description:
        Ballchasing API wrapper class with SQLite-backed caching mechanism.
    """

    def __init__(self, key=None, delay=0.35):
        """
        Description:
            Initializes the Ballchasing client.
        Arguments:
            key: Ballchasing API Key.
            delay: Time delay between API requests.
        Returns:
            None
        """
        self.key = key or os.getenv("BALLCHASING_API_KEY") or ""
        if not self.key: raise RuntimeError("set BALLCHASING API KEY env or pass key=...")
        self.sess = requests.Session()
        self.sess.headers.update({"Authorization": self.key, "Accept": "application/json"})
        self.delay = delay
        initialize_database()

    def cache_key(self, path, params=None):
        """
        Description:
            Generates the MD5 cache key identifier.
        Arguments:
            path: API Endpoint string.
            params: API Request parameters.
        Returns:
            String: MD5 hash string representing the cache key.
        """
        key_str = f"{path}?{urlencode(params or {})}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def fetch_api(self, path, params=None):
        """
        Description:
            Fetches parsed JSON data from the API endpoint (hits local cache first).
        Arguments:
            path: API Endpoint path string.
            params: API Request parameters dictionary.
        Returns:
            Parsed JSON dictionary response.
        """
        ckey = self.cache_key(path, params)
        cached = get_cached_api_response(ckey)
        if cached is not None: return cached

        url = f"{BC_API}{path}"
        r = self.sess.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.25)
            r = self.sess.get(url, params=params, timeout=30)
        r.raise_for_status()
        time.sleep(self.delay)
        
        data = r.json()
        raw = json.dumps(data)
        cache_api_response(ckey, path, raw)

        if isinstance(data, dict) and "blue" in data and "orange" in data:
            persist_replay(data)

        return data
    
    def get_replay(self, replay_id):
        """
        Description:
            Fetches parsed JSON details of a single replay.
        Arguments:
            replay_id: Ballchasing Replay ID string.
        Returns:
            Replay detail dictionary.
        """
        return self.fetch_api(f"/replays/{replay_id}")

    def get_group(self, group_id):
        """
        Description:
            Fetches parsed JSON details of a series group.
        Arguments:
            group_id: Ballchasing group ID string.
        Returns:
            Group detail dictionary.
        """
        return self.fetch_api(f"/groups/{group_id}")

    def list_replays(self, **params):
        """
        Description:
            Fetches parsed JSON listing replays.
        Arguments:
            params: Query parameters.
        Returns:
            Replay list dictionary.
        """
        return self.fetch_api("/replays", params=params)


def persist_replay(detail: dict):
    """
    Description:
        Extracts and records parsed replay information into local database tables.
    Arguments:
        detail: Replay detail parsed JSON dictionary.
    Returns:
        None
    """
    rid = detail.get("id")
    if not rid: return
    date_val = str(detail.get("date", ""))
    playlist_id = str(detail.get("playlist_id", ""))
    playlist_name = str(detail.get("playlist_name", ""))
    raw = json.dumps(detail)

    conn = get_connection()
    try:
        cache_replay(rid, date_val, playlist_id, playlist_name, raw, conn=conn)

        for side in ("blue", "orange"):
            team = detail.get(side) or {}
            for pl in team.get("players", []) or []:
                name = pl.get("name") or (pl.get("player") or {}).get("name")
                if not name: continue
                    
                plat = (pl.get("id") or {}).get("platform")
                p_id = (pl.get("id") or {}).get("id")
                
                cid = auto_detect_aliases(name, p_id, plat, date_val, conn=conn)
                c_name = conn.execute("SELECT canonical_name FROM canonical_players WHERE canonical_player_id = ?", (cid,)).fetchone()["canonical_name"]

                stats = pl.get("stats") or {}
                core = stats.get("core") or {}
                demo = stats.get("demo") or {}
                goals = core.get("goals", 0)
                shots = core.get("shots", 0)
                cache_player_stats(
                    rid, cid, c_name, name, p_id, plat,
                    side,
                    goals, shots,
                    core.get("saves", 0),
                    demo.get("inflicted", 0),
                    core.get("score", 0),
                    (goals / shots) if shots else 0.0,
                    date_val,
                    conn=conn,
                )
        conn.commit()
    finally:
        conn.close()


def players_in_replay(detail):
    """
    Description:
        Extracts raw player names active in a single replay detail.
    Arguments:
        detail: Replay detail dictionary.
    Returns:
        List of raw player name strings.
    """
    out = []
    blue = (detail.get("blue") or {}).get("players") or []
    orange = (detail.get("orange") or {}).get("players") or []

    for pl in blue + orange:
        name = pl.get("name") or (pl.get("player") or {}).get("name")
        if name: out.append(name)
    return out


def extract_stats(detail):
    """
    Description:
        Extracts stats data rows per player from a replay detail structure.
    Arguments:
        detail: Replay detail parsed JSON dictionary.
    Returns:
        List of dictionaries with stats.
    """
    rows = []
    date_val = detail.get("date")
    for side in ("blue", "orange"):
        team = detail.get(side) or {}
        for pl in team.get("players", []) or []:
            name = pl.get("name") or (pl.get("player") or {}).get("name")
            if not name: continue
            
            plat = (pl.get("id") or {}).get("platform")
            p_id = (pl.get("id") or {}).get("id")
            cid = auto_detect_aliases(name, p_id, plat, date_val)
            
            stats = pl.get("stats") or {}
            core = stats.get("core") or {}
            demo = stats.get("demo") or {}
            rows.append({
                "canonical_player_id": cid,
                "Player": name,
                "Goals": core.get("goals", 0),
                "Shots": core.get("shots", 0),
                "Shot %": (core.get("goals", 0) / core.get("shots", 1)) if core.get("shots") else 0.0,
                "Saves": core.get("saves", 0),
                "Demos": demo.get("inflicted", 0),
                "replay_id": detail.get("id"),
                "Date": detail.get("date")
            })
    return rows


def aggregate_players(rows):
    """
    Description:
        Aggregates raw player stats into consolidated records.
    Arguments:
        rows: List of player stats dictionaries.
    Returns:
        Pandas DataFrame containing aggregated player statistics.
    """
    if not rows: return pd.DataFrame(columns=["canonical_player_id", "Player", "Games", "Goals", "Shots", "Shot %", "Saves", "Demos"])
    df = pd.DataFrame(rows)
    g = df.groupby("canonical_player_id", dropna=False).agg(
        Player=("Player", "first"),
        Games=("replay_id", "nunique"),
        Goals=("Goals", "sum"),
        Shots=("Shots", "sum"),
        Saves=("Saves", "sum"),
        Demos=("Demos", "sum"), 
    ).reset_index()
    g["Shot %"] = g.apply(lambda r: (r["Goals"] / r["Shots"]) if r["Shots"] else 0.0, axis=1)
    return g[["canonical_player_id", "Player", "Games", "Goals", "Shots", "Shot %", "Saves", "Demos"]].sort_values(["Games", "Shot %"], ascending=[False, False])


def get_h2h_stats(t1, t2, r1, r2, bc: Ballchasing, limit: int = 6, fallback: int = 30):
    """
    Description:
        Resolves active rosters and fetches direct H2H statistics from private replays.
    Arguments:
        t1: Team 1 name.
        t2: Team 2 name.
        r1: Roster player names for Team 1.
        r2: Roster player names for Team 2.
        bc: Ballchasing client.
        limit: Max direct replays to query details for.
        fallback: Total private games list count to pull.
    Returns:
        Tuple: (Pandas DataFrame containing stats, List of query logs).
    """
    logs = []
    id_map = load_player_id_map()
    ids1 = resolve_ids(r1, id_map)
    ids2 = resolve_ids(r2, id_map)
    
    if not ids1 or not ids2:
        logs.append("Could not resolve player IDs for both teams to perform H2H.")
        return pd.DataFrame(), logs
        
    p1_str = ids1[0]
    p2_str = ids2[0]
    logs.append(f"Querying Ballchasing for private matches containing {p1_str} and {p2_str}...")
    
    params = [
        ("player-id", p1_str),
        ("player-id", p2_str),
        ("playlist", "private"),
        ("count", fallback)
    ]
    
    try:
        key_str = f"h2h_replays_{p1_str}_{p2_str}_{fallback}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        
        cached = get_cached_api_response(cache_key)
        if cached is not None:
            data = cached
        else:
            url = "https://ballchasing.com/api/replays"
            r = bc.sess.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            time.sleep(bc.delay)
            cache_api_response(cache_key, "h2h_replays", json.dumps(data))
            
    except Exception as e:
        logs.append(f"Ballchasing API request failed: {e}")
        return pd.DataFrame(), logs
        
    replays = data.get("list", [])
    if not replays:
        logs.append("No direct H2H replays found on Ballchasing.")
        return pd.DataFrame(), logs
        
    replays = replays[:limit]
    
    per_player_rows = []
    for rep in replays:
        rid = rep.get("id")
        if not rid: continue
        try:
            d = bc.get_replay(rid)
            per_player_rows.extend(extract_stats(d))
        except Exception as e:
            logs.append(f"Failed to fetch replay stats {rid}: {e}")
            continue
            
    df = pd.DataFrame(per_player_rows)
    return df, logs
