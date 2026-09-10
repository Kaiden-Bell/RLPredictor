import os, re, time, requests, pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus, urlencode
from bs4 import BeautifulSoup
import json, unicodedata
from pathlib import Path


ID_FILE = Path(__file__).resolve().parents[1] / "data" / "ids.json"

_PLAYER_ID_RE = re.compile(r"^(steam|epic|xbox|ps|psn|ps4|ps5):", re.I)

def _canon(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s).replace("\u200b", "")
    return " ".join(s.strip().split()).lower()

def load_player_id_map(path: Path = ID_FILE) -> dict:
    if not path.exists():
        return {"aliases": {}, "players": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    aliases = { _canon(k): v for k, v in (data.get("aliases") or {}).items() }
    players = {}
    for k, v in (data.get("players") or {}).items():
        key = _canon(k)
        ids = v if isinstance(v, list) else [v]
        clean = [pid for pid in ids if isinstance(pid, str) and _PLAYER_ID_RE.search(pid)]
        if clean:
            players[key] = clean
    return {"aliases": aliases, "players": players}

def resolve_ids(names, idmap) -> list[str]:
    if not names: return []
    aliases = idmap.get("aliases", {})
    table   = idmap.get("players", {})
    out = []
    for name in names:
        if not name: continue
        c = _canon(name)
        if c in aliases:
            c = _canon(aliases[c])
        ids = table.get(c)
        if ids:
            out.extend(ids)
    seen, uniq = set(), []
    for pid in out:
        if pid not in seen:
            uniq.append(pid); seen.add(pid)
    return uniq




LP_BASE = "https://liquipedia.net"
LP_RL = f"{LP_BASE}/rocketleague"
BC_API = "https://ballchasing.com/api"

HEADERS = {
    "User-Agent": "RL-PredictorBot/1.0 (https://example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}

def _soup(url, session=None):
    sess = session or requests.Session()
    r = sess.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# Step 1.) Fetch LP H2H

def buildH2H(t1, t2):
    params = {
        "Headtohead[team1]": t1,
        "Headtohead[team2]": t2,
        "RunQuery": "Run",
        "pfRunQueryFormName": "Head2head"
    }

    return f"{LP_RL}/Special:RunQuery/Head2head?{urlencode(params)}"

def parseH2H(t1, t2):
    # Return a list of key terms from past series (date, event, match link, score)
    url = buildH2H(t1, t2)
    s = _soup(url)
    rows = []

    for tr in s.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        
        a = tr.select_one("a[href*='/rocketleague/']")
        if not a:
            continue
        href = a.get("href")
        if not href:
            continue
        ml = href if href.startswith("http") else (LP_BASE + href)
        date = (tds[0].get_text(" ", strip=True) if tds else " ")[:32]
        score = tr.get_text(" ", strip=True)
        rows.append({"date": date, "matchLink": ml, "score": score})
    return rows

BC_ID_RE = re.compile(r"(?:ballchasing\.com/(?:replay|group)/)([A-Za-z0-9-]+)")

def extractBallchasing(url, session):
    s = _soup(url, session=session)
    out = []

    for a in s.select("a[href*='ballchasing.com']"):
        href = a.get("href") or "" 
        m = BC_ID_RE.search(href)

        if m:
            rid = m.group(1)
            tt = "group" if "/group/" in href else "replay"
            out.append((tt, rid))

    return out

# Ballchasing API setup

import hashlib

class Ballchasing:
    def __init__(self, key=None, delay=0.35, cache_file=".bc_cache.json"):
        self.key = key or os.getenv("BALLCHASING_API_KEY") or ""
        if not self.key:
            raise RuntimeError("set BALLCHASING API KEY env or pass key=...")
        self.sess = requests.Session()
        self.sess.headers.update({"Authorization": self.key, "Accept": "application/json"})
        self.delay = delay
        self.cache_file = cache_file
        self.cache = {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except:
                pass

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def __get(self, path, params=None):
        key_str = f"{path}?{urlencode(params or {})}"
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = f"{BC_API}{path}"
        r = self.sess.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.25)
            r = self.sess.get(url, params=params, timeout=30)
        r.raise_for_status()
        time.sleep(self.delay)
        
        data = r.json()
        self.cache[cache_key] = data
        self._save_cache()
        return data
    
    def getReplay(self, replayID):
        return self.__get(f"/replays/{replayID}")
    def getGroup(self, groupID):
        return self.__get(f"/groups/{groupID}")
    def listReplays(self, **params):
        return self.__get("/replays", params=params)
    
# Parse Rosters, Players, Stats

def playersInReplay(detail):
    out = []

    blue = (detail.get("blue") or {}).get("players") or []
    orange = (detail.get("orange") or {}).get("players") or []

    for pl in blue + orange:
        name = pl.get("name") or (pl.get("player") or {}).get("name")
        if name: out.append(name)

    return out

def extractStats(detail):
    rows = []
    for side in ("blue", "orange"):
        team = (detail.get(side) or {})
        for pl in team.get("players", []) or []:
            name = pl.get("name") or (pl.get("player") or {}).get("name")
            stats = (pl.get("stats") or {})
            core = stats.get("core") or {}
            demo = stats.get("demo") or {}
            rows.append({
                "Player": name,
                "Goals": core.get("goals", 0),
                "Shots": core.get("shots", 0),
                "Shot %": (core.get("goals",0) / core.get("shots",1)) if core.get("shots") else 0.0,
                "Saves": core.get("saves", 0),
                "Demos": demo.get("inflicted", 0),
                "replay_id": detail.get("id"),
                "Date": detail.get("date")
            })
    return rows



def aggregatePlayers(rows):
    if not rows:
        return pd.DataFrame(columns=["Player", "Games", "Goals", "Shots", "Shot %", "Saves", "Demos"])
    df = pd.DataFrame(rows)
    g = df.groupby("Player", dropna=False).agg(
        Games = ("replay_id", "nunique"),
        Goals = ("Goals", "sum"),
        Shots = ("Shots", "sum"),
        Saves = ("Saves", "sum"),
        Demos = ("Demos", "sum"), 
    ).reset_index()
    g["Shot %"] = g.apply(lambda r: (r["Goals"]/r["Shots"]) if r["Shots"] else 0.0, axis=1)
    return g[["Player", "Games", "Goals", "Shots", "Shot %", "Saves", "Demos"]].sort_values(["Games", "Shot %"], ascending=[False, False])


def getH2HStats(t1, t2, r1, r2, bc: Ballchasing, limit: int=6, fallback: int=30):
    logs = []
    
    idMap = load_player_id_map()
    ids1 = resolve_ids(r1, idMap)
    ids2 = resolve_ids(r2, idMap)
    
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
        
        if cache_key in bc.cache:
            data = bc.cache[cache_key]
        else:
            url = "https://ballchasing.com/api/replays"
            r = bc.sess.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            time.sleep(bc.delay)
            bc.cache[cache_key] = data
            bc._save_cache()
            
    except Exception as e:
        logs.append(f"Ballchasing API request failed: {e}")
        return pd.DataFrame(), logs
        
    replays = data.get("list", [])
    if not replays:
        logs.append("No direct H2H replays found on Ballchasing.")
        return pd.DataFrame(), logs
        
    replays = replays[:limit]
    
    perPlayerRows = []
    for rep in replays:
        rid = rep.get("id")
        if not rid: continue
        try:
            d = bc.getReplay(rid)
            perPlayerRows.extend(extractStats(d))
        except Exception as e:
            logs.append(f"Failed to fetch replay stats {rid}: {e}")
            continue
            
    df = pd.DataFrame(perPlayerRows)
    return df, logs
