"""r/RocketLeagueEsports community sentiment, logged going forward (context only).

Posts and top-level comments are stored with their own `created_utc`, so a prediction
for date D only ever sees text written before D. There is no point-in-time archive for
past seasons, which is why this is shown as unweighted context: once enough matches
have pre-match sentiment logged, `rlpredict backtest` reports whether it would help.

Auth is Reddit's app-only OAuth (`client_credentials`); create a "script" app at
https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET /
REDDIT_USER_AGENT in .env.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable

import requests

from rlpredictor import config
from rlpredictor.data.http import ThrottledClient

log = logging.getLogger(__name__)

SUBREDDIT = "RocketLeagueEsports"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_API = "https://oauth.reddit.com"
MIN_TERM = 4          # shorter player names ("zen", "Lj") match too much unrelated text
WINDOW_DAYS = 14


# ---- scoring & matching (pure) ---------------------------------------------
_analyzer = None

# VADER is a general-purpose lexicon; these esports usages flip or add polarity
# (VADER's scale is roughly -4..+4). e.g. stock VADER scores "Vatira is cracked" negative.
ESPORTS_LEXICON = {
    "cracked": 2.5, "goated": 3.0, "goat": 2.5, "nutty": 2.0, "clutch": 2.0,
    "cooking": 1.5, "cooked": -2.0, "washed": -2.5, "throw": -1.5, "throwing": -1.8,
    "threw": -1.8, "choke": -2.0, "choked": -2.2, "choking": -2.0, "carried": 1.0,
    "diff": 1.0, "fraud": -2.5, "frauds": -2.5, "mid": -1.0, "bottled": -2.0,
    "dominant": 2.0, "dominated": 1.5, "sweep": 1.5, "swept": -0.5,
}


def compound(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    global _analyzer
    if _analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as e:
            raise config.ConfigError(
                "Sentiment needs the optional dependency: pip install -e '.[sentiment]'") from e
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(ESPORTS_LEXICON)
    return float(_analyzer.polarity_scores(text[:5000])["compound"])


def team_terms(conn: sqlite3.Connection, team_id: int) -> list[str]:
    """Team name, longer aliases, and current players (team-page squad, else last event)."""
    name = conn.execute("SELECT name FROM teams WHERE id = ?", (team_id,)).fetchone()[0]
    terms = [re.sub(r"\s*\(.*?\)$", "", name)]
    terms += [a for (a,) in conn.execute("SELECT alias FROM team_aliases WHERE team_id = ?",
                                         (team_id,)) if len(a) >= MIN_TERM]
    players = conn.execute(
        """SELECT p.name FROM team_squads s JOIN players p ON p.id = s.player_id
           WHERE s.team_id = ? AND s.slot <= 3 ORDER BY s.slot""", (team_id,)).fetchall()
    if not players:
        players = conn.execute(
            """SELECT p.name FROM event_rosters er JOIN players p ON p.id = er.player_id
               JOIN pages pg ON pg.title = er.event_root
               WHERE er.team_id = ? AND er.slot <= 3 AND er.event_root = (
                   SELECT er2.event_root FROM event_rosters er2 JOIN pages pg2 ON pg2.title = er2.event_root
                   WHERE er2.team_id = ? ORDER BY pg2.start_date DESC LIMIT 1)
               ORDER BY er.slot""", (team_id, team_id)).fetchall()
    terms += [re.sub(r"\s*\(.*?\)$", "", p[0]) for p in players]
    seen, out = set(), []
    for t in terms:
        k = t.casefold().strip()
        if len(k) >= MIN_TERM and k not in seen:
            seen.add(k)
            out.append(t.strip())
    return out


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms
            if re.search(rf"(?<![\w]){re.escape(t)}(?![\w])", text, flags=re.I)]


# ---- client ----------------------------------------------------------------
class Reddit:
    def __init__(self, conn: sqlite3.Connection | None, client_id: str | None = None,
                 secret: str | None = None, user_agent: str | None = None):
        cid, sec, ua = config.reddit_credentials()
        self.client_id, self.secret = client_id or cid, secret or sec
        ua = user_agent or ua
        self.http = ThrottledClient(conn, headers={"User-Agent": ua}, min_interval=0.7)
        self._expires = 0.0

    def _auth(self) -> None:
        if time.time() < self._expires - 60:
            return
        r = requests.post(TOKEN_URL, auth=(self.client_id, self.secret),
                          data={"grant_type": "client_credentials"},
                          headers={"User-Agent": self.http.session.headers["User-Agent"]},
                          timeout=30)
        if r.status_code != 200:
            raise config.ConfigError(f"Reddit OAuth failed ({r.status_code}): check "
                                     "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET")
        tok = r.json()
        self.http.session.headers["Authorization"] = f"bearer {tok['access_token']}"
        self._expires = time.time() + float(tok.get("expires_in", 3600))

    def search(self, query: str, t: str = "week", limit: int = 100) -> list[dict]:
        self._auth()
        d = self.http.get_json(f"{OAUTH_API}/r/{SUBREDDIT}/search",
                               params={"q": query, "restrict_sr": 1, "sort": "new", "t": t,
                                       "limit": limit, "raw_json": 1, "type": "link"},
                               ttl=1800)
        return [c["data"] for c in d.get("data", {}).get("children", [])]

    def comments(self, post_id: str, limit: int = 100) -> list[dict]:
        self._auth()
        d = self.http.get_json(f"{OAUTH_API}/comments/{post_id}",
                               params={"limit": limit, "depth": 1, "sort": "top", "raw_json": 1},
                               ttl=1800)
        if not isinstance(d, list) or len(d) < 2:
            return []
        return [c["data"] for c in d[1].get("data", {}).get("children", [])
                if c.get("kind") == "t1"]


# ---- storage ---------------------------------------------------------------
@dataclass
class SentimentReport:
    teams: int = 0
    posts_seen: int = 0
    stored: int = 0
    comments_fetched: int = 0
    per_team: dict[str, int] = field(default_factory=dict)


def store_item(conn: sqlite3.Connection, item_id: str, kind: str, created_utc: float,
               text: str, title: str | None, permalink: str | None,
               teams: dict[int, list[str]]) -> bool:
    """Store one post/comment linked to the teams it mentions. Returns True if new."""
    if not teams:
        return False
    new = conn.execute("SELECT 1 FROM reddit_posts WHERE id = ?", (item_id,)).fetchone() is None
    if new:
        conn.execute(
            "INSERT INTO reddit_posts (id, kind, created_utc, title, body, compound, permalink, fetched_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (item_id, kind, created_utc, title, text[:5000], compound(text), permalink, time.time()))
    for tid, terms in teams.items():
        conn.execute("INSERT OR REPLACE INTO reddit_post_teams (post_id, team_id, terms) VALUES (?,?,?)",
                     (item_id, tid, ", ".join(terms)))
    return new


def collect(conn: sqlite3.Connection, rd: Reddit, team_ids: list[int], t: str = "week",
            comment_posts: int = 10,
            progress: Callable[[str], None] = lambda s: None) -> SentimentReport:
    """Search posts mentioning each team, plus top-level comments of the most relevant
    posts (match threads carry most of the opinion)."""
    rep = SentimentReport(teams=len(team_ids))
    terms = {tid: team_terms(conn, tid) for tid in team_ids}
    names = {tid: conn.execute("SELECT name FROM teams WHERE id = ?", (tid,)).fetchone()[0]
             for tid in team_ids}
    thread_scores: dict[str, int] = {}
    for tid in team_ids:
        q = " OR ".join(f'"{x}"' for x in terms[tid])[:500]
        if not q:
            continue
        posts = rd.search(q, t=t)
        rep.posts_seen += len(posts)
        for p in posts:
            text = f"{p.get('title', '')}\n{p.get('selftext', '')}"
            hits = {o: m for o in team_ids if (m := matched_terms(text, terms[o]))}
            if store_item(conn, f"t3_{p['id']}", "post", float(p["created_utc"]), text,
                          p.get("title"), p.get("permalink"), hits):
                rep.stored += 1
            for o in hits:
                rep.per_team[names[o]] = rep.per_team.get(names[o], 0) + 1
            if hits:
                thread_scores[p["id"]] = max(thread_scores.get(p["id"], 0),
                                             int(p.get("num_comments") or 0))
        conn.commit()
        progress(f"reddit: {names[tid]} — {len(posts)} posts")

    for pid in sorted(thread_scores, key=thread_scores.get, reverse=True)[:comment_posts]:
        for c in rd.comments(pid):
            body = c.get("body") or ""
            hits = {o: m for o in team_ids if (m := matched_terms(body, terms[o]))}
            if store_item(conn, f"t1_{c['id']}", "comment", float(c["created_utc"]), body,
                          None, c.get("permalink"), hits):
                rep.stored += 1
                rep.comments_fetched += 1
        conn.commit()
    return rep


def _cutoff(on: date) -> float:
    """Text must be written before `on` (start of day UTC); for today/future, before now."""
    start = datetime(on.year, on.month, on.day, tzinfo=timezone.utc).timestamp()
    return min(time.time(), start) if on < date.today() else time.time()


def has_data(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM reddit_posts LIMIT 1").fetchone() is not None


def sentiment_for(conn: sqlite3.Connection, team_id: int, on: date,
                  days: int = WINDOW_DAYS) -> dict | None:
    end = _cutoff(on)
    row = conn.execute(
        """SELECT COUNT(*), AVG(p.compound),
                  SUM(p.compound > 0.05), SUM(p.compound < -0.05)
           FROM reddit_posts p JOIN reddit_post_teams pt ON pt.post_id = p.id
           WHERE pt.team_id = ? AND p.created_utc >= ? AND p.created_utc < ?""",
        (team_id, end - days * 86400, end)).fetchone()
    if not row or not row[0]:
        return None
    return {"n": row[0], "mean": row[1], "pos": row[2] or 0, "neg": row[3] or 0}


def evidence(conn: sqlite3.Connection, a: tuple[int, str], b: tuple[int, str], on: date) -> str:
    parts = []
    for tid, name in (a, b):
        s = sentiment_for(conn, tid, on)
        parts.append(f"{name}: no posts logged" if s is None else
                     f"{name} {s['mean']:+.2f} avg over {s['n']} ({s['pos']} positive, {s['neg']} negative)")
    return f"r/{SUBREDDIT}, {WINDOW_DAYS} days before: " + "; ".join(parts)


def coverage(conn: sqlite3.Connection, min_posts: int = 5) -> list[tuple[str, float]]:
    """Finished series where both teams had >= min_posts pre-match items:
    (series id, team1 − team2 mean compound)."""
    first = conn.execute("SELECT MIN(created_utc) FROM reddit_posts").fetchone()[0]
    if first is None:
        return []
    since = str(datetime.fromtimestamp(first, tz=timezone.utc).date())
    out = []
    for s in conn.execute("SELECT id, date, team1_id, team2_id FROM series "
                          "WHERE finished = 1 AND date > ?", (since,)).fetchall():
        on = date.fromisoformat(s[1])
        a, b = sentiment_for(conn, s[2], on), sentiment_for(conn, s[3], on)
        if a and b and a["n"] >= min_posts and b["n"] >= min_posts:
            out.append((s[0], a["mean"] - b["mean"]))
    return out
