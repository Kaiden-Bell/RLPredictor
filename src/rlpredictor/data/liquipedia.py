"""Liquipedia MediaWiki API client (keyless, read-only).

Follows the Liquipedia API terms: descriptive User-Agent, gzip, <= 1 request / 2 s,
and batching up to 50 titles per request. Rendered-page scraping is never used.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Iterator

from rlpredictor import config
from rlpredictor.data.http import ThrottledClient

BATCH = 50


@dataclass
class Page:
    title: str
    revid: int | None
    timestamp: str | None
    text: str | None


class Liquipedia:
    def __init__(self, conn: sqlite3.Connection | None, user_agent: str | None = None):
        self.http = ThrottledClient(
            conn,
            headers={
                "User-Agent": user_agent or config.liquipedia_user_agent(),
                "Accept-Encoding": "gzip",
            },
            min_interval=config.LIQUIPEDIA_MIN_INTERVAL,
        )
        self._expand = ThrottledClient(
            conn, headers=dict(self.http.session.headers), min_interval=30.0
        )

    def _get(self, params: dict, ttl: float | None, cache: bool = True) -> dict:
        params = {**params, "format": "json", "formatversion": 2}
        return self.http.get_json(
            config.LIQUIPEDIA_API, params=params, ttl=ttl, use_cache=cache, cache=cache
        )

    def list_pages(self, prefix: str, ttl: float = 6 * 3600) -> list[str]:
        """All page titles under a prefix (namespace 0)."""
        titles: list[str] = []
        cont: dict = {}
        while True:
            d = self._get(
                {"action": "query", "list": "allpages", "apprefix": prefix, "aplimit": 500, **cont},
                ttl=ttl,
            )
            titles += [p["title"] for p in d.get("query", {}).get("allpages", [])]
            if "continue" in d:
                cont = {k: v for k, v in d["continue"].items()}
            else:
                return titles

    def revisions(self, titles: list[str], ttl: float = 3600) -> dict[str, int]:
        """Latest revision id per title (cheap: no content)."""
        out: dict[str, int] = {}
        for chunk in _chunks(titles, BATCH):
            d = self._get(
                {"action": "query", "prop": "revisions", "rvprop": "ids", "titles": "|".join(chunk)},
                ttl=ttl,
            )
            for p in d.get("query", {}).get("pages", []):
                if p.get("revisions"):
                    out[p["title"]] = p["revisions"][0]["revid"]
        return out

    def fetch(self, titles: list[str]) -> Iterator[Page]:
        """Wikitext for many pages, 50 per request. Not HTTP-cached: callers store parsed
        results keyed by revision id and only re-fetch pages whose revid changed."""
        for chunk in _chunks(titles, BATCH):
            d = self._get(
                {
                    "action": "query",
                    "prop": "revisions",
                    "rvprop": "content|ids|timestamp",
                    "rvslots": "main",
                    "titles": "|".join(chunk),
                },
                ttl=None,
                cache=False,
            )
            for p in d.get("query", {}).get("pages", []):
                revs = p.get("revisions") or []
                if not revs:
                    yield Page(p["title"], None, None, None)
                    continue
                r = revs[0]
                yield Page(p["title"], r.get("revid"), r.get("timestamp"),
                           r.get("slots", {}).get("main", {}).get("content"))


    def resolve_team_pages(self, names: list[str], batch: int = 250) -> dict[str, str | None]:
        """Map team-template names (e.g. "nip") to team page titles ("Ninjas in Pyjamas")
        by expanding {{TeamPage|name}}. Heavier than a query, so it uses its own 30 s
        throttle and results are cached for 30 days."""
        out: dict[str, str | None] = {}
        safe = [n for n in dict.fromkeys(names) if n and not re.search(r"[|{}\[\]<>§\n]", n)]
        for n in names:
            if n not in safe:
                out[n] = None
        for chunk in _chunks(sorted(safe), batch):
            text = "".join(f"§{i}§{{{{TeamPage|{n}}}}}\n" for i, n in enumerate(chunk))
            d = self._expand.get_json(
                config.LIQUIPEDIA_API,
                params={"action": "expandtemplates", "prop": "wikitext", "text": text,
                        "format": "json", "formatversion": 2},
                ttl=30 * 86400,
                method="POST",
            )
            expanded = d.get("expandtemplates", {}).get("wikitext", "")
            for m in re.finditer(r"§(\d+)§(.*)", expanded):
                val = m.group(2).strip()
                ok = val and "<" not in val and "error" not in val.lower()
                out[chunk[int(m.group(1))]] = val if ok else None
        return out

    def active_squads(self, teams: list[str], batch: int = 40,
                      ttl: float = 12 * 3600, cache: bool = True) -> dict[str, str | None]:
        """Expanded `{{ActiveSquadAuto|team=...}}` per team page title. Team pages only hold
        that Lua template (filled from Liquipedia's transfer data), so the squad has to be
        expanded rather than read from page wikitext. Shares the 30 s expandtemplates throttle."""
        from rlpredictor.data.wikitext import split_expanded

        out: dict[str, str | None] = {}
        safe = [t for t in dict.fromkeys(teams) if t and not re.search(r"[|{}\[\]<>§\n=]", t)]
        for t in teams:
            out.setdefault(t, None)
        for chunk in _chunks(safe, batch):
            text = "".join(f"§{i}§{{{{ActiveSquadAuto|team={t}}}}}\n" for i, t in enumerate(chunk))
            d = self._expand.get_json(
                config.LIQUIPEDIA_API,
                params={"action": "expandtemplates", "prop": "wikitext", "text": text,
                        "title": "Main Page", "format": "json", "formatversion": 2},
                ttl=ttl, use_cache=cache, cache=cache, method="POST",
            )
            for i, part in split_expanded(d.get("expandtemplates", {}).get("wikitext", "")).items():
                if i < len(chunk):
                    out[chunk[i]] = part
        return out


def _chunks(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
