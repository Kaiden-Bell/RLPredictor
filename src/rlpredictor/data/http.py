"""Throttled HTTP client with a SQLite-backed response cache.

One `ThrottledClient` per host. Requests are spaced at least `min_interval` seconds
apart across threads, 429/5xx responses back off exponentially (honoring Retry-After),
and successful JSON bodies can be cached with a TTL (None = forever).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from typing import Any

import requests

log = logging.getLogger(__name__)


class HttpError(RuntimeError):
    def __init__(self, status: int, url: str, body: str = ""):
        super().__init__(f"HTTP {status} for {url}: {body[:200]}")
        self.status = status


def cache_key(url: str, params: dict | None) -> str:
    blob = url + "?" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class ThrottledClient:
    def __init__(
        self,
        conn: sqlite3.Connection | None,
        headers: dict[str, str],
        min_interval: float,
        max_retries: int = 5,
        timeout: float = 30.0,
    ):
        self.conn = conn
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._lock = threading.Lock()
        self._last = 0.0
        self.network_calls = 0

    # ---- cache -------------------------------------------------------------
    def _cache_get(self, key: str) -> Any | None:
        if self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT body, fetched_at, ttl FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        body, fetched_at, ttl = row
        if ttl is not None and time.time() - fetched_at > ttl:
            return None
        return json.loads(body)

    def _cache_put(self, key: str, url: str, body: Any, ttl: float | None) -> None:
        if self.conn is None:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO http_cache (key, url, body, fetched_at, ttl) VALUES (?,?,?,?,?)",
            (key, url, json.dumps(body), time.time(), ttl),
        )
        self.conn.commit()

    # ---- network -----------------------------------------------------------
    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get_json(
        self,
        url: str,
        params: dict | None = None,
        ttl: float | None = None,
        use_cache: bool = True,
        cache: bool = True,
        method: str = "GET",
    ) -> Any:
        key = cache_key(method + " " + url, params) if method != "GET" else cache_key(url, params)
        if use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return hit

        backoff = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            self.network_calls += 1
            try:
                if method == "GET":
                    r = self.session.get(url, params=params, timeout=self.timeout)
                else:
                    r = self.session.request(method, url, data=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                log.warning("network error %s (attempt %d): %s", url, attempt + 1, e)
                time.sleep(backoff)
                backoff *= 2
                continue

            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    raise HttpError(r.status_code, url, r.text)
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                log.warning("HTTP %d from %s, retrying in %.1fs", r.status_code, url, delay)
                time.sleep(delay)
                backoff *= 2
                continue
            if r.status_code >= 400:
                raise HttpError(r.status_code, url, r.text)

            body = r.json()
            if cache:
                self._cache_put(key, url, body, ttl)
            return body
        raise RuntimeError("unreachable")
