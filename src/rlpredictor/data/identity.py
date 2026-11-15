"""Player identity: match Ballchasing players to Liquipedia roster players.

`normalize_player_name` is carried over from the original backend/utils/player_identity.py.
"""

from __future__ import annotations

import difflib
import re
import sqlite3


def normalize_player_name(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"\(.*?\)", "", n)
    n = n.strip()
    n = re.sub(r"^[a-z0-9]{2,4}[\.\-\s]+(?=[a-z0-9])", "", n)  # team tag prefix, e.g. "KC Vatira"
    n = n.replace("_", "").replace("-", "").replace(" ", "").replace(".", "")
    if n.endswith("rl") and len(n) > 4:
        n = n[:-2]
    return n


def match_player(conn: sqlite3.Connection, platform_id: str | None, bc_name: str,
                 candidates: dict[str, str]) -> str | None:
    """Return the Liquipedia player id for a Ballchasing player, or None.

    candidates: {player_id: display name} for the players listed on the two teams.
    Order: known platform id -> exact normalized name -> close fuzzy match (>= 0.8).
    """
    if platform_id:
        row = conn.execute("SELECT player_id FROM player_platform_ids WHERE platform_id = ?",
                           (platform_id,)).fetchone()
        if row and row[0] in candidates:
            return row[0]
    target = normalize_player_name(bc_name)
    if not target:
        return None
    norm = {pid: normalize_player_name(name) for pid, name in candidates.items()}
    exact = [pid for pid, n in norm.items() if n == target]
    if len(exact) == 1:
        return exact[0]
    best, best_r = None, 0.0
    for pid, n in norm.items():
        r = difflib.SequenceMatcher(None, target, n).ratio()
        if (target in n or n in target) and min(len(n), len(target)) >= 3:
            r = max(r, 0.85)
        if r > best_r:
            best, best_r = pid, r
    return best if best_r >= 0.8 else None


def remember(conn: sqlite3.Connection, platform_id: str | None, player_id: str, bc_name: str) -> None:
    if platform_id:
        conn.execute(
            "INSERT OR IGNORE INTO player_platform_ids (platform_id, player_id, bc_name) VALUES (?,?,?)",
            (platform_id, player_id, bc_name))
