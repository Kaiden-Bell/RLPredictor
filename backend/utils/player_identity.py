"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Player identity resolution, alias parsing, platform ID matching, and deduplication logic.
- Usage: Imported across backend modules to resolve raw player names into unique canonical profiles.
"""

import difflib
import re
from typing import Optional, List, Dict

import utils.database as database


def normalize_player_name(name: str) -> str:
    """
    Description:
        Normalizes a display name to generate a potential alias key.
    Arguments:
        name: Raw display name string.
    Returns:
        String: Standardized normalized name string.
    """
    if not name: return ""
    
    n = name.lower()
    n = re.sub(r'\(.*?\)', '', n)
    n = n.strip()
    n = re.sub(r'^[a-z0-9]{2,4}[\.\-\s]+(?=[a-z0-9])', '', n)
    n = n.replace('_', '').replace('-', '').replace(' ', '')
    n = n.rstrip('.')
    
    if n.endswith('rl') and len(n) > 4: n = n[:-2]
        
    return n


def resolve_player_alias(alias: str, conn=None) -> Optional[str]:
    """
    Description:
        Resolves a player alias to its canonical player ID.
    Arguments:
        alias: Raw alias string.
        conn: Optional DB connection.
    Returns:
        Optional canonical player ID string or None.
    """
    own = conn is None
    if own: conn = database.get_connection()
    
    norm = normalize_player_name(alias)
    row = conn.execute("SELECT canonical_player_id FROM player_aliases WHERE alias = ?", (norm,)).fetchone()
    
    if own: conn.close()
    return row["canonical_player_id"] if row else None


def get_canonical_player(canonical_player_id: str, conn=None) -> Optional[Dict]:
    """
    Description:
        Fetches full canonical player profile details.
    Arguments:
        canonical_player_id: Unique player ID.
        conn: Optional DB connection.
    Returns:
        Optional profile details dictionary or None.
    """
    own = conn is None
    if own: conn = database.get_connection()
    
    row = conn.execute("SELECT * FROM canonical_players WHERE canonical_player_id = ?", (canonical_player_id,)).fetchone()
    if not row:
        if own: conn.close()
        return None
        
    player = dict(row)
    player["aliases"] = [r["alias"] for r in conn.execute("SELECT alias FROM player_aliases WHERE canonical_player_id = ?", (canonical_player_id,))]
    player["ids"] = [r["platform_id"] for r in conn.execute("SELECT platform_id FROM player_ids WHERE canonical_player_id = ?", (canonical_player_id,))]
    
    if own: conn.close()
    return player


def get_available_players(conn=None) -> List[Dict]:
    """
    Description:
        Pulls all active canonical players and their linked aliases.
    Arguments:
        conn: Optional DB connection.
    Returns:
        List of player detail dictionaries.
    """
    own = conn is None
    if own: conn = database.get_connection()
    
    players = []
    rows = conn.execute("SELECT * FROM canonical_players").fetchall()
    for r in rows:
        cid = r["canonical_player_id"]
        aliases = [a["alias"] for a in conn.execute("SELECT alias FROM player_aliases WHERE canonical_player_id = ?", (cid,))]
        players.append({
            "canonical_player_id": cid,
            "canonical_name": r["canonical_name"],
            "aliases": aliases
        })
        
    if own: conn.close()
    return players


def auto_detect_aliases(display_name: str, platform_player_id: str, platform: str, date_seen: str, conn=None) -> str:
    """
    Description:
        Processes a newly seen player name and platform ID, auto detecting or creating aliases.
    Arguments:
        display_name: Display name.
        platform_player_id: Platform ID.
        platform: Platform type (steam, epic, etc.).
        date_seen: Timestamp string.
        conn: Optional DB connection.
    Returns:
        String: Resolved canonical player ID.
    """
    own = conn is None
    if own: conn = database.get_connection()
    
    norm = normalize_player_name(display_name)
    full_platform_id = f"{platform}:{platform_player_id}" if platform and platform_player_id else None
    
    if full_platform_id:
        row = conn.execute("SELECT canonical_player_id FROM player_ids WHERE platform_id = ?", (full_platform_id,)).fetchone()
        if row:
            cid = row["canonical_player_id"]
            conn.execute("INSERT OR IGNORE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)", (norm, cid))
            if own:
                conn.commit()
                conn.close()
            return cid
            
    if not full_platform_id:
        row = conn.execute("SELECT canonical_player_id FROM player_aliases WHERE alias = ?", (norm,)).fetchone()
        if row:
            cid = row["canonical_player_id"]
            if own:
                conn.commit()
                conn.close()
            return cid
        
    cid = norm
    if full_platform_id:
        row = conn.execute("SELECT canonical_player_id FROM canonical_players WHERE canonical_player_id = ?", (cid,)).fetchone()
        if row: cid = f"{norm}_{platform_player_id}"
            
    canonical_name = display_name
    conn.execute("INSERT OR IGNORE INTO canonical_players (canonical_player_id, canonical_name, first_seen, last_seen) VALUES (?, ?, ?, ?)",
                 (cid, canonical_name, date_seen, date_seen))
    conn.execute("INSERT OR IGNORE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)", (norm, cid))
    
    if full_platform_id:
        conn.execute("INSERT OR IGNORE INTO player_ids (canonical_player_id, platform_id) VALUES (?, ?)", (cid, full_platform_id))
        
    if own:
        conn.commit()
        conn.close()
    return cid


def merge_players(primary_id: str, duplicate_id: str, conn=None):
    """
    Description:
        Merges a duplicate player profile ID into a primary player profile ID.
    Arguments:
        primary_id: Target canonical player ID.
        duplicate_id: Source canonical player ID to merge from.
        conn: Optional DB connection.
    Returns:
        None
    """
    own = conn is None
    if own: conn = database.get_connection()
    
    conn.execute("UPDATE player_aliases SET canonical_player_id = ? WHERE canonical_player_id = ?", (primary_id, duplicate_id))
    
    conn.execute("UPDATE OR IGNORE player_ids SET canonical_player_id = ? WHERE canonical_player_id = ?", (primary_id, duplicate_id))
    conn.execute("DELETE FROM player_ids WHERE canonical_player_id = ?", (duplicate_id,))
    
    conn.execute("UPDATE player_stats SET canonical_player_id = ?, canonical_name = (SELECT canonical_name FROM canonical_players WHERE canonical_player_id = ?) WHERE canonical_player_id = ?", (primary_id, primary_id, duplicate_id))
    
    conn.execute("DELETE FROM canonical_players WHERE canonical_player_id = ?", (duplicate_id,))
    
    if own:
        conn.commit()
        conn.close()


def find_possible_duplicates(conn=None) -> List[Dict]:
    """
    Description:
        Finds pairs of canonical players that could be duplicate profiles.
    Arguments:
        conn: Optional DB connection.
    Returns:
        List of duplicate candidate dictionaries.
    """
    own = conn is None
    if own: conn = database.get_connection()
    
    candidates = []
    players = conn.execute("SELECT canonical_player_id, canonical_name FROM canonical_players").fetchall()
    
    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            n1 = normalize_player_name(p1["canonical_name"])
            n2 = normalize_player_name(p2["canonical_name"])
            
            if not (n1 and n2) or len(n1) <= 3 or len(n2) <= 3: continue
                
            ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
            is_substring = n1 in n2 or n2 in n1
            
            if ratio > 0.85:
                candidates.append({
                    "canonical_player_id": p1["canonical_player_id"],
                    "possible_alias": p2["canonical_name"],
                    "duplicate_id": p2["canonical_player_id"],
                    "reason": f"High name similarity ({ratio:.2f})",
                    "confidence": "high",
                    "recommended_action": "merge"
                })
            elif is_substring:
                candidates.append({
                    "canonical_player_id": p1["canonical_player_id"],
                    "possible_alias": p2["canonical_name"],
                    "duplicate_id": p2["canonical_player_id"],
                    "reason": "One name is a substring of the other",
                    "confidence": "low",
                    "recommended_action": "review"
                })
                
    if own: conn.close()
    return candidates
