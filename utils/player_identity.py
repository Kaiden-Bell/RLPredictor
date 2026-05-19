import re
from typing import Optional, List, Dict
from utils.database import get_connection

def normalize_player_name(name: str) -> str:
    """
    Normalizes a display name to generate a potential alias key.
    - lowercases
    - strips whitespace
    - removes common team prefixes (e.g. 'SSG.', 'NRG ')
    - removes spaces, dashes, underscores
    """
    if not name:
        return ""
    
    n = name.lower().strip()
    
    # Remove common team prefix patterns (2-4 letters followed by dot, space, or dash)
    n = re.sub(r'^[a-z0-9]{2,4}[\.\-\s]+', '', n)
    
    # Remove spaces, underscores, dashes, but KEEP periods, accents, etc.
    n = n.replace('_', '').replace('-', '').replace(' ', '')
    
    # Optional: strip trailing 'rl' if it's longer than 3 chars (e.g., mechrl -> mech, but not 'carl' -> 'ca')
    if n.endswith('rl') and len(n) > 4:
        n = n[:-2]
        
    return n

def resolve_player_alias(alias: str, conn=None) -> Optional[str]:
    """Returns canonical_player_id for a given alias, or None."""
    own = conn is None
    if own: conn = get_connection()
    
    norm = normalize_player_name(alias)
    row = conn.execute("SELECT canonical_player_id FROM player_aliases WHERE alias = ?", (norm,)).fetchone()
    
    if own: conn.close()
    return row["canonical_player_id"] if row else None

def get_canonical_player(canonical_player_id: str, conn=None) -> Optional[Dict]:
    """Fetch full details of a canonical player."""
    own = conn is None
    if own: conn = get_connection()
    
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
    """Return all canonical players with their aliases."""
    own = conn is None
    if own: conn = get_connection()
    
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
    Process a newly seen player from a replay.
    Returns the canonical_player_id.
    """
    own = conn is None
    if own: conn = get_connection()
    
    norm = normalize_player_name(display_name)
    full_platform_id = f"{platform}:{platform_player_id}" if platform and platform_player_id else None
    
    # 1. Try to match by platform ID
    if full_platform_id:
        row = conn.execute("SELECT canonical_player_id FROM player_ids WHERE platform_id = ?", (full_platform_id,)).fetchone()
        if row:
            cid = row["canonical_player_id"]
            # Add alias if new
            conn.execute("INSERT OR IGNORE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)", (norm, cid))
            if own: conn.commit(); conn.close()
            return cid
            
    # 2. Try to match by existing alias
    row = conn.execute("SELECT canonical_player_id FROM player_aliases WHERE alias = ?", (norm,)).fetchone()
    if row:
        cid = row["canonical_player_id"]
        # Add ID if new
        if full_platform_id:
            conn.execute("INSERT OR IGNORE INTO player_ids (canonical_player_id, platform_id) VALUES (?, ?)", (cid, full_platform_id))
        if own: conn.commit(); conn.close()
        return cid
        
    # 3. Create new canonical player
    cid = norm
    canonical_name = display_name
    conn.execute("INSERT OR IGNORE INTO canonical_players (canonical_player_id, canonical_name, first_seen, last_seen) VALUES (?, ?, ?, ?)",
                 (cid, canonical_name, date_seen, date_seen))
    conn.execute("INSERT OR IGNORE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)", (norm, cid))
    if full_platform_id:
        conn.execute("INSERT OR IGNORE INTO player_ids (canonical_player_id, platform_id) VALUES (?, ?)", (cid, full_platform_id))
        
    if own: conn.commit(); conn.close()
    return cid

def merge_players(primary_id: str, duplicate_id: str, conn=None):
    """Merge duplicate_id into primary_id."""
    own = conn is None
    if own: conn = get_connection()
    
    # 1. Move aliases
    conn.execute("UPDATE player_aliases SET canonical_player_id = ? WHERE canonical_player_id = ?", (primary_id, duplicate_id))
    
    # 2. Move platform IDs
    conn.execute("UPDATE OR IGNORE player_ids SET canonical_player_id = ? WHERE canonical_player_id = ?", (primary_id, duplicate_id))
    conn.execute("DELETE FROM player_ids WHERE canonical_player_id = ?", (duplicate_id,))
    
    # 3. Update player stats
    conn.execute("UPDATE player_stats SET canonical_player_id = ?, canonical_name = (SELECT canonical_name FROM canonical_players WHERE canonical_player_id = ?) WHERE canonical_player_id = ?", (primary_id, primary_id, duplicate_id))
    
    # 4. Remove duplicate canonical player
    conn.execute("DELETE FROM canonical_players WHERE canonical_player_id = ?", (duplicate_id,))
    
    if own: conn.commit(); conn.close()

def find_possible_duplicates(conn=None) -> List[Dict]:
    """Find players that might be the same person."""
    own = conn is None
    if own: conn = get_connection()
    
    candidates = []
    # Simplified version: Look for players with very similar canonical names
    # This can be expanded based on roster context in the future
    players = conn.execute("SELECT canonical_player_id, canonical_name FROM canonical_players").fetchall()
    
    for i, p1 in enumerate(players):
        for p2 in players[i+1:]:
            n1 = normalize_player_name(p1["canonical_name"])
            n2 = normalize_player_name(p2["canonical_name"])
            
            if n1 and n2 and (n1 in n2 or n2 in n1) and len(n1) > 3 and len(n2) > 3:
                candidates.append({
                    "canonical_player_id": p1["canonical_player_id"],
                    "possible_alias": p2["canonical_name"],
                    "duplicate_id": p2["canonical_player_id"],
                    "reason": "Normalized names are very similar or subset",
                    "confidence": "medium",
                    "recommended_action": "review"
                })
                
    if own: conn.close()
    return candidates
