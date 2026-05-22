import requests
from bs4 import BeautifulSoup
import re
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RL-PredictorBot/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

def scrape_player_profile(url: str, session=None):
    sess = session or requests.Session()
    r = sess.get(url, headers=HEADERS, timeout=20)
    if r.status_code != 200:
        return None
        
    soup = BeautifulSoup(r.text, 'html.parser')
    
    player_data = {
        "url": url,
        "display_name": None,
        "alternate_ids": []
    }
    
    # 1. Get the display name from the page title
    title = soup.select_one('h1#firstHeading')
    if title:
        player_data["display_name"] = title.get_text(strip=True)
        
    # 2. Extract Alternate IDs from the infobox
    cells = soup.find_all('div', class_='infobox-description')
    for cell in cells:
        txt = cell.get_text(strip=True)
        if txt == "Alternate IDs:":
            alt_cell = cell.find_next_sibling('div')
            if alt_cell:
                strings = list(alt_cell.stripped_strings)
                for s in strings:
                    s = s.strip().strip(',')
                    if s:
                        player_data["alternate_ids"].append(s)
            break
            
    # Also find Steam profile link to get Steam ID
    steam_links = soup.select('a[href*="steamcommunity.com/id/"], a[href*="steamcommunity.com/profiles/"]')
    for a in steam_links:
        href = a.get('href', '')
        if 'steamcommunity.com/id/' in href:
            p = href.split('steamcommunity.com/id/')[-1].strip('/')
            if p: player_data["alternate_ids"].append(p)
        elif 'steamcommunity.com/profiles/' in href:
            p = href.split('steamcommunity.com/profiles/')[-1].strip('/')
            if p: player_data["alternate_ids"].append(f"steam:{p}")
            
    # deduplicate
    player_data["alternate_ids"] = list(set(player_data["alternate_ids"]))
            
    return player_data

def populate_players_from_liquipedia(limit=50, conn=None):
    from utils.database import get_connection
    from utils.player_identity import auto_detect_aliases
    import time
    
    sess = requests.Session()
    r = sess.get("https://liquipedia.net/rocketleague/Portal:Players", headers=HEADERS, timeout=20)
    if r.status_code != 200:
        print("Failed to fetch Portal:Players")
        return
        
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Extract links to player profiles
    # Usually in tables or list items. Top earnings is typically in a table.
    player_links = []
    for a in soup.select('.mw-parser-output a[href^="/rocketleague/"]'):
        href = a.get('href')
        title = a.get('title')
        if title and not any(x in href for x in [':', 'Portal', 'Rocket_League', 'Category']):
            url = f"https://liquipedia.net{href}"
            if url not in player_links:
                player_links.append(url)
                
    player_links = player_links[:limit]
    print(f"Found {len(player_links)} potential player profiles. Scraping...")
    
    own = conn is None
    if own: conn = get_connection()
    
    for i, url in enumerate(player_links):
        print(f"[{i+1}/{len(player_links)}] Scraping {url}...")
        try:
            data = scrape_player_profile(url, session=sess)
            time.sleep(1) # Be nice to Liquipedia
            
            if not data or not data["display_name"]:
                continue
                
            display_name = data["display_name"]
            
            # Ensure the canonical player exists
            from utils.player_identity import normalize_player_name
            cid = normalize_player_name(display_name)
            if not cid:
                continue
                
            conn.execute("INSERT OR IGNORE INTO canonical_players (canonical_player_id, canonical_name) VALUES (?, ?)", (cid, display_name))
            conn.execute("INSERT OR IGNORE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)", (cid, cid))
            
            for alt_id in data["alternate_ids"]:
                if alt_id.startswith('steam:') or alt_id.startswith('epic:'):
                    conn.execute("INSERT OR IGNORE INTO player_ids (canonical_player_id, platform_id) VALUES (?, ?)", (cid, alt_id))
                else:
                    alt_norm = normalize_player_name(alt_id)
                    if alt_norm:
                        conn.execute("INSERT OR IGNORE INTO player_aliases (alias, canonical_player_id) VALUES (?, ?)", (alt_norm, cid))
                        
            conn.commit()
                
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            
    if own: conn.close()

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Populating DB with top {limit} players from Liquipedia...")
    populate_players_from_liquipedia(limit=limit)
    print("Done!")
