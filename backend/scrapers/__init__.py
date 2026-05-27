"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Packages the web scrapers sub-module exports.
- Usage: Imported by main.py and stats.py to load Liquipedia and Ballchasing scrapers.
"""

from .playoff_scraper import scrape_playoffs
from .h2h_ballchasing import (
    Ballchasing,
    get_h2h_stats,
    load_player_id_map,
    resolve_ids,
)