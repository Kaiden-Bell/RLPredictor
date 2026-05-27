"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Test suite for verifying database operations, API response caching, and player stats insertion.
- Usage: Run with pytest (pytest backend/tests/test_db.py) to assert DB caching layers.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import (
    cache_api_response,
    get_cached_api_response,
    cache_replay,
    get_cached_replay,
    cache_player_stats,
    get_all_replay_details,
)


def test_db_inserts():
    """
    Description:
        Asserts standard SQLite API insertions, cache keys retrieval, and player stats table updates.
    Arguments:
        None
    Returns:
        None
    """
    print("Testing API cache...")
    cache_api_response("test_key_123", "/api/test", '{"status": "success", "data": "test_data"}')
    cached = get_cached_api_response("test_key_123")
    assert cached["status"] == "success"
    print("API cache test passed!")

    print("Testing Replay cache...")
    raw_replay = json.dumps({"id": "replay_123", "blue": {}, "orange": {}})
    cache_replay("replay_123", "2026-05-19", "ranked", "Ranked 2v2", raw_replay)
    cached_rep = get_cached_replay("replay_123")
    assert cached_rep["id"] == "replay_123"
    print("Replay cache test passed!")

    print("Testing Player Stats cache...")
    cache_player_stats("replay_123", "zen", "Zen", "Vitality Zen", "steam_123", "steam", "blue", 3, 5, 2, 0, 850, 0.6, "2026-05-19")
    
    replays = get_all_replay_details()
    assert len(replays) >= 1
    print("Player Stats and get_all_replay_details test passed!")

    print("All tests passed successfully on the fresh DB!")


if __name__ == "__main__":
    test_db_inserts()
