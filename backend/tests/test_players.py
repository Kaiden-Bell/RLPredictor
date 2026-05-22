import os
import sys

# Add the root directory to sys.path to allow importing utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_connection
from utils.player_identity import get_available_players, find_possible_duplicates, merge_players

def main():
    conn = get_connection()
    players = get_available_players(conn)
    print(f"Total players found: {len(players)}")
    
    print("\nFinding possible duplicates to map together...")
    duplicates = find_possible_duplicates(conn)
    print(f"Found {len(duplicates)} possible duplicates.")
    
    if duplicates:
        print("Please review the following candidates:")
        for dup in duplicates:
            print(f"  - [{dup['confidence'].upper()}] {dup['duplicate_id']} (Alias: {dup['possible_alias']}) -> {dup['canonical_player_id']}")
            print(f"    Reason: {dup['reason']} | Action: {dup['recommended_action']}")
            
    conn.close()

if __name__ == "__main__":
    main()
