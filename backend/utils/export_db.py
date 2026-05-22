import os
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictor.db"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dumps"

def export_to_csv():
    """Exports key database tables to CSV format for easy visualization."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    
    # List of tables to export
    tables = [
        "canonical_players",
        "player_ids",
        "player_aliases",
        "player_stats",
        "tournaments",
        "matchups"
    ]
    
    print(f"Exporting tables to {OUTPUT_DIR}/...")
    for table in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            out_file = OUTPUT_DIR / f"{table}.csv"
            df.to_csv(out_file, index=False)
            print(f"  ✓ {table}.csv ({len(df)} rows)")
        except pd.errors.DatabaseError:
            print(f"  ✗ Failed to export {table} (table might not exist yet)")
            
    conn.close()

def export_sql_dump():
    """Creates a raw .sql dump file of the entire database."""
    if not DB_PATH.exists():
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = OUTPUT_DIR / "predictor_dump.sql"
    
    print(f"Generating full SQL dump at {out_file}...")
    conn = sqlite3.connect(str(DB_PATH))
    with open(out_file, 'w') as f:
        for line in conn.iterdump():
            f.write('%s\n' % line)
    conn.close()
    print("  ✓ Full SQL dump complete.")

if __name__ == "__main__":
    print("--- RLPredictor Database Exporter ---")
    export_to_csv()
    print("")
    export_sql_dump()
    print("Done!")
