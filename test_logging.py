#!/usr/bin/env python3
"""
test_logging.py — Unit tests for the prediction logging pipeline.

Tests the logging, verification, and training ingestion WITHOUT
requiring selenium or other heavy scraper dependencies.
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# Fix Windows console encoding for emoji output
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

# ─── Temp directory for all test artifacts ───
TEST_DIR = tempfile.mkdtemp(prefix="rlpred_test_")
TEST_LOG = os.path.join(TEST_DIR, "prediction_log.csv")
TEST_CACHE = os.path.join(TEST_DIR, "test_cache.json")

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ PASS: {name}")
        passed += 1
    else:
        print(f"  ❌ FAIL: {name} — {detail}")
        failed += 1


# ═══════════════════════════════════════════════════════════════
# Inline the log_prediction function (same logic as chat.py)
# to avoid importing the full chat module + scrapers chain
# ═══════════════════════════════════════════════════════════════
import datetime as dt_module

def log_prediction(player, stat, threshold, is_over, num_games, nn_prob, features, log_path=TEST_LOG):
    """Direct copy of chat.py's log_prediction, parameterized for testing."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,player,stat,threshold,is_over,num_games,nn_prob,features\n")
            
    feat_str = ";".join([f"{x:.4f}" for x in features])
    timestamp = dt_module.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{player},{stat},{threshold},{is_over},{num_games or 1},{nn_prob:.4f},{feat_str}\n")


# ═══════════════════════════════════════════════════════════════
# TEST 1: log_prediction writes correct CSV
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 1: log_prediction writes correct CSV")
print("=" * 60)

fake_features = np.array([0.1, 0.5, 0.3, 1.0, 0.4, 0.6, 0.2, 0.15, 0.08, 0.35, 0.65, 0.55, 1.0], dtype=np.float32)

log_prediction(
    player="Zen", stat="Goals", threshold=2.5,
    is_over=True, num_games=3, nn_prob=0.7234,
    features=fake_features,
)

test("Log file created", os.path.exists(TEST_LOG))

df = pd.read_csv(TEST_LOG)
test("Has 1 row", len(df) == 1, f"got {len(df)}")
test("Player is 'Zen'", df.iloc[0]["player"] == "Zen", f"got {df.iloc[0]['player']}")
test("Stat is 'Goals'", df.iloc[0]["stat"] == "Goals")
test("Threshold is 2.5", df.iloc[0]["threshold"] == 2.5)
test("is_over is True", df.iloc[0]["is_over"] == True)
test("num_games is 3", df.iloc[0]["num_games"] == 3)
test("nn_prob ~0.7234", abs(df.iloc[0]["nn_prob"] - 0.7234) < 0.001, f"got {df.iloc[0]['nn_prob']}")

# Check feature serialization
feat_str = df.iloc[0]["features"]
parsed_feats = [float(x) for x in feat_str.split(";")]
test("Features has 13 values", len(parsed_feats) == 13, f"got {len(parsed_feats)}")
test("Feature[0] ~0.1", abs(parsed_feats[0] - 0.1) < 0.01)
test("Feature[3] = 1.0 (h2h_confident)", abs(parsed_feats[3] - 1.0) < 0.01)

# Log a second prediction
log_prediction(
    player="LJ", stat="Saves", threshold=4.0,
    is_over=False, num_games=None, nn_prob=0.3100,
    features=np.zeros(13, dtype=np.float32),
)

df2 = pd.read_csv(TEST_LOG)
test("Has 2 rows after second log", len(df2) == 2)
test("Second player is 'LJ'", df2.iloc[1]["player"] == "LJ")
test("num_games defaults to 1 when None", df2.iloc[1]["num_games"] == 1)

# Check timestamp is valid ISO format
ts = df2.iloc[0]["timestamp"]
try:
    datetime.fromisoformat(ts)
    test("Timestamp is valid ISO", True)
except Exception:
    test("Timestamp is valid ISO", False, f"got {ts}")


# ═══════════════════════════════════════════════════════════════
# TEST 2: verify_predictions matches outcomes
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 2: verify_predictions matches outcomes")
print("=" * 60)

# Build a fake cache with a replay that happened AFTER the logged predictions
now = datetime.now(timezone.utc)
future_date = (now + timedelta(hours=2)).isoformat()

fake_cache = {
    "replay_001": {
        "id": "replay_001",
        "date": future_date,
        "blue": {
            "players": [{
                "name": "Zen",
                "stats": {
                    "core": {"goals": 3, "shots": 5, "saves": 1, "score": 450},
                    "demo": {"inflicted": 2}
                }
            }]
        },
        "orange": {
            "players": [{
                "name": "LJ",
                "stats": {
                    "core": {"goals": 1, "shots": 3, "saves": 5, "score": 380},
                    "demo": {"inflicted": 0}
                }
            }]
        }
    }
}

with open(TEST_CACHE, "w") as f:
    json.dump(fake_cache, f)

# Run verify
from verify_predictions import verify_predictions
verify_predictions(log_path=TEST_LOG, cache_path=TEST_CACHE)

df_verified = pd.read_csv(TEST_LOG)
test("'label' column exists", "label" in df_verified.columns)

# Zen: Goals > 2.5, actual = 3 → label = 1.0
zen_row = df_verified[df_verified["player"] == "Zen"].iloc[0]
test("Zen Goals>2.5: label=1.0 (actual=3)", zen_row["label"] == 1.0, f"got {zen_row['label']}")

# LJ: Saves > 4.0, actual = 5 → label = 1.0
lj_row = df_verified[df_verified["player"] == "LJ"].iloc[0]
test("LJ Saves>4.0: label=1.0 (actual=5)", lj_row["label"] == 1.0, f"got {lj_row['label']}")

# Re-running verify should NOT re-label already-labeled rows
verify_predictions(log_path=TEST_LOG, cache_path=TEST_CACHE)
df_rerun = pd.read_csv(TEST_LOG)
test("Re-run doesn't duplicate labels", len(df_rerun) == 2, f"got {len(df_rerun)} rows")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Unmatched predictions stay unlabeled
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 3: Unmatched predictions stay unlabeled")
print("=" * 60)

# Log a prediction for a player not in the cache
log_prediction(
    player="Monkey Moon", stat="Demos", threshold=1.5,
    is_over=True, num_games=1, nn_prob=0.55,
    features=np.ones(13, dtype=np.float32) * 0.5,
)

verify_predictions(log_path=TEST_LOG, cache_path=TEST_CACHE)
df3 = pd.read_csv(TEST_LOG)
mm_row = df3[df3["player"] == "Monkey Moon"].iloc[0]
test("Unmatched prediction has NaN label", pd.isna(mm_row["label"]))


# ═══════════════════════════════════════════════════════════════
# TEST 4: train.py ingestion (feature parsing + merging)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST 4: train.py log ingestion (feature parsing)")
print("=" * 60)

df_log = pd.read_csv(TEST_LOG)
df_live = df_log.dropna(subset=["label"])

test("Verified rows available", len(df_live) > 0)

live_feats = []
live_labels = []
parse_error = None
try:
    for _, row in df_live.iterrows():
        f_vec = np.array([float(x) for x in str(row["features"]).split(";")], dtype=np.float32)
        live_feats.append(f_vec)
        live_labels.append(float(row["label"]))
except Exception as e:
    parse_error = str(e)

test("Feature parsing succeeds", parse_error is None, parse_error or "")
test("Parsed correct number of rows", len(live_feats) == len(df_live))

if live_feats:
    test("Each feature vec has 13 dims", all(len(f) == 13 for f in live_feats))
    
    # Simulate the vstack that train.py does
    base_features = np.zeros((5, 13), dtype=np.float32)
    base_labels = np.zeros(5, dtype=np.float32)
    
    merged_features = np.vstack([base_features, np.array(live_feats)])
    merged_labels = np.concatenate([base_labels, np.array(live_labels)])
    
    test("vstack works (shape correct)", merged_features.shape == (5 + len(live_feats), 13), f"got {merged_features.shape}")
    test("Labels concat works", len(merged_labels) == 5 + len(live_labels))


# ═══════════════════════════════════════════════════════════════
# CLEANUP & SUMMARY
# ═══════════════════════════════════════════════════════════════
shutil.rmtree(TEST_DIR, ignore_errors=True)

print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ALL TESTS PASSED!")
else:
    print("  Some tests failed. Review above.")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
