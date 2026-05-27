"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Unit tests for validating verification, log ingestion, and feature parsing.
- Usage: Run directly (python3 backend/tests/test_logging.py) to check training pipeline integration.
"""

import datetime as dt_module
import json
import os
import shutil
import sys
import tempfile

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

import numpy as np
import pandas as pd

import verify_predictions as vp


sys.stdout.reconfigure(encoding='utf-8')


TEST_DIR = tempfile.mkdtemp(prefix="rlpred_test_")
TEST_LOG = os.path.join(TEST_DIR, "prediction_log.csv")
TEST_CACHE = os.path.join(TEST_DIR, "test_cache.json")


CURRENT_TEST_CACHE = TEST_CACHE


def mock_extract_player_game_stats_from_db():
    if not os.path.exists(CURRENT_TEST_CACHE): return pd.DataFrame()
    with open(CURRENT_TEST_CACHE, "r") as f:
        cache = json.load(f)
    rows = []
    for rid, data in cache.items():
        date_val = data.get("date", "")
        for side in ("blue", "orange"):
            team = data.get(side) or {}
            for pl in team.get("players", []) or []:
                name = pl.get("name") or (pl.get("player") or {}).get("name")
                stats = pl.get("stats") or {}
                core = stats.get("core") or {}
                demo = stats.get("demo") or {}
                rows.append({
                    "canonical_player_id": name.lower(),
                    "Player": name,
                    "display_name_seen": name,
                    "Goals": core.get("goals", 0),
                    "Shots": core.get("shots", 0),
                    "Saves": core.get("saves", 0),
                    "Demos": demo.get("inflicted", 0),
                    "Score": core.get("score", 0),
                    "replay_date": date_val,
                    "replay_id": rid,
                })
    return pd.DataFrame(rows)


def mock_resolve_player_to_canonical(player_name):
    return player_name.lower(), player_name


vp.extract_player_game_stats_from_db = mock_extract_player_game_stats_from_db
vp.resolve_player_to_canonical = mock_resolve_player_to_canonical


passed = 0
failed = 0


def test(name, condition, detail=""):
    """
    Description:
        Asserts a test case condition, updating the global passed/failed counters.
    Arguments:
        name: Name of the test case.
        condition: Boolean evaluation of success.
        detail: Diagnostic failure detail string.
    Returns:
        None
    """
    global passed, failed
    if condition:
        print(f"  [OK] PASS: {name}")
        passed += 1
    else:
        print(f"  [FAIL] FAIL: {name} — {detail}")
        failed += 1


def log_prediction(player, stat, threshold, is_over, num_games, nn_prob, features, team1="", team2="", log_path=TEST_LOG):
    """
    Description:
        Simulated copy of log_prediction for local verification tests.
    Arguments:
        player: Canonical player name string.
        stat: Target prediction statistic.
        threshold: Over/under target value.
        is_over: True if over, False if under.
        num_games: Match window count.
        nn_prob: Predictor neural network probability.
        features: 13-dimensional feature array.
        team1: Team 1 name.
        team2: Team 2 name.
        log_path: Target log filepath.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("timestamp,player,stat,threshold,is_over,num_games,nn_prob,features,team1,team2\n")
            
    feat_str = ";".join([f"{x:.4f}" for x in features])
    timestamp = dt_module.datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{player},{stat},{threshold},{is_over},{num_games or 1},{nn_prob:.4f},{feat_str},{team1},{team2}\n")


print("\n" + "=" * 60)
print("TEST 1: log_prediction writes correct CSV")
print("=" * 60)

fake_features = np.array([0.1, 0.5, 0.3, 1.0, 0.4, 0.6, 0.2, 0.15, 0.08, 0.35, 0.65, 0.55, 1.0], dtype=np.float32)

log_prediction(
    player="Zen", stat="Goals", threshold=2.5,
    is_over=True, num_games=3, nn_prob=0.7234,
    features=fake_features, team1="Vitality", team2="Karmine Corp",
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
test("team1 is 'Vitality'", df.iloc[0]["team1"] == "Vitality", f"got {df.iloc[0].get('team1')}")
test("team2 is 'Karmine Corp'", df.iloc[0]["team2"] == "Karmine Corp", f"got {df.iloc[0].get('team2')}")

feat_str = df.iloc[0]["features"]
parsed_feats = [float(x) for x in feat_str.split(";")]
test("Features has 13 values", len(parsed_feats) == 13, f"got {len(parsed_feats)}")
test("Feature[0] ~0.1", abs(parsed_feats[0] - 0.1) < 0.01)
test("Feature[3] = 1.0 (h2h_confident)", abs(parsed_feats[3] - 1.0) < 0.01)

log_prediction(
    player="LJ", stat="Saves", threshold=4.0,
    is_over=False, num_games=None, nn_prob=0.3100,
    features=np.zeros(13, dtype=np.float32),
)

df2 = pd.read_csv(TEST_LOG)
test("Has 2 rows after second log", len(df2) == 2)
test("Second player is 'LJ'", df2.iloc[1]["player"] == "LJ")
test("num_games defaults to 1 when None", df2.iloc[1]["num_games"] == 1)

ts = df2.iloc[0]["timestamp"]
try:
    dt_module.datetime.fromisoformat(ts)
    test("Timestamp is valid ISO", True)
except Exception:
    test("Timestamp is valid ISO", False, f"got {ts}")


print("\n" + "=" * 60)
print("TEST 2: verify_predictions — single-game verification")
print("=" * 60)

now = dt_module.datetime.now(dt_module.timezone.utc)
future_date = (now + dt_module.timedelta(hours=2)).isoformat()

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

CURRENT_TEST_CACHE = TEST_CACHE
vp.verify_predictions(log_path=TEST_LOG, cache_path=TEST_CACHE)

df_verified = pd.read_csv(TEST_LOG)
test("'label' column exists", "label" in df_verified.columns)

zen_row = df_verified[df_verified["player"] == "Zen"].iloc[0]
test("Zen skipped (only 1/3 games found)", pd.isna(zen_row["label"]), f"got {zen_row.get('label')}")

lj_row = df_verified[df_verified["player"] == "LJ"].iloc[0]
test("LJ Saves>4.0: label=1.0 (actual=5)", lj_row["label"] == 1.0, f"got {lj_row['label']}")


print("\n" + "=" * 60)
print("TEST 3: Multi-game verification")
print("=" * 60)

TEST_LOG_MG = os.path.join(TEST_DIR, "prediction_log_mg.csv")
TEST_CACHE_MG = os.path.join(TEST_DIR, "test_cache_mg.json")

log_prediction(
    player="Andy", stat="Saves", threshold=2.0,
    is_over=True, num_games=3, nn_prob=0.22,
    features=np.zeros(13, dtype=np.float32),
    team1="Shopify", team2="Fellas",
    log_path=TEST_LOG_MG,
)

base_time = now + dt_module.timedelta(hours=1)
multi_cache = {}
for g in range(3):
    game_date = (base_time + dt_module.timedelta(minutes=g*15)).isoformat()
    multi_cache[f"replay_g{g}"] = {
        "id": f"replay_g{g}",
        "date": game_date,
        "blue": {
            "players": [{
                "name": "Andy",
                "stats": {
                    "core": {"goals": 1, "shots": 3, "saves": 3 + g, "score": 300},
                    "demo": {"inflicted": 1}
                }
            }]
        },
        "orange": {
            "players": [{
                "name": "Someone",
                "stats": {
                    "core": {"goals": 0, "shots": 2, "saves": 1, "score": 200},
                    "demo": {"inflicted": 0}
                }
            }]
        }
    }

with open(TEST_CACHE_MG, "w") as f:
    json.dump(multi_cache, f)

CURRENT_TEST_CACHE = TEST_CACHE_MG
vp.verify_predictions(log_path=TEST_LOG_MG, cache_path=TEST_CACHE_MG)

df_mg = pd.read_csv(TEST_LOG_MG)
andy_row = df_mg.iloc[0]
test("Andy multi-game saves labeled", pd.notna(andy_row["label"]), f"label is {andy_row.get('label')}")
test("Andy Saves>2.0 (3g): label=1.0 (total=12)", andy_row["label"] == 1.0, f"got {andy_row['label']}")


print("\n" + "=" * 60)
print("TEST 4: Unmatched predictions stay unlabeled")
print("=" * 60)

log_prediction(
    player="Monkey Moon", stat="Demos", threshold=1.5,
    is_over=True, num_games=1, nn_prob=0.55,
    features=np.ones(13, dtype=np.float32) * 0.5,
)

CURRENT_TEST_CACHE = TEST_CACHE
vp.verify_predictions(log_path=TEST_LOG, cache_path=TEST_CACHE)
df3 = pd.read_csv(TEST_LOG)
mm_row = df3[df3["player"] == "Monkey Moon"].iloc[0]
test("Unmatched prediction has NaN label", pd.isna(mm_row["label"]))

CURRENT_TEST_CACHE = TEST_CACHE
vp.verify_predictions(log_path=TEST_LOG, cache_path=TEST_CACHE)
df_rerun = pd.read_csv(TEST_LOG)
test("Re-run doesn't duplicate labels", len(df_rerun) == 3, f"got {len(df_rerun)} rows")


print("\n" + "=" * 60)
print("TEST 5: train.py log ingestion (feature parsing)")
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
    
    base_features = np.zeros((5, 13), dtype=np.float32)
    base_labels = np.zeros(5, dtype=np.float32)
    
    merged_features = np.vstack([base_features, np.array(live_feats)])
    merged_labels = np.concatenate([base_labels, np.array(live_labels)])
    
    test("vstack works (shape correct)", merged_features.shape == (5 + len(live_feats), 13), f"got {merged_features.shape}")
    test("Labels concat works", len(merged_labels) == 5 + len(live_labels))


print("\n" + "=" * 60)
print("TEST 6: Team columns preserved through pipeline")
print("=" * 60)

df_mg_final = pd.read_csv(TEST_LOG_MG)
test("team1 column present", "team1" in df_mg_final.columns)
test("team2 column present", "team2" in df_mg_final.columns)
test("team1 value preserved", df_mg_final.iloc[0]["team1"] == "Shopify", f"got {df_mg_final.iloc[0].get('team1')}")
test("team2 value preserved", df_mg_final.iloc[0]["team2"] == "Fellas", f"got {df_mg_final.iloc[0].get('team2')}")


shutil.rmtree(TEST_DIR, ignore_errors=True)

print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0: print("  ALL TESTS PASSED!")
else: print("  Some tests failed. Review above.")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
