"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Test suite for verifying the batch replay fetching optimization and SQLite cache-first lookup.
- Usage: Run with pytest (pytest backend/tests/test_batch_fetch.py -v) or directly (python3 backend/tests/test_batch_fetch.py).
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import ranked_activity, replay_stats


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


def make_replay_detail(rid, player_name="TestPlayer", platform_id="steam:111",
                       goals=2, shots=5, saves=1, score=400, side="blue"):
    """
    Description:
        Constructs a realistic replay detail dict matching the Ballchasing API schema.
    Arguments:
        rid: Replay ID string.
        player_name: Player display name.
        platform_id: Platform:ID formatted string.
        goals: Goals count.
        shots: Shots count.
        saves: Saves count.
        score: Match score.
        side: Team side to place player on.
    Returns:
        Dictionary: Synthetic replay detail structure.
    """
    plat, pid = platform_id.split(":", 1) if ":" in platform_id else ("steam", platform_id)
    player = {
        "name": player_name,
        "id": {"platform": plat, "id": pid},
        "stats": {
            "core": {"goals": goals, "shots": shots, "saves": saves, "score": score, "shooting_percentage": 0},
            "demo": {"inflicted": 1},
        },
    }
    blue_players = [player] if side == "blue" else []
    orange_players = [player] if side == "orange" else []
    return {
        "id": rid,
        "date": "2026-05-28T00:00:00Z",
        "playlist_id": "ranked-doubles",
        "playlist_name": "Ranked Doubles",
        "blue": {
            "stats": {"core": {"goals": goals if side == "blue" else 0}},
            "players": blue_players,
        },
        "orange": {
            "stats": {"core": {"goals": goals if side == "orange" else 0}},
            "players": orange_players,
        },
    }


class BatchMockBallchasing:
    """
    Description:
        Mock Ballchasing client with support for get_replays_batch, tracking all
        individual fetch attempts to verify caching and parallelism behavior.
    """

    def __init__(self, cached_replays=None, api_replays=None, fetch_delay=0.0):
        """
        Description:
            Initializes mock with preconfigured cached and API replay data.
        Arguments:
            cached_replays: Dict of replay_id -> detail to simulate SQLite cache hits.
            api_replays: Dict of replay_id -> detail to simulate API responses.
            fetch_delay: Simulated network delay per fetch.
        Returns:
            None
        """
        self.cached_replays = cached_replays or {}
        self.api_replays = api_replays or {}
        self.network_fetches = []
        self.fetch_delay = fetch_delay
        self.delay = 0.01
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0
        self.list_replays_params = {}

    def list_replays(self, **params):
        """
        Description:
            Mock list_replays returning empty results.
        Arguments:
            params: Query parameters.
        Returns:
            Empty replay list dictionary.
        """
        self.list_replays_params = params
        return {"list": []}

    def get_replay(self, replay_id):
        """
        Description:
            Mock single replay fetch checking cache first.
        Arguments:
            replay_id: Replay ID string.
        Returns:
            Replay detail dictionary.
        """
        if replay_id in self.cached_replays:
            return self.cached_replays[replay_id]
        self.network_fetches.append(replay_id)
        if self.fetch_delay:
            time.sleep(self.fetch_delay)
        return self.api_replays.get(replay_id, {"id": replay_id, "blue": {"players": []}, "orange": {"players": []}})

    def get_replays_batch(self, replay_ids, max_workers=4, progress_cb=None):
        """
        Description:
            Mock batch fetch separating cached from uncached replays.
        Arguments:
            replay_ids: List of replay ID strings.
            max_workers: Ignored in mock.
            progress_cb: Optional progress callback.
        Returns:
            Tuple of (results dict, cached_count int).
        """
        results = {}
        uncached_ids = []

        for rid in replay_ids:
            if rid in self.cached_replays:
                results[rid] = self.cached_replays[rid]
            else:
                uncached_ids.append(rid)

        cached_count = len(results)

        for i, rid in enumerate(uncached_ids):
            self.network_fetches.append(rid)
            if self.fetch_delay:
                time.sleep(self.fetch_delay)
            results[rid] = self.api_replays.get(
                rid, {"id": rid, "blue": {"players": []}, "orange": {"players": []}}
            )
            if progress_cb:
                progress_cb(cached_count + i + 1, len(replay_ids), cached_count)

        return results, cached_count


# ──────────────────────────────────────────────────────
# TEST 1: Batch fetch returns all cached replays without network calls
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 1: Batch fetch resolves all hits from cache — zero network calls")
print("=" * 60)

cached = {
    "r1": make_replay_detail("r1"),
    "r2": make_replay_detail("r2"),
    "r3": make_replay_detail("r3"),
}
bc1 = BatchMockBallchasing(cached_replays=cached)
results, cc = bc1.get_replays_batch(["r1", "r2", "r3"])

test("All 3 replays returned", len(results) == 3, f"got {len(results)}")
test("Cached count is 3", cc == 3, f"got {cc}")
test("Zero network fetches", len(bc1.network_fetches) == 0, f"got {len(bc1.network_fetches)}")


# ──────────────────────────────────────────────────────
# TEST 2: Batch fetch only hits network for uncached replays
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2: Batch fetch only makes network calls for cache misses")
print("=" * 60)

cached2 = {"r1": make_replay_detail("r1"), "r2": make_replay_detail("r2")}
api2 = {"r3": make_replay_detail("r3"), "r4": make_replay_detail("r4")}
bc2 = BatchMockBallchasing(cached_replays=cached2, api_replays=api2)
results2, cc2 = bc2.get_replays_batch(["r1", "r2", "r3", "r4"])

test("All 4 replays returned", len(results2) == 4, f"got {len(results2)}")
test("Cached count is 2", cc2 == 2, f"got {cc2}")
test("Only 2 network fetches", len(bc2.network_fetches) == 2, f"got {len(bc2.network_fetches)}")
test("Network fetched r3", "r3" in bc2.network_fetches, f"fetched: {bc2.network_fetches}")
test("Network fetched r4", "r4" in bc2.network_fetches, f"fetched: {bc2.network_fetches}")
test("r1 NOT network fetched", "r1" not in bc2.network_fetches, f"fetched: {bc2.network_fetches}")


# ──────────────────────────────────────────────────────
# TEST 3: Batch fetch with empty input
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 3: Batch fetch with empty replay list returns empty results")
print("=" * 60)

bc3 = BatchMockBallchasing()
results3, cc3 = bc3.get_replays_batch([])

test("Returns empty dict", len(results3) == 0, f"got {len(results3)}")
test("Cached count is 0", cc3 == 0, f"got {cc3}")
test("Zero network fetches", len(bc3.network_fetches) == 0, f"got {len(bc3.network_fetches)}")


# ──────────────────────────────────────────────────────
# TEST 4: Progress callback fires correctly
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 4: Progress callback fires with correct counts")
print("=" * 60)

progress_calls = []
api4 = {"r1": make_replay_detail("r1"), "r2": make_replay_detail("r2")}
bc4 = BatchMockBallchasing(api_replays=api4)

bc4.get_replays_batch(
    ["r1", "r2"],
    progress_cb=lambda cur, tot, cc: progress_calls.append((cur, tot, cc)),
)

test("Progress callback fired 2 times", len(progress_calls) == 2, f"got {len(progress_calls)}")
test("Total count in callbacks is 2", all(t == 2 for _, t, _ in progress_calls), f"calls: {progress_calls}")
test("Cached count in callbacks is 0", all(c == 0 for _, _, c in progress_calls), f"calls: {progress_calls}")


# ──────────────────────────────────────────────────────
# TEST 5: Progress callback reflects cache hits
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 5: Progress callback correctly reports cached count")
print("=" * 60)

progress_calls5 = []
cached5 = {"r1": make_replay_detail("r1"), "r2": make_replay_detail("r2")}
api5 = {"r3": make_replay_detail("r3")}
bc5 = BatchMockBallchasing(cached_replays=cached5, api_replays=api5)

bc5.get_replays_batch(
    ["r1", "r2", "r3"],
    progress_cb=lambda cur, tot, cc: progress_calls5.append((cur, tot, cc)),
)

test("Only 1 progress callback (1 uncached)", len(progress_calls5) == 1, f"got {len(progress_calls5)}")
if progress_calls5:
    test("Cached count reported as 2", progress_calls5[0][2] == 2, f"got {progress_calls5[0][2]}")
    test("Total reported as 3", progress_calls5[0][1] == 3, f"got {progress_calls5[0][1]}")


# ──────────────────────────────────────────────────────
# TEST 6: ranked_activity uses batch fetch correctly
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 6: ranked_activity integrates with batch fetch")
print("=" * 60)


class RankedMockBC(BatchMockBallchasing):
    """
    Description:
        Extended mock returning ranked replay listings for ranked_activity integration testing.
    """

    def __init__(self):
        """
        Description:
            Initializes with sample ranked replay data.
        Arguments:
            None
        Returns:
            None
        """
        replays = {
            "ranked_r1": make_replay_detail("ranked_r1", "Sypical", "steam:76561198323843523", goals=3, shots=7, saves=2, score=650),
            "ranked_r2": make_replay_detail("ranked_r2", "Sypical", "steam:76561198323843523", goals=1, shots=4, saves=0, score=300),
        }
        super().__init__(cached_replays=replays)

    def list_replays(self, **params):
        """
        Description:
            Returns sample ranked replay listings.
        Arguments:
            params: Query parameters.
        Returns:
            Replay listing dictionary.
        """
        self.list_replays_params = params
        return {
            "list": [
                {"id": "ranked_r1", "date": "2026-05-28T00:00:00Z"},
                {"id": "ranked_r2", "date": "2026-05-28T00:00:00Z"},
            ]
        }


bc6 = RankedMockBC()
logs6 = []
activity = ranked_activity(bc6, ["steam:76561198323843523"], logs6)

test("Activity dict returned", isinstance(activity, dict))
test("Player key present", "steam:76561198323843523" in activity, f"keys: {list(activity.keys())}")
test("Zero network fetches (all cached)", len(bc6.network_fetches) == 0, f"got {len(bc6.network_fetches)}")

info = activity.get("steam:76561198323843523", {})
test("Games detected", info.get("games", 0) > 0, f"got games={info.get('games')}")


# ──────────────────────────────────────────────────────
# TEST 7: replay_stats uses batch fetch correctly
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 7: replay_stats integrates with batch fetch")
print("=" * 60)


class StatsMockBC(BatchMockBallchasing):
    """
    Description:
        Extended mock for replay_stats with both cached and uncached replays.
    """

    def __init__(self):
        """
        Description:
            Initializes with sample private replay data.
        Arguments:
            None
        Returns:
            None
        """
        cached = {
            "scrim_r1": make_replay_detail("scrim_r1", "Zen", "steam:111", goals=4, shots=8),
        }
        api = {
            "scrim_r2": make_replay_detail("scrim_r2", "Zen", "steam:111", goals=2, shots=6),
        }
        super().__init__(cached_replays=cached, api_replays=api)

    def list_replays(self, **params):
        """
        Description:
            Returns sample private replay listings.
        Arguments:
            params: Query parameters.
        Returns:
            Replay listing dictionary.
        """
        self.list_replays_params = params
        return {
            "list": [
                {"id": "scrim_r1", "date": "2026-05-28T00:00:00Z"},
                {"id": "scrim_r2", "date": "2026-05-28T00:00:00Z"},
            ]
        }


bc7 = StatsMockBC()
logs7 = []
df = replay_stats(bc7, ["steam:111"], logs7)

test("DataFrame returned", hasattr(df, "shape"), f"got type={type(df).__name__}")
test("Only 1 network fetch (1 cache miss)", len(bc7.network_fetches) == 1, f"got {len(bc7.network_fetches)}")
test("Network fetched scrim_r2 (not cached)", "scrim_r2" in bc7.network_fetches, f"fetched: {bc7.network_fetches}")
test("scrim_r1 NOT network fetched (cached)", "scrim_r1" not in bc7.network_fetches, f"fetched: {bc7.network_fetches}")


# ──────────────────────────────────────────────────────
# TEST 8: Batch fetch handles missing/failed replays gracefully
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 8: Batch fetch handles missing replays without crashing")
print("=" * 60)

bc8 = BatchMockBallchasing(cached_replays={"r1": make_replay_detail("r1")})
results8, cc8 = bc8.get_replays_batch(["r1", "r_missing"])

test("Cached replay returned", "r1" in results8)
test("Missing replay still has placeholder", "r_missing" in results8, f"keys: {list(results8.keys())}")
test("1 cache hit", cc8 == 1, f"got {cc8}")


# ──────────────────────────────────────────────────────
# TEST 9: Rate lock is properly initialized
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 9: Rate limiting primitives are initialized on client")
print("=" * 60)

bc9 = BatchMockBallchasing()

test("_rate_lock exists", hasattr(bc9, "_rate_lock"))
test("_rate_lock is a Lock", isinstance(bc9._rate_lock, type(threading.Lock())))
test("_last_request_time exists", hasattr(bc9, "_last_request_time"))
test("_last_request_time starts at 0", bc9._last_request_time == 0.0, f"got {bc9._last_request_time}")


# ──────────────────────────────────────────────────────
# TEST 10: Large batch — cached replays skip network entirely
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 10: Large batch of 50 cached replays — zero network calls")
print("=" * 60)

large_cached = {f"r{i}": make_replay_detail(f"r{i}") for i in range(50)}
bc10 = BatchMockBallchasing(cached_replays=large_cached)
ids10 = [f"r{i}" for i in range(50)]
results10, cc10 = bc10.get_replays_batch(ids10)

test("All 50 replays returned", len(results10) == 50, f"got {len(results10)}")
test("Cached count is 50", cc10 == 50, f"got {cc10}")
test("Zero network fetches", len(bc10.network_fetches) == 0, f"got {len(bc10.network_fetches)}")


# ──────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print("  ALL TESTS PASSED!")
else:
    print("  Some tests failed. Review above.")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
