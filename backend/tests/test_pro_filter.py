"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Test suite for verifying the Ballchasing API pro=true filter integration in stats.py.
- Usage: Run with pytest (pytest backend/tests/test_pro_filter.py -v) or directly (python3 backend/tests/test_pro_filter.py).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import pull_replays, ranked_activity


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


class MockBallchasing:
    """
    Description:
        Lightweight mock of the Ballchasing API wrapper for intercepting list_replays calls.
    """

    def __init__(self):
        """
        Description:
            Initializes the mock with tracking attributes.
        Arguments:
            None
        Returns:
            None
        """
        self.captured_params = {}
        self.call_count = 0
        self.delay = 0.0

    def list_replays(self, **params):
        """
        Description:
            Mock implementation that captures parameters for assertion.
        Arguments:
            params: Keyword arguments forwarded from stats.py calls.
        Returns:
            Dictionary mimicking the Ballchasing API response format.
        """
        self.captured_params = params
        self.call_count += 1
        return {"list": []}

    def get_replay(self, replay_id):
        """
        Description:
            Mock implementation returning an empty replay detail.
        Arguments:
            replay_id: Replay ID string.
        Returns:
            Empty replay detail dictionary.
        """
        return {
            "id": replay_id,
            "date": "2026-05-28T00:00:00Z",
            "blue": {"stats": {"core": {"goals": 0}}, "players": []},
            "orange": {"stats": {"core": {"goals": 0}}, "players": []},
        }


# ──────────────────────────────────────────────────────
# TEST 1: pull_replays defaults to pro=true
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 1: pull_replays injects pro='true' by default")
print("=" * 60)

bc = MockBallchasing()
pull_replays(bc, "steam:76561198141161044")

test(
    "'pro' key present in params",
    "pro" in bc.captured_params,
    f"params were: {bc.captured_params}",
)
test(
    "'pro' value is 'true'",
    bc.captured_params.get("pro") == "true",
    f"got pro={bc.captured_params.get('pro')}",
)
test(
    "'player-id' passed through",
    bc.captured_params.get("player-id") == "steam:76561198141161044",
    f"got player-id={bc.captured_params.get('player-id')}",
)
test(
    "Default playlist is 'private'",
    bc.captured_params.get("playlist") == "private",
    f"got playlist={bc.captured_params.get('playlist')}",
)


# ──────────────────────────────────────────────────────
# TEST 2: pull_replays with pro_only=False omits pro param
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 2: pull_replays with pro_only=False skips pro param")
print("=" * 60)

bc2 = MockBallchasing()
pull_replays(bc2, "steam:76561198141161044", pro_only=False)

test(
    "'pro' key NOT present when pro_only=False",
    "pro" not in bc2.captured_params,
    f"params were: {bc2.captured_params}",
)
test(
    "Other params still intact",
    bc2.captured_params.get("player-id") == "steam:76561198141161044",
    f"got player-id={bc2.captured_params.get('player-id')}",
)


# ──────────────────────────────────────────────────────
# TEST 3: pull_replays respects custom playlist + pro
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 3: pull_replays with custom playlist still includes pro")
print("=" * 60)

bc3 = MockBallchasing()
pull_replays(bc3, "steam:76561198141161044", playlist="ranked-doubles")

test(
    "Custom playlist forwarded",
    bc3.captured_params.get("playlist") == "ranked-doubles",
    f"got playlist={bc3.captured_params.get('playlist')}",
)
test(
    "'pro' still present with custom playlist",
    bc3.captured_params.get("pro") == "true",
    f"got pro={bc3.captured_params.get('pro')}",
)


# ──────────────────────────────────────────────────────
# TEST 4: pull_replays with no playlist
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 4: pull_replays with playlist=None omits playlist key")
print("=" * 60)

bc4 = MockBallchasing()
pull_replays(bc4, "steam:76561198141161044", playlist=None)

test(
    "'playlist' key omitted when None",
    "playlist" not in bc4.captured_params,
    f"params were: {bc4.captured_params}",
)
test(
    "'pro' still present even without playlist",
    bc4.captured_params.get("pro") == "true",
    f"got pro={bc4.captured_params.get('pro')}",
)


# ──────────────────────────────────────────────────────
# TEST 5: pull_replays count clamping + pro
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 5: pull_replays count clamped to 200 max with pro")
print("=" * 60)

bc5 = MockBallchasing()
pull_replays(bc5, "steam:76561198141161044", count=500)

test(
    "Count clamped to 200",
    bc5.captured_params.get("count") == 200,
    f"got count={bc5.captured_params.get('count')}",
)
test(
    "'pro' present alongside clamped count",
    bc5.captured_params.get("pro") == "true",
    f"got pro={bc5.captured_params.get('pro')}",
)


# ──────────────────────────────────────────────────────
# TEST 6: ranked_activity includes pro=true
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 6: ranked_activity injects pro='true' into params")
print("=" * 60)

bc6 = MockBallchasing()
logs = []
ranked_activity(bc6, ["steam:76561198141161044"], logs)

test(
    "ranked_activity calls list_replays",
    bc6.call_count >= 1,
    f"call_count={bc6.call_count}",
)
test(
    "'pro' present in ranked_activity call",
    bc6.captured_params.get("pro") == "true",
    f"got pro={bc6.captured_params.get('pro')}",
)
test(
    "Playlist is 'ranked-doubles'",
    bc6.captured_params.get("playlist") == "ranked-doubles",
    f"got playlist={bc6.captured_params.get('playlist')}",
)
test(
    "Count is 50 for ranked activity",
    bc6.captured_params.get("count") == 50,
    f"got count={bc6.captured_params.get('count')}",
)


# ──────────────────────────────────────────────────────
# TEST 7: pull_replays returns empty list gracefully
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 7: pull_replays returns empty list when API returns none")
print("=" * 60)

bc7 = MockBallchasing()
result = pull_replays(bc7, "steam:76561198141161044")

test(
    "Returns a list",
    isinstance(result, list),
    f"got type={type(result).__name__}",
)
test(
    "List is empty when mock returns no replays",
    len(result) == 0,
    f"got {len(result)} items",
)


# ──────────────────────────────────────────────────────
# TEST 8: ranked_activity handles empty replay list
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 8: ranked_activity returns zero metrics on empty results")
print("=" * 60)

bc8 = MockBallchasing()
logs8 = []
activity = ranked_activity(bc8, ["steam:76561198141161044"], logs8)

test(
    "Activity dict returned",
    isinstance(activity, dict),
    f"got type={type(activity).__name__}",
)
test(
    "Player key present in result",
    "steam:76561198141161044" in activity,
    f"keys={list(activity.keys())}",
)

player_info = activity.get("steam:76561198141161044", {})
test(
    "Games count is 0 (no replays)",
    player_info.get("games") == 0,
    f"got games={player_info.get('games')}",
)
test(
    "Win rate is 0.0 (no data)",
    player_info.get("win_rate") == 0.0,
    f"got win_rate={player_info.get('win_rate')}",
)


# ──────────────────────────────────────────────────────
# TEST 9: Multiple players in ranked_activity all get pro filter
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 9: Multiple players in ranked_activity all receive pro filter")
print("=" * 60)


class MultiCaptureBallchasing:
    """
    Description:
        Mock capturing all list_replays invocations for multi-player tests.
    """

    def __init__(self):
        """
        Description:
            Initializes multi-capture tracking lists.
        Arguments:
            None
        Returns:
            None
        """
        self.all_params = []
        self.delay = 0.0

    def list_replays(self, **params):
        """
        Description:
            Captures each call's parameters.
        Arguments:
            params: Query parameters.
        Returns:
            Empty response dictionary.
        """
        self.all_params.append(params)
        return {"list": []}

    def get_replay(self, replay_id):
        """
        Description:
            Returns an empty replay structure.
        Arguments:
            replay_id: Replay ID string.
        Returns:
            Empty replay detail dictionary.
        """
        return {"id": replay_id, "blue": {"players": []}, "orange": {"players": []}}


bc9 = MultiCaptureBallchasing()
logs9 = []
ranked_activity(
    bc9,
    ["steam:76561198141161044", "steam:76561198323843523"],
    logs9,
)

test(
    "Two list_replays calls made (one per player)",
    len(bc9.all_params) == 2,
    f"got {len(bc9.all_params)} calls",
)

all_have_pro = all(p.get("pro") == "true" for p in bc9.all_params)
test(
    "All calls include pro='true'",
    all_have_pro,
    f"params list: {bc9.all_params}",
)


# ──────────────────────────────────────────────────────
# TEST 10: Deduplication in ranked_activity with pro filter
# ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TEST 10: Duplicate player IDs are deduplicated before API calls")
print("=" * 60)

bc10 = MultiCaptureBallchasing()
logs10 = []
ranked_activity(
    bc10,
    ["steam:76561198141161044", "steam:76561198141161044"],
    logs10,
)

test(
    "Only one API call for duplicate IDs",
    len(bc10.all_params) == 1,
    f"got {len(bc10.all_params)} calls",
)
test(
    "Deduplicated call still has pro='true'",
    bc10.all_params[0].get("pro") == "true" if bc10.all_params else False,
    f"params: {bc10.all_params}",
)


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
