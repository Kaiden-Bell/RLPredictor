"""Synthetic league for model/API tests: teams with hidden strengths play series."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from rlpredictor.data import db

STRENGTH = {"Alpha": 2.0, "Bravo": 1.0, "Charlie": 0.0, "Delta": -1.0, "Echo": -2.0, "Foxtrot": 0.5}
REGION = {"Alpha": "Europe", "Bravo": "Europe", "Charlie": "Europe",
          "Delta": "North America", "Echo": "North America", "Foxtrot": "North America"}


def seed_league(conn, n_events: int = 24, seed: int = 7) -> None:
    rng = random.Random(seed)
    teams = {}
    for name in STRENGTH:
        teams[name] = db.get_or_create_team(conn, name.lower(), name, REGION[name])
        for i in range(1, 4):
            conn.execute("INSERT OR IGNORE INTO players (id, name) VALUES (?, ?)",
                         (f"{name.lower()}{i}", f"{name}{i}"))
    start = date(2024, 1, 6)
    for e in range(n_events):
        root = f"Event {e}"
        d = start + timedelta(days=14 * e)
        conn.execute("INSERT INTO pages (title, event_root, has_infobox, tier, start_date) "
                     "VALUES (?,?,1,2,?)", (root, root, str(d)))
        for name, tid in teams.items():
            # Bravo swaps a player half-way through.
            ids = [f"{name.lower()}{i}" for i in range(1, 4)]
            if name == "Bravo" and e >= n_events // 2:
                ids[2] = "newguy"
                conn.execute("INSERT OR IGNORE INTO players (id, name) VALUES ('newguy','NewGuy')")
            for slot, pid in enumerate(ids, 1):
                conn.execute("INSERT INTO event_rosters VALUES (?,?,?,?)", (root, tid, pid, slot))
        names = list(teams)
        k = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if rng.random() < 0.5:
                    a, b = b, a
                p = 1 / (1 + 2.718 ** (-(STRENGTH[a] - STRENGTH[b]) * 0.6))
                w1 = w2 = 0
                seq = []
                while w1 < 3 and w2 < 3:
                    if rng.random() < p:
                        w1 += 1
                        seq.append(1)
                    else:
                        w2 += 1
                        seq.append(2)
                sid = f"{root}#m{k}"
                k += 1
                conn.execute(
                    "INSERT INTO series (id, page, event_root, date, tier, region, best_of, team1_id, "
                    "team2_id, score1, score2, finished, walkover) VALUES (?,?,?,?,?,?,?,?,?,?,?,1,0)",
                    (sid, root, root, str(d), 2, "Europe", 5, teams[a], teams[b], w1, w2))
                for n, w in enumerate(seq, 1):
                    g1, g2 = (3, 1) if w == 1 else (1, 3)
                    conn.execute("INSERT INTO games VALUES (?,?,?,?,?,0)", (sid, n, g1, g2, w))
    # one upcoming series
    conn.execute(
        "INSERT INTO series (id, page, event_root, date, tier, best_of, team1_id, team2_id, finished) "
        "VALUES ('Upcoming Cup#m0','Upcoming Cup','Upcoming Cup',?,1,7,?,?,0)",
        (str(start + timedelta(days=14 * n_events)), teams["Alpha"], teams["Echo"]))
    conn.commit()


@pytest.fixture
def league_db(tmp_path):
    conn = db.connect(tmp_path / "test.db")
    seed_league(conn)
    return conn
