"""Point-in-time features.

`History` replays every finished series in date order. Before applying a series it
records the features as they were known at that moment, so training rows never see
their own outcome or anything after it. The final state is used for live predictions.

All features are team1 − team2 differences (antisymmetric), so swapping the teams
flips the sign and the model needs no intercept.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from rlpredictor.model.ratings import RatingSystem, interleave

import math
import sqlite3
import numpy as np


FEATURES = [
    "rating",          # roster-aware Elo difference / 100
    "rating_xregion",  # same, only when teams come from different regions
    "h2h",             # shrunk head-to-head game win rate − 0.5 (last 365 d)
    "form",            # shrunk game win rate difference (last 20 games)
    "goal_diff",       # shrunk goal differential per game difference (last 20 games)
    "continuity",      # log(1 + series played by the current trio) difference
    "new_players",     # players new to the lineup within 120 d (team1 − team2)
    "player_value",    # mean Ballchasing composite z-score of the 3 players, difference
]
FACTOR_GROUPS = {
    "Rating": ["rating", "rating_xregion"],
    "Head-to-head": ["h2h"],
    "Recent form": ["form", "goal_diff"],
    "Roster stability": ["continuity", "new_players"],
    "Player value": ["player_value"],
}

H2H_WINDOW = timedelta(days=365)
FORM_GAMES = 20
CHANGE_WINDOW = timedelta(days=120)
PV_METRICS = ["score", "ga", "saves", "shots"]
THIN_GAMES = 50                 # rated games before a player's Elo is trusted


@dataclass
class SeriesRecord:
    id: str
    date: date
    tier: int | None
    event_root: str | None
    t1: int
    t2: int
    s1: int | None
    s2: int | None
    best_of: int | None
    finished: bool
    winners: list[int]
    goals: list[tuple[int, int]]
    roster1: tuple[str, ...] | None
    roster2: tuple[str, ...] | None
    bc: dict[str, dict] = field(default_factory=dict)  # player_id -> per-game stats


@dataclass
class Dataset:
    X: np.ndarray            # one row per game
    y: np.ndarray
    series_of_row: np.ndarray
    series: list[SeriesRecord]
    series_X: np.ndarray     # one row per series (same order as `series`)
    series_U: np.ndarray     # per series: trio series together (t1, t2), idle days (t1, t2),
                             # share of starters with thin rating history (t1, t2); -1 = unknown

def load_series(conn: sqlite3.Connection, finished_only: bool = True) -> list[SeriesRecord]:
    rosters: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for r in conn.execute("SELECT event_root, team_id, player_id, slot FROM event_rosters"):
        rosters[(r[0], r[1])].append((r[3], r[2]))
    roster_t = {k: tuple(p for _, p in sorted(v)) for k, v in rosters.items()}

    games: dict[str, list[tuple]] = defaultdict(list)
    for r in conn.execute("SELECT series_id, n, score1, score2, winner FROM games ORDER BY series_id, n"):
        games[r[0]].append((r[1], r[2], r[3], r[4]))

    bc: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in conn.execute("SELECT * FROM player_series_stats"):
        g = r["games"] or 0
        if g <= 0: continue

        bc[r["series_id"]][r["player_id"]] = {
            "team_id": r["team_id"],
            "score": (r["score"] or 0) / g,
            "ga": ((r["goals"] or 0) + (r["assists"] or 0)) / g,
            "saves": (r["saves"] or 0) / g,
            "shots": (r["shots"] or 0) / g,
        }

    q = "SELECT * FROM series WHERE date IS NOT NULL"
    if finished_only:
        q += " AND finished = 1 AND walkover = 0 AND score1 IS NOT NULL AND score2 IS NOT NULL"

    q += " ORDER BY date, tier DESC, id"
    out: list[SeriesRecord] = []
    for s in conn.execute(q):
        gl = games.get(s["id"], [])
        winners = [g[3] for g in gl if g[3] in (1, 2)]
        s1, s2 = s["score1"], s["score2"]
        if s1 is not None and s2 is not None and (len(winners) != s1 + s2 or
                                                  winners.count(1) != s1):
            winners = interleave(s1, s2)

        if finished_only and not winners: continue

        goals = [(g[1], g[2]) for g in gl if g[1] is not None and g[2] is not None]
        out.append(SeriesRecord(
            id=s["id"], date=date.fromisoformat(s["date"]), tier=s["tier"],
            event_root=s["event_root"], t1=s["team1_id"], t2=s["team2_id"], s1=s1, s2=s2,
            best_of=s["best_of"], finished=bool(s["finished"]), winners=winners, goals=goals,
            roster1=roster_t.get((s["event_root"], s["team1_id"])),
            roster2=roster_t.get((s["event_root"], s["team2_id"])),
            bc=bc.get(s["id"], {}),
        ))

    return out

def team_regions(conn: sqlite3.Connection) -> dict[int, str | None]:
    return {r[0]: r[1] for r in conn.execute("SELECT id, region FROM teams")}

class _Welford:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 1.0


class History:
    def __init__(self, regions: dict[int, str | None] | None = None):
        self.ratings = RatingSystem()
        self.regions = regions or {}
        self.team_games: dict[int, list[tuple[date, int, int | None]]] = defaultdict(list)
        self.h2h: dict[tuple[int, int], list[tuple[date, int, int]]] = defaultdict(list)
        self.trio_count: dict[tuple[int, frozenset], int] = defaultdict(int)
        self.lineups: dict[int, list[tuple[date, frozenset]]] = defaultdict(list)
        self.player_first: dict[tuple[int, str], date] = {}
        self.pv: dict[str, float] = {}
        self.pv_n: dict[str, int] = defaultdict(int)
        self.league = {m: _Welford() for m in PV_METRICS}
        self.last_date: date | None = None

    def _form(self, t: int, on: date) -> tuple[float, float, int, int, float]:
        rows = [g for g in self.team_games.get(t, []) if g[0] < on][-FORM_GAMES:]
        w = sum(g[1] for g in rows)
        n = len(rows)
        gds = [g[2] for g in rows if g[2] is not None]
        win = (w + 2) / (n + 4)
        gd = sum(gds) / (len(gds) + 3)

        return win, gd, w, n - w, (sum(gds) / len(gds) if gds else 0.0)

    def _h2h(self, t1: int, t2: int, on: date) -> tuple[float, dict]:
        key = (min(t1, t2), max(t1, t2))
        rows = [r for r in self.h2h.get(key, []) if on - H2H_WINDOW <= r[0] < on]
        # stored as (date, games won by key[0], games won by key[1])
        g1 = sum(r[1] if key[0] == t1 else r[2] for r in rows)
        g2 = sum(r[2] if key[0] == t1 else r[1] for r in rows)
        s1 = sum(1 for r in rows if (r[1] > r[2]) == (key[0] == t1))
        k = 4.0
        val = (g1 + k * 0.5) / (g1 + g2 + k) - 0.5

        return val, {"series": (s1, len(rows) - s1), "games": (g1, g2),
                     "last": max((r[0] for r in rows), default=None)}

    def _continuity(self, t: int, roster: tuple[str, ...] | None, on: date):
        if not roster or len(roster) < 3:
            return None, None, []
        trio = frozenset(roster[:3])
        together = self.trio_count.get((t, trio), 0)
        # "New" = first ever appearance with this team within CHANGE_WINDOW (or never
        # played for it, e.g. a what-if lineup). Robust to stand-ins swapping in and out.
        # A brand-new team has no "previous" roster, so nobody counts as new.
        new: list[str] = []
        lu = self.lineups.get(t, [])
        if lu:
            team_start = lu[0][0]
            for p in roster[:3]:
                first = self.player_first.get((t, p))
                if first is None or (first > team_start and on - first < CHANGE_WINDOW):
                    new.append(p)

        return math.log1p(together), together, new

    def _idle(self, t: int, on: date) -> float | None:
        """Days since the team's last game before `on` (None if it never played)."""
        for g in reversed(self.team_games.get(t, [])):
            if g[0] < on:
                return float((on - g[0]).days)
        return None

    def _thin(self, roster: tuple[str, ...] | None) -> tuple[float | None, list[tuple[str, int]]]:
        """Share of the three starters with fewer than THIN_GAMES rated games (cold-start Elo)."""
        if not roster or len(roster) < 3:
            return None, []
        thin = [(p, self.ratings.players.games(p)) for p in roster[:3]
                if self.ratings.players.games(p) < THIN_GAMES]
        return len(thin) / 3, thin

    def _player_value(self, roster: tuple[str, ...] | None):
        if not roster or len(roster) < 3:
            return None, {}
        vals = {p: self.pv[p] for p in roster[:3] if p in self.pv and self.pv_n[p] >= 2}
        if len(vals) < 2:
            return None, vals

        return sum(vals.values()) / len(vals), vals

    def features(self, t1: int, t2: int, r1, r2, on: date) -> tuple[np.ndarray, dict]:
        rs = self.ratings
        st1, st2 = rs.strength(t1, r1, on), rs.strength(t2, r2, on)
        reg1, reg2 = self.regions.get(t1), self.regions.get(t2)
        xregion = bool(reg1 and reg2 and reg1 != reg2)
        h2h, h2h_ev = self._h2h(t1, t2, on)
        f1, gd1, w1, l1, rgd1 = self._form(t1, on)
        f2, gd2, w2, l2, rgd2 = self._form(t2, on)
        c1, n1, new1 = self._continuity(t1, r1, on)
        c2, n2, new2 = self._continuity(t2, r2, on)
        pv1, pvs1 = self._player_value(r1)
        pv2, pvs2 = self._player_value(r2)

        x = np.array([
            (st1 - st2) / 100.0,
            (st1 - st2) / 100.0 if xregion else 0.0,
            h2h,
            f1 - f2,
            gd1 - gd2,
            (c1 - c2) if c1 is not None and c2 is not None else 0.0,
            float(len(new1) - len(new2)) if c1 is not None and c2 is not None else 0.0,
            (pv1 - pv2) if pv1 is not None and pv2 is not None else 0.0,
        ])
        ev = {
            "rating": (st1, st2), "regions": (reg1, reg2), "xregion": xregion,
            "h2h": h2h_ev,
            "form": ((w1, l1, rgd1), (w2, l2, rgd2)),
            "continuity": ((n1, new1), (n2, new2)),
            "player_value": ((pv1, pvs1), (pv2, pvs2)),
            "idle": (self._idle(t1, on), self._idle(t2, on)),
            "thin": (self._thin(r1), self._thin(r2)),
        }

        return x, ev

    # Update the history with a new series, so future features reflect it.
    def update(self, s: SeriesRecord) -> None:
        self.ratings.update_series(s.t1, s.t2, s.roster1, s.roster2, s.winners, s.date, s.tier)

        gds = [g1 - g2 for g1, g2 in s.goals] if len(s.goals) == len(s.winners) else [None] * len(s.winners)
        for w, gd in zip(s.winners, gds):
            self.team_games[s.t1].append((s.date, int(w == 1), gd))
            self.team_games[s.t2].append((s.date, int(w == 2), -gd if gd is not None else None))

        key = (min(s.t1, s.t2), max(s.t1, s.t2))
        a, b = s.winners.count(1), s.winners.count(2)
        self.h2h[key].append((s.date, a, b) if key[0] == s.t1 else (s.date, b, a))

        for t, r in ((s.t1, s.roster1), (s.t2, s.roster2)):
            if r and len(r) >= 3:
                trio = frozenset(r[:3])
                self.trio_count[(t, trio)] += 1
                if not self.lineups[t] or self.lineups[t][-1][1] != trio:
                    self.lineups[t].append((s.date, trio))
                for p in r[:3]:
                    self.player_first.setdefault((t, p), s.date)

        # Player value: league-relative z-scores, exponentially weighted per player.
        for pid, st in s.bc.items():
            for m in PV_METRICS:
                self.league[m].add(st[m])
        for pid, st in s.bc.items():
            if self.league["score"].n < 30: break

            z = sum((st[m] - self.league[m].mean) / (self.league[m].std or 1.0)
                    for m in PV_METRICS) / len(PV_METRICS)
            prev = self.pv.get(pid)
            self.pv[pid] = z if prev is None else 0.7 * prev + 0.3 * z
            self.pv_n[pid] += 1
        self.last_date = s.date

def build(conn: sqlite3.Connection) -> tuple[Dataset, History]:
    series = load_series(conn)
    hist = History(team_regions(conn))
    X_rows, y, sidx, sX, sU = [], [], [], [], []
    for i, s in enumerate(series):
        x, ev = hist.features(s.t1, s.t2, s.roster1, s.roster2, s.date)
        sX.append(x)
        (n1, _), (n2, _) = ev["continuity"]
        i1, i2 = ev["idle"]
        (th1, _), (th2, _) = ev["thin"]
        sU.append(tuple(-1 if v is None else v for v in (n1, n2, i1, i2, th1, th2)))
        for w in s.winners:
            X_rows.append(x)
            y.append(1 if w == 1 else 0)
            sidx.append(i)
        hist.update(s)
    nf = len(FEATURES)
    ds = Dataset(
        X=np.array(X_rows).reshape(-1, nf), y=np.array(y), series_of_row=np.array(sidx),
        series=series, series_X=np.array(sX).reshape(-1, nf),
        series_U=np.array(sU, dtype=float).reshape(-1, 6),
    )
    
    return ds, hist
