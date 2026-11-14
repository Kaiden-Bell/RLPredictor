"""Elo ratings for players and teams, updated per game.

A team's strength is the mean rating of its three listed players when the roster is
known, so a roster change moves team strength immediately. Teams with no listed roster
(common in early open-qualifier rounds) fall back to a team-level Elo. Both are updated
on every game.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date

BASE = 1500.0

def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

@dataclass
class EloConfig:
    # Tuned by walk-forward log-loss on RLCS 2024-26 (see backtest.py).
    k: float = 24.0
    k_new: float = 56.0          # larger K for entities with few games (faster convergence)
    new_games: int = 15
    decay_per_30d: float = 0.99  # inactivity shrink toward BASE
    tier_k: dict[int, float] = field(default_factory=lambda: {1: 1.0, 2: 0.9, 3: 0.8})


class Elo:
    def __init__(self, cfg: EloConfig | None = None):
        self.cfg = cfg or EloConfig()
        self.r: dict = {}
        self.n: dict = {}
        self.last: dict = {}

    def get(self, key, on: date | None = None) -> float:
        r = self.r.get(key, BASE)
        if on is not None and key in self.last:
            days = (on - self.last[key]).days
            if days > 30:
                r = BASE + (r - BASE) * self.cfg.decay_per_30d ** (days / 30.0)
        return r

    def games(self, key) -> int:
        return self.n.get(key, 0)

    def _k(self, key, tier: int | None) -> float:
        base = self.cfg.k_new if self.n.get(key, 0) < self.cfg.new_games else self.cfg.k
        return base * self.cfg.tier_k.get(tier or 3, 0.8)

    def touch(self, keys, on: date) -> None:
        """Apply inactivity decay before an update."""
        for k in keys:
            self.r[k] = self.get(k, on)
            self.last[k] = on

    def apply(self, key, delta_unit: float, tier: int | None) -> None:
        self.r[key] = self.r.get(key, BASE) + self._k(key, tier) * delta_unit
        self.n[key] = self.n.get(key, 0) + 1

class RatingSystem:
    """Player + team Elo with roster-aware team strength."""

    def __init__(self, cfg: EloConfig | None = None):
        self.players = Elo(cfg)
        self.teams = Elo(cfg)

    def strength(self, team_id: int, roster: tuple[str, ...] | None, on: date | None) -> float:
        if roster and len(roster) >= 3:
            vals = [self.players.get(p, on) for p in roster[:3]]
            known = [p for p in roster[:3] if self.players.games(p) > 0]
            if known:
                # Blend toward team Elo while players have little history.
                pr = sum(vals) / len(vals)
                w = min(1.0, sum(self.players.games(p) for p in roster[:3]) / (3 * 10))
                return w * pr + (1 - w) * self.teams.get(team_id, on)

        return self.teams.get(team_id, on)

    def update_series(self, t1: int, t2: int, r1: tuple[str, ...] | None, r2: tuple[str, ...] | None,
                      winners: list[int], on: date, tier: int | None) -> None:
        p1 = tuple(r1[:3]) if r1 and len(r1) >= 3 else ()
        p2 = tuple(r2[:3]) if r2 and len(r2) >= 3 else ()
        self.teams.touch((t1, t2), on)
        self.players.touch(p1 + p2, on)
        for w in winners:
            e = expected(self.strength(t1, p1, on), self.strength(t2, p2, on))
            s = 1.0 if w == 1 else 0.0
            d = s - e
            self.teams.apply(t1, d, tier)
            self.teams.apply(t2, -d, tier)
            for p in p1:
                self.players.apply(p, d, tier)
            for p in p2:
                self.players.apply(p, -d, tier)


def interleave(s1: int, s2: int) -> list[int]:
    """Game winner sequence when only the series score is known (alternate to avoid bias)."""
    out: list[int] = []
    a, b = s1, s2
    while a or b:
        if a:
            out.append(1)
            a -= 1
        if b:
            out.append(2)
            b -= 1
            
    return out
