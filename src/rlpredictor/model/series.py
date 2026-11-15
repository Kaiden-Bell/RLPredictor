"""Exact best-of-N series math from a per-game win probability."""

from __future__ import annotations
from math import comb


def score_distribution(p: float, best_of: int) -> dict[tuple[int, int], float]:
    """P(final score) for team1 with per-game win prob p, assuming independent games.

    Returns {(team1_wins, team2_wins): prob}. Probabilities sum to 1.
    """
    if best_of < 1 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number")
    n = (best_of + 1) // 2
    q = 1.0 - p
    out: dict[tuple[int, int], float] = {}
    for k in range(n):
        ways = comb(n - 1 + k, k)
        out[(n, k)] = ways * p**n * q**k
        out[(k, n)] = ways * q**n * p**k

    return out


def series_win_prob(p: float, best_of: int) -> float:
    n = (best_of + 1) // 2

    return sum(v for (a, b), v in score_distribution(p, best_of).items() if a == n)


def length_buckets(dist: dict[tuple[int, int], float], best_of: int) -> dict[str, float]:
    """DESIGN.MD buckets: sweep / dominant / competitive / decider."""
    n = (best_of + 1) // 2
    out = {"sweep": 0.0, "dominant": 0.0, "competitive": 0.0, "decider": 0.0}
    for (a, b), v in dist.items():
        loser = min(a, b)
        if loser == 0:
            out["sweep"] += v
        elif loser == n - 1:
            out["decider"] += v
        elif loser == 1:
            out["dominant"] += v
        else:
            out["competitive"] += v

    return out


def game_prob_from_series(p_series: float, best_of: int, tol: float = 1e-9) -> float:
    """Inverse of series_win_prob (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if series_win_prob(mid, best_of) < p_series:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
            
    return (lo + hi) / 2


def series_win_prob_vec(p, best_of):
    """Vectorised series_win_prob over arrays of p and best_of."""
    import numpy as np

    p = np.asarray(p, dtype=float)
    bo = np.broadcast_to(np.asarray(best_of), p.shape)
    out = np.zeros_like(p)
    for b in np.unique(bo):
        m = bo == b
        n = (int(b) + 1) // 2
        pm, qm = p[m], 1.0 - p[m]
        out[m] = sum(comb(n - 1 + k, k) * pm**n * qm**k for k in range(n))
    return out
