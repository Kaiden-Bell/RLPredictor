"""Prediction confidence: how much to trust a win probability, separate from who it favours.

Every model feature is a team1 − team2 difference, so "both rosters are brand new"
cancels out and the model alone can't be less sure about it. Uncertainty `u` is a
symmetric score built from how settled each lineup is and how long the teams have been
idle; the game logit is divided by (1 + beta·u), which pulls the probability toward 50%
without ever flipping the pick. beta/k/gamma are tuned by walk-forward log-loss
(see backtest.tune_shrink), and the Low/Medium/High levels carry the accuracy that
predictions at that level actually achieved out of sample.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

IDLE_FULL = 90.0   # days idle at which the idle term saturates
LEVELS = ("Low", "Medium", "High")


def team_unsettled(n_together: int | None, k: float) -> float:
    """1.0 for an unknown or brand-new trio, falling toward 0 as it plays together."""
    if n_together is None:
        return 1.0
    return 1.0 / (1.0 + max(n_together, 0) / k)


def uncertainty(n1: int | None, n2: int | None, idle1: float | None, idle2: float | None,
                k: float = 5.0, gamma: float = 0.0, thin1: float | None = None,
                thin2: float | None = None, delta: float = 0.0) -> float:
    """roster newness + gamma·idle + delta·(share of starters with thin rating history)."""
    roster = (team_unsettled(n1, k) + team_unsettled(n2, k)) / 2
    idle = max(idle1 or 0.0, idle2 or 0.0)
    thin = ((1.0 if thin1 is None else thin1) + (1.0 if thin2 is None else thin2)) / 2
    return roster + gamma * min(1.0, idle / IDLE_FULL) + delta * thin


def u_from_ev(ev: dict, shrink: dict) -> float:
    (n1, _), (n2, _) = ev["continuity"]
    i1, i2 = ev.get("idle", (None, None))
    (t1, _), (t2, _) = ev.get("thin", ((None, []), (None, [])))
    return uncertainty(n1, n2, i1, i2, shrink.get("k", 5.0), shrink.get("gamma", 0.0),
                       t1, t2, shrink.get("delta", 0.0))


@dataclass
class Confidence:
    level: str                       # Low | Medium | High ("Unknown" for untuned models)
    score: int                       # 0..100, higher = more settled / more data
    historical_accuracy: float | None
    n: int                           # backtest series at this level
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def level_for(u: float, levels: dict | None) -> tuple[str, float | None, int]:
    """Map u to a level using the cut points stored at train time."""
    if not levels or "cuts" not in levels:
        return "Unknown", None, 0
    lo, hi = levels["cuts"]            # u terciles: <= lo is High, > hi is Low
    name = "High" if u <= lo else ("Medium" if u <= hi else "Low")
    st = levels["levels"].get(name, {})
    return name, st.get("acc"), st.get("n", 0)


def assess(ev: dict, shrink: dict, levels: dict | None, a: str, b: str,
           pname=lambda p: p) -> tuple[float, Confidence]:
    u = u_from_ev(ev, shrink)
    k = shrink.get("k", 5.0)
    gamma = shrink.get("gamma", 0.0)
    delta = shrink.get("delta", 0.0)
    worst = 1.0 + gamma + delta    # max possible u
    score = int(round(100 * (1 - min(u, worst) / worst)))
    name, acc, n = level_for(u, levels)

    reasons = []
    for team, (n_tog, new) in zip((a, b), ev["continuity"]):
        if n_tog is None:
            reasons.append(f"{team}: lineup not known")
        elif team_unsettled(n_tog, k) >= 0.5:
            s = f"{team}: current trio has {n_tog} series together"
            if new:
                s += f" ({len(new)} of 3 new: {', '.join(pname(p) for p in new)})"
            reasons.append(s)
    for team, (share, thin) in zip((a, b), ev.get("thin", ((None, []), (None, [])))):
        if delta and thin:
            reasons.append(f"{team}: little rating history for "
                           + ", ".join(f"{pname(p)} ({g} games)" for p, g in thin))
    idle = ev.get("idle", (None, None))
    if gamma and max(i or 0 for i in idle) >= 45:
        reasons.append(f"{max(i or 0 for i in idle):.0f} days since last match "
                       f"({a} {idle[0] or 0:.0f}, {b} {idle[1] or 0:.0f})")
    if not reasons:
        reasons.append("both lineups settled and active")
    return u, Confidence(name, score, acc, n, reasons)


def meter(score: int, width: int = 5) -> str:
    filled = max(0, min(width, math.ceil(score / (100 / width)))) if score else 0
    return "▮" * filled + "▯" * (width - filled)
