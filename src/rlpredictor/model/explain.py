"""Turn a feat vec + evidence into ranked, human-readable factors."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from rlpredictor.model.features import FACTOR_GROUPS, FEATURES
from rlpredictor.model.series import series_win_prob
from rlpredictor.model.train import GameModel

import numpy as np



@dataclass
class Factor:
    name: str
    contribution_pp: float     # series-win percentage points this factor adds for `favors`
    favors: str | None
    evidence: str
    in_model: bool

    def to_dict(self) -> dict:
        return asdict(self)


def contributions(model: GameModel, x: np.ndarray, best_of: int, u: float = 0.0) -> dict[str, float]:
    """Team1 series-win pp attributable to each factor group: P(full) − P(group neutral)."""
    p_full = series_win_prob(float(model.p_game(x, u)), best_of)
    out = {}
    for group, feats in FACTOR_GROUPS.items():
        x0 = x.copy()
        for f in feats:
            x0[FEATURES.index(f)] = 0.0
        out[group] = (p_full - series_win_prob(float(model.p_game(x0, u)), best_of)) * 100

    return out


def _fmt_date(d) -> str:
    return d.strftime("%b %Y") if d else "?"


def evidence_text(group: str, ev: dict, a: str, b: str, pname) -> str:
    if group == "Rating":
        r1, r2 = ev["rating"]
        s = f"{a} rated {r1:.0f} vs {b} {r2:.0f} (roster-weighted Elo)"
        if ev["xregion"]:
            s += f"; cross-region matchup ({ev['regions'][0]} vs {ev['regions'][1]})"
        return s

    if group == "Head-to-head":
        h = ev["h2h"]
        (s1, s2), (g1, g2) = h["series"], h["games"]
        if s1 + s2 == 0:
            return "No meetings in the last 12 months"
        return (f"{a} {s1}–{s2} in series ({g1}–{g2} in games) over the last 12 months; "
                f"last met {_fmt_date(h['last'])}")

    if group == "Recent form":
        (w1, l1, gd1), (w2, l2, gd2) = ev["form"]
        return (f"Last 20 games: {a} {w1}–{l1} ({gd1:+.1f} goal diff/game), "
                f"{b} {w2}–{l2} ({gd2:+.1f})")

    if group == "Roster stability":
        parts = []
        for team, (n, new) in zip((a, b), ev["continuity"]):
            if n is None:
                parts.append(f"{team}: roster not listed")
                continue
            s = f"{team}: current trio has {n} series together"
            if new:
                s += f", recently added {', '.join(pname(p) for p in new)}"
            parts.append(s)
        return "; ".join(parts)

    if group == "Player value":
        parts = []
        for team, (avg, vals) in zip((a, b), ev["player_value"]):
            if avg is None:
                parts.append(f"{team}: not enough Ballchasing stats yet")
                continue
            top = max(vals, key=vals.get)
            parts.append(f"{team} avg {avg:+.2f}σ (top: {pname(top)} {vals[top]:+.2f}σ)")
        return "; ".join(parts)

    return ""


def explain(model: GameModel, x: np.ndarray, ev: dict, best_of: int, a: str, b: str,
            pname=lambda p: p, u: float = 0.0) -> list[Factor]:

    contrib = contributions(model, x, best_of, u)
    out = []
    for group, feats in FACTOR_GROUPS.items():
        c = contrib[group]
        in_model = any(f in model.features for f in feats)
        favors = None if abs(c) < 0.05 else (a if c > 0 else b)
        out.append(Factor(group, round(abs(c), 1), favors, evidence_text(group, ev, a, b, pname),
                          in_model))
                          
    out.sort(key=lambda f: (-f.in_model, -f.contribution_pp))

    return out
