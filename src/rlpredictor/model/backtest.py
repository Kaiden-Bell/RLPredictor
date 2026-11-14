"""Walk-forward backtest: for each month, train on everything before it, test on it.

Compares the full model against a 50/50 coin flip and an Elo-only model, and runs
leave-one-factor-out ablations so factors that don't improve log-loss can be dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from rlpredictor.model.features import FACTOR_GROUPS, FEATURES, Dataset
from rlpredictor.model.series import series_win_prob
from rlpredictor.model.train import brier, fit, log_loss

ELO_ONLY = FACTOR_GROUPS["Rating"]

@dataclass
class Result:
    name: str
    n_series: int
    series_logloss: float
    series_brier: float
    series_acc: float
    game_logloss: float

    def row(self) -> str:
        return (f"{self.name:<28} {self.n_series:>7} {self.series_logloss:>9.4f} "
                f"{self.series_brier:>8.4f} {self.series_acc:>7.1%} {self.game_logloss:>9.4f}")

HEADER = f"{'model':<28} {'series':>7} {'logloss':>9} {'brier':>8} {'acc':>7} {'game_ll':>9}"

def _months(start: date, end: date) -> list[date]:

    out, y, m = [], start.year, start.month
    while date(y, m, 1) <= end:
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    return out

def walk_forward(ds: Dataset, variants: dict[str, list[str] | None], warmup_months: int = 4,
                 min_train_games: int = 2000) -> dict[str, Result]:

    dates = np.array([s.date for s in ds.series])
    row_dates = dates[ds.series_of_row]
    months = _months(dates.min(), dates.max())[warmup_months:]

    preds: dict[str, list] = {k: [] for k in variants}
    gpreds: dict[str, list] = {k: [] for k in variants}
    ys, gys = [], []

    for i, m in enumerate(months):
        nxt = months[i + 1] if i + 1 < len(months) else date(9999, 1, 1)
        train = row_dates < m
        if train.sum() < min_train_games: continue

        test_s = [j for j, d in enumerate(dates) if m <= d < nxt]
        if not test_s: continue

        test_rows = np.isin(ds.series_of_row, test_s)
        for name, active in variants.items():
            if active is None:
                ps = [0.5] * len(test_s)
                pg = np.full(test_rows.sum(), 0.5)
            else:
                model = fit(ds.X[train], ds.y[train], active)
                pg = model.p_game(ds.X[test_rows])
                ps = []
                for j in test_s:
                    s = ds.series[j]
                    bo = s.best_of or (2 * max(s.s1, s.s2) - 1)
                    ps.append(series_win_prob(float(model.p_game(ds.series_X[j])), bo))

            preds[name] += ps
            gpreds[name] += list(pg)

        ys += [1 if ds.series[j].s1 > ds.series[j].s2 else 0 for j in test_s]
        gys += list(ds.y[test_rows])

    y = np.array(ys)
    gy = np.array(gys)
    out = {}
    for name in variants:
        p = np.array(preds[name])
        gp = np.array(gpreds[name])
        out[name] = Result(
            name=name, n_series=len(y), series_logloss=log_loss(p, y), series_brier=brier(p, y),
            series_acc=float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1)))) if len(y) else 0.0,
            game_logloss=log_loss(gp, gy),
        )

    return out


def run(ds: Dataset) -> tuple[dict[str, Result], list[str]]:
    """Full comparison + ablations. Returns (results, factor groups worth keeping)."""
    variants: dict[str, list[str] | None] = {
        "coin flip": None,
        "elo only": ELO_ONLY,
        "full model": list(FEATURES),
    }

    for group, feats in FACTOR_GROUPS.items():
        if group == "Rating":
            continue
        variants[f"full − {group}"] = [f for f in FEATURES if f not in feats]
    res = walk_forward(ds, variants)
    full = res["full model"].game_logloss
    keep = ["Rating"]
    for group in FACTOR_GROUPS:
        if group == "Rating": continue

        if res[f"full − {group}"].game_logloss > full + 1e-5:
            keep.append(group)
            
    return res, keep
