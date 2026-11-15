"""Walk-forward backtest: for each month, train on everything before it, test on it.

Compares the full model against a 50/50 coin flip and an Elo-only model, and runs
leave-one-factor-out ablations so factors that don't improve log-loss can be dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from rlpredictor.model.confidence import LEVELS, IDLE_FULL
from rlpredictor.model.features import FACTOR_GROUPS, FEATURES, Dataset
from rlpredictor.model.series import series_win_prob, series_win_prob_vec
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

def _folds(ds: Dataset, warmup_months: int = 4, min_train_games: int = 2000):
    """Monthly walk-forward folds: (month, train row mask, test series idx, test row mask)."""
    dates = np.array([s.date for s in ds.series])
    row_dates = dates[ds.series_of_row]
    months = _months(dates.min(), dates.max())[warmup_months:]
    for i, m in enumerate(months):
        nxt = months[i + 1] if i + 1 < len(months) else date(9999, 1, 1)
        train = row_dates < m
        if train.sum() < min_train_games: continue

        test_s = [j for j, d in enumerate(dates) if m <= d < nxt]
        if not test_s: continue

        yield m, train, test_s, np.isin(ds.series_of_row, test_s)


def _bo(s) -> int:
    return s.best_of or (2 * max(s.s1, s.s2) - 1)


def walk_forward(ds: Dataset, variants: dict[str, list[str] | None], warmup_months: int = 4,
                 min_train_games: int = 2000) -> dict[str, Result]:

    preds: dict[str, list] = {k: [] for k in variants}
    gpreds: dict[str, list] = {k: [] for k in variants}
    ys, gys = [], []

    for m, train, test_s, test_rows in _folds(ds, warmup_months, min_train_games):
        for name, active in variants.items():
            if active is None:
                ps = [0.5] * len(test_s)
                pg = np.full(test_rows.sum(), 0.5)
            else:
                model = fit(ds.X[train], ds.y[train], active)
                pg = model.p_game(ds.X[test_rows])
                ps = []
                for j in test_s:
                    ps.append(series_win_prob(float(model.p_game(ds.series_X[j])), _bo(ds.series[j])))

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


# ---- confidence: out-of-sample shrink tuning and calibration levels ----------
@dataclass
class OOS:
    """Out-of-sample series predictions from the walk-forward folds."""
    z: np.ndarray        # game logit (unshrunk)
    U: np.ndarray        # (n1, n2, idle1, idle2, thin1, thin2) with -1 = unknown
    bo: np.ndarray
    y: np.ndarray        # 1 if team1 won the series
    month: np.ndarray    # fold month (date)
    idx: np.ndarray      # index into Dataset.series

    def subset(self, mask) -> "OOS":
        return OOS(self.z[mask], self.U[mask], self.bo[mask], self.y[mask], self.month[mask],
                   self.idx[mask])


def oos_predictions(ds: Dataset, active: list[str], warmup_months: int = 4,
                    min_train_games: int = 2000) -> OOS:
    z, U, bo, y, mo, ix = [], [], [], [], [], []
    for m, train, test_s, _ in _folds(ds, warmup_months, min_train_games):
        model = fit(ds.X[train], ds.y[train], active)
        z += list(model.logit(ds.series_X[test_s]))
        U.append(ds.series_U[test_s])
        bo += [_bo(ds.series[j]) for j in test_s]
        y += [int(ds.series[j].s1 > ds.series[j].s2) for j in test_s]
        mo += [m] * len(test_s)
        ix += list(test_s)
    return OOS(np.array(z), np.vstack(U) if U else np.zeros((0, 6)), np.array(bo), np.array(y),
               np.array(mo), np.array(ix, dtype=int))


def u_array(U: np.ndarray, k: float, gamma: float, delta: float = 0.0) -> np.ndarray:
    """Vectorised confidence.uncertainty over OOS.U rows."""
    def unsettled(n):
        return np.where(n < 0, 1.0, 1.0 / (1.0 + np.maximum(n, 0) / k))
    roster = (unsettled(U[:, 0]) + unsettled(U[:, 1])) / 2
    idle = np.maximum(np.maximum(U[:, 2], U[:, 3]), 0)
    thin = (np.where(U[:, 4] < 0, 1.0, U[:, 4]) + np.where(U[:, 5] < 0, 1.0, U[:, 5])) / 2
    return roster + gamma * np.minimum(1.0, idle / IDLE_FULL) + delta * thin


def _u(o: "OOS", shrink: dict) -> np.ndarray:
    return u_array(o.U, shrink.get("k", 5.0), shrink.get("gamma", 0.0), shrink.get("delta", 0.0))


def series_probs(o: OOS, shrink: dict) -> np.ndarray:
    beta = shrink.get("beta", 0.0)
    z = o.z
    if beta:
        z = z / (1.0 + beta * _u(o, shrink))
    return series_win_prob_vec(1.0 / (1.0 + np.exp(-z)), o.bo)


BETAS = [round(0.1 * i, 1) for i in range(11)]
KS = [3.0, 5.0, 10.0]
GAMMAS = [0.0, 0.5, 1.0]
DELTAS = [0.0, 0.5, 1.0, 2.0]


def tune_shrink(o: OOS) -> dict:
    """Grid-search beta/k/gamma/delta by series log-loss. Only rescales logits: no refits."""
    from itertools import product

    best = {"beta": 0.0, "k": 5.0, "gamma": 0.0, "delta": 0.0}
    best_ll = log_loss(series_probs(o, {}), o.y)
    for k, g, dl, b in product(KS, GAMMAS, DELTAS, BETAS[1:]):
        cand = {"beta": b, "k": k, "gamma": g, "delta": dl}
        ll = log_loss(series_probs(o, cand), o.y)
        if ll < best_ll - 1e-6:
            best, best_ll = cand, ll
    return best


def calibrate_levels(o: OOS, shrink: dict) -> dict:
    """Low/Medium/High by u terciles, with the out-of-sample accuracy at each level."""
    u = _u(o, shrink)
    lo, hi = (float(v) for v in np.quantile(u, [1 / 3, 2 / 3]))
    p_raw, p = series_probs(o, {}), series_probs(o, shrink)
    out = {"cuts": [lo, hi], "levels": {}}
    names = np.where(u <= lo, "High", np.where(u <= hi, "Medium", "Low"))
    for name in LEVELS:
        m = names == name
        if not m.any():
            continue
        out["levels"][name] = {
            "n": int(m.sum()),
            "acc": float(np.mean((p[m] > 0.5) == (o.y[m] == 1))),
            "claimed_raw": float(np.mean(np.maximum(p_raw[m], 1 - p_raw[m]))),
            "claimed": float(np.mean(np.maximum(p[m], 1 - p[m]))),
            "logloss_raw": log_loss(p_raw[m], o.y[m]),
            "logloss": log_loss(p[m], o.y[m]),
        }
    return out


def confidence_report(o: OOS) -> tuple[dict, dict, dict]:
    """Tune on the first half of the OOS months, evaluate on the second half.
    Returns (shrink, held-out summary, held-out levels)."""
    months = np.unique(o.month)
    split = months[len(months) // 2]
    tune, test = o.subset(o.month < split), o.subset(o.month >= split)
    shrink = tune_shrink(tune)
    p_raw, p = series_probs(test, {}), series_probs(test, shrink)
    summary = {
        "from": str(split), "n": len(test.y),
        "logloss_raw": log_loss(p_raw, test.y), "logloss": log_loss(p, test.y),
        "brier_raw": brier(p_raw, test.y), "brier": brier(p, test.y),
        "acc_raw": float(np.mean((p_raw > 0.5) == (test.y == 1))),
        "acc": float(np.mean((p > 0.5) == (test.y == 1))),
    }
    return shrink, summary, calibrate_levels(test, shrink)


LEVEL_HEADER = (f"{'level':<8} {'series':>7} {'acc':>7} {'claimed (raw → shrunk)':>24} "
                f"{'logloss (raw → shrunk)':>24}")


def level_rows(levels: dict) -> list[str]:
    rows = []
    for name in reversed(LEVELS):
        st = levels["levels"].get(name)
        if not st:
            continue
        rows.append(f"{name:<8} {st['n']:>7} {st['acc']:>7.1%} "
                    f"{st['claimed_raw']:>13.1%} → {st['claimed']:<7.1%} "
                    f"{st['logloss_raw']:>13.4f} → {st['logloss']:<8.4f}")
    return rows


def sentiment_ablation(ds: Dataset, o: OOS, shrink: dict, diffs: dict[str, float]) -> dict | None:
    """Does adding w·(sentiment diff) to the game logit help? w tuned on the earlier half
    of the covered series, scored on the later half."""
    ids = [ds.series[j].id for j in o.idx]
    mask = np.array([i in diffs for i in ids])
    if mask.sum() < 40:
        return None
    sub = o.subset(mask)
    sd = np.array([diffs[ds.series[j].id] for j in sub.idx])
    order = np.argsort(sub.month, kind="stable")
    half = len(order) // 2
    tune, test = order[:half], order[half:]

    def probs(rows, w):
        beta = shrink.get("beta", 0.0)
        u = _u(sub.subset(rows), shrink)
        z = sub.z[rows] / (1.0 + beta * u) + w * sd[rows]
        return series_win_prob_vec(1.0 / (1.0 + np.exp(-z)), sub.bo[rows])

    grid = np.linspace(-2, 2, 41)
    w = float(grid[np.argmin([log_loss(probs(tune, g), sub.y[tune]) for g in grid])])
    return {"n": int(mask.sum()), "n_test": len(test), "weight": w,
            "logloss_without": log_loss(probs(test, 0.0), sub.y[test]),
            "logloss_with": log_loss(probs(test, w), sub.y[test])}
