from datetime import date

import numpy as np
import pytest

from rlpredictor.model import backtest as bt
from rlpredictor.model.confidence import assess, level_for, meter, uncertainty
from rlpredictor.model.features import FEATURES, build
from rlpredictor.model.series import series_win_prob, series_win_prob_vec
from rlpredictor.model.train import GameModel, fit
from rlpredictor.predictor import Predictor


def test_uncertainty_monotonic():
    assert uncertainty(None, None, None, None) == 1.0
    assert uncertainty(0, 0, 0, 0) == 1.0
    us = [uncertainty(n, n, 0, 0, k=5) for n in (0, 2, 5, 20, 100)]
    assert us == sorted(us, reverse=True) and us[-1] < 0.1
    # an unknown roster on one side is as uncertain as a brand-new one
    assert uncertainty(None, 50, 0, 0) == uncertainty(0, 50, 0, 0)
    assert uncertainty(10, 10, 200, 0, gamma=0.5) == pytest.approx(uncertainty(10, 10, 0, 0) + 0.5)
    # thin rating history: unknown counts as fully thin
    base = uncertainty(10, 10, 0, 0, thin1=0.0, thin2=0.0, delta=1.0)
    assert uncertainty(10, 10, 0, 0, thin1=1 / 3, thin2=0.0, delta=1.0) == pytest.approx(base + 1 / 6)
    assert uncertainty(10, 10, 0, 0, delta=1.0) == pytest.approx(base + 1.0)


def test_shrink_moves_toward_half_never_flips():
    m = GameModel(features=list(FEATURES), weights=np.ones(len(FEATURES)))
    x = np.full(len(FEATURES), 0.2)
    base = float(m.p_game(x))
    assert float(m.p_game(x, u=1.0)) == base           # no shrink configured
    m.shrink = {"beta": 0.5, "k": 5.0, "gamma": 0.0}
    shrunk = float(m.p_game(x, u=1.0))
    assert 0.5 < shrunk < base
    assert float(m.p_game(x, u=0.0)) == base
    assert float(m.p_game(-x, u=1.0)) == pytest.approx(1 - shrunk)


def test_old_model_dict_loads(tmp_path):
    import joblib

    path = tmp_path / "m.joblib"
    joblib.dump({"features": ["rating"], "weights": np.zeros(len(FEATURES)),
                 "trained_through": None, "n_games": 0, "metrics": {}}, path)
    m = GameModel.load(path)
    assert m.shrink == {}


def test_vectorised_series_prob():
    p = np.array([0.3, 0.5, 0.62, 0.9])
    bo = np.array([1, 3, 5, 7])
    assert np.allclose(series_win_prob_vec(p, bo), [series_win_prob(a, b) for a, b in zip(p, bo)])


def test_levels_and_meter():
    levels = {"cuts": [0.3, 0.6], "levels": {"High": {"acc": 0.8, "n": 10}}}
    assert level_for(0.1, levels) == ("High", 0.8, 10)
    assert level_for(0.5, levels)[0] == "Medium"
    assert level_for(0.9, levels)[0] == "Low"
    assert level_for(0.1, {})[0] == "Unknown"
    assert meter(0) == "▯▯▯▯▯" and meter(100) == "▮▮▮▮▮" and meter(50) == "▮▮▮▯▯"


def test_assess_reasons():
    ev = {"continuity": ((0, ["a", "b", "c"]), (40, [])), "idle": (10.0, 120.0),
          "thin": ((1 / 3, [("rookie", 12)]), (0.0, []))}
    u, c = assess(ev, {"beta": 0.3, "k": 5.0, "gamma": 0.5, "delta": 1.0}, None, "New", "Old")
    assert any("New: little rating history for rookie (12 games)" in r for r in c.reasons)
    assert c.level == "Unknown" and 0 <= c.score <= 100
    assert any("New: current trio has 0 series together (3 of 3 new" in r for r in c.reasons)
    assert not any(r.startswith("Old") for r in c.reasons)
    assert any("120 days since last match" in r for r in c.reasons)


def test_tuning_and_confidence_in_prediction(league_db):
    ds, _ = build(league_db)
    o = bt.oos_predictions(ds, list(FEATURES), warmup_months=2, min_train_games=200)
    assert len(o.y) > 0 and o.U.shape == (len(o.y), 6)
    shrink = bt.tune_shrink(o)
    assert set(shrink) == {"beta", "k", "gamma", "delta"}
    levels = bt.calibrate_levels(o, shrink)
    assert set(levels["levels"]) <= {"Low", "Medium", "High"} and len(levels["cuts"]) == 2

    model = fit(ds.X, ds.y)
    model.shrink = {"beta": 0.5, "k": 5.0, "gamma": 0.0}
    model.metrics["confidence"] = levels
    p = Predictor(league_db, model)
    on = date(2025, 1, 1)
    steady = p.predict("Alpha", "Echo", on=on)
    # Bravo's trio changed half-way through the season, so it has fewer series together.
    changed = p.predict("Bravo", "Echo", on=on)
    assert changed.confidence.score < steady.confidence.score
    d = steady.to_dict()["confidence"]
    assert d["level"] in ("Low", "Medium", "High") and "reasons" in d
    # an unsettled side is less extreme than the unshrunk model would be
    model.shrink = {}
    raw = Predictor(league_db, model).predict("Bravo", "Echo", on=on)
    assert abs(changed.p_game - 0.5) < abs(raw.p_game - 0.5)
