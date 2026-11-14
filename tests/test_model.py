from datetime import date

import numpy as np
import pytest

from rlpredictor.model import backtest as bt
from rlpredictor.model.features import FEATURES, History, build, load_series, team_regions
from rlpredictor.model.train import fit
from rlpredictor.predictor import PredictError, Predictor


def test_no_leakage(league_db):
    """Features for series i must not change when later results change."""
    ds, _ = build(league_db)
    i = len(ds.series) // 2
    sid = ds.series[i].id
    # Flip the result of series i and everything after it.
    later = [s.id for s in ds.series[i:]]
    for s in later:
        league_db.execute("UPDATE series SET score1 = score2, score2 = score1 WHERE id = ?", (s,))
        league_db.execute("UPDATE games SET winner = 3 - winner WHERE series_id = ?", (s,))
    ds2, _ = build(league_db)
    j = [k for k, s in enumerate(ds2.series) if s.id == sid][0]
    assert j == i
    np.testing.assert_allclose(ds.series_X[: i + 1], ds2.series_X[: i + 1])


def test_antisymmetric(league_db):
    _, hist = build(league_db)
    series = load_series(league_db)
    s = series[-1]
    x1, _ = hist.features(s.t1, s.t2, s.roster1, s.roster2, s.date)
    x2, _ = hist.features(s.t2, s.t1, s.roster2, s.roster1, s.date)
    np.testing.assert_allclose(x1, -x2)


def test_model_learns_strength_and_beats_coin(league_db):
    ds, _ = build(league_db)
    model = fit(ds.X, ds.y)
    assert model.weights[FEATURES.index("rating")] > 0
    res = bt.walk_forward(ds, {"coin": None, "elo": bt.ELO_ONLY}, warmup_months=3, min_train_games=100)
    assert res["elo"].series_logloss < res["coin"].series_logloss


def test_roster_change_detected(league_db):
    _, hist = build(league_db)
    bravo = league_db.execute("SELECT id FROM teams WHERE name='Bravo'").fetchone()[0]
    on = hist.last_date
    _, n_together, new = hist._continuity(bravo, ("bravo1", "bravo2", "someone"), on)
    assert n_together == 0 and new == ["someone"]


def test_predictor_end_to_end(league_db):
    ds, hist = build(league_db)
    model = fit(ds.X, ds.y)
    p = Predictor(league_db, model)
    pred = p.predict("Alpha", "Echo", best_of=7, on=date(2025, 1, 1))
    assert pred.winner == "Alpha" and pred.winner_prob > 0.8
    assert sum(pred.score_distribution.values()) == pytest.approx(1.0, abs=1e-3)
    assert pred.factors[0].name == "Rating" and pred.factors[0].favors == "Alpha"
    assert pred.team1.roster == ["Alpha1", "Alpha2", "Alpha3"]
    # roster override is reflected
    pred2 = p.predict("Alpha", "Echo", best_of=7, roster1=["Echo1", "Echo2", "Echo3"])
    assert pred2.p_series < pred.p_series
    with pytest.raises(PredictError):
        p.predict("Alpha", "Alpha")
    up = p.upcoming("Upcoming Cup")
    assert len(up) == 1 and up[0]["prediction"]["best_of"] == 7
