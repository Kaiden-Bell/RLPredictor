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


def _squad(conn, team: str, players: list[str], joined: str):
    from rlpredictor.data.ingest import store_squad
    from rlpredictor.data.wikitext import SquadMember

    tid = conn.execute("SELECT id FROM teams WHERE name = ?", (team,)).fetchone()[0]
    store_squad(conn, tid, [SquadMember(p, p, "", date.fromisoformat(joined)) for p in players])
    return tid


def test_team_page_squad_overrides_stale_event_roster(league_db):
    ds, _ = build(league_db)
    model = fit(ds.X, ds.y)
    last_event = league_db.execute("SELECT MAX(start_date) FROM pages").fetchone()[0]
    # Alpha picks up two Echo players after the last event.
    _squad(league_db, "Alpha", ["Alpha1", "Echo1", "Echo2"], "2025-06-01")
    p = Predictor(league_db, model)
    pred = p.predict("Alpha", "Echo", on=date(2025, 7, 1))
    assert pred.team1.roster_ids[:3] == ("alpha1", "echo1", "echo2")
    assert pred.team1.roster_source == "Liquipedia team page, since 2025-06-01"
    assert any("roster changed since Event 23" in n and "−Alpha2, Alpha3" in n for n in pred.notes)
    # Before the squad formed, the event roster is the point-in-time truth.
    old = p.predict("Alpha", "Echo", on=date.fromisoformat(last_event))
    assert old.team1.roster_ids == ("alpha1", "alpha2", "alpha3")
    assert old.team1.roster_source == "Event 23"
    # Event roster respects `on` too: early in the season Bravo hasn't swapped yet.
    early = p.predict("Bravo", "Echo", on=date(2024, 2, 1))
    assert early.team1.roster_source == "Event 1" and "newguy" not in early.team1.roster_ids


def test_squad_formed_before_event_does_not_override(league_db):
    ds, _ = build(league_db)
    _squad(league_db, "Charlie", ["x1", "x2", "x3"], "2024-01-01")
    p = Predictor(league_db, fit(ds.X, ds.y))
    pred = p.predict("Charlie", "Echo", on=date(2025, 7, 1))
    assert pred.team1.roster_source == "Event 23"


def test_new_team_gets_squad(league_db):
    from rlpredictor.data import db

    ds, _ = build(league_db)
    db.get_or_create_team(league_db, "newco", "NewCo", "Europe")
    _squad(league_db, "NewCo", ["Alpha1", "Bravo1", "Charlie1"], "2025-05-01")
    p = Predictor(league_db, fit(ds.X, ds.y))
    pred = p.predict("NewCo", "Echo", on=date(2025, 7, 1))
    assert pred.team1.roster == ["Alpha1", "Bravo1", "Charlie1"]
    assert not any("No listed roster for NewCo" in n for n in pred.notes)
