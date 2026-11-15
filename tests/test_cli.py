import re

import pytest
import typer
from typer.testing import CliRunner

from rlpredictor import cli
from rlpredictor.cli import split_matchup
from rlpredictor.predictor import PredictError


@pytest.mark.parametrize("tokens, expected", [
    (["Gentle", "Mates", "vs", "Shopify", "Rebellion"], ("Gentle Mates", "Shopify Rebellion")),
    (["Gentle", "Mates", "VS.", "SR"], ("Gentle Mates", "SR")),
    (["kc", "v", "Team", "Vitality"], ("kc", "Team Vitality")),
    (["Gentle Mates", "Shopify Rebellion"], ("Gentle Mates", "Shopify Rebellion")),
    (["Gentle Mates vs Shopify Rebellion"], ("Gentle Mates", "Shopify Rebellion")),
    (["Gentle Mates", "vs", "Shopify Rebellion"], ("Gentle Mates", "Shopify Rebellion")),
])
def test_split_matchup(tokens, expected):
    assert split_matchup(tokens) == expected


@pytest.mark.parametrize("tokens", [
    ["Gentle", "Mates", "Shopify", "Rebellion"],   # no way to know the split
    ["a", "vs", "b", "vs", "c"],
    ["vs", "Shopify", "Rebellion"],
    ["Gentle Mates"],
])
def test_split_matchup_rejects(tokens):
    with pytest.raises(typer.BadParameter, match="Use 'vs' or quotes"):
        split_matchup(tokens)


def test_predict_cli_accepts_unquoted_names(monkeypatch):
    seen = {}

    class FakePredictor:
        def predict(self, t1, t2, **kw):
            seen.update(t1=t1, t2=t2, **kw)
            raise PredictError("stop here")

    monkeypatch.setattr(cli, "_predictor", lambda: FakePredictor())
    runner = CliRunner()
    res = runner.invoke(cli.app, ["predict", "Gentle", "Mates", "vs", "Shopify", "Rebellion",
                                  "--bo", "7"])
    assert seen["t1"] == "Gentle Mates" and seen["t2"] == "Shopify Rebellion"
    assert seen["best_of"] == 7
    assert res.exit_code == 1

    res = runner.invoke(cli.app, ["predict", "Gentle", "Mates", "Shopify", "Rebellion"])
    flat = " ".join(re.sub(r"[│╭╮╰╯─]", " ", res.output).split())  # undo rich box wrapping
    assert res.exit_code == 2 and "Use 'vs' or quotes" in flat


def test_render_shows_confidence(league_db):
    from datetime import date

    from rlpredictor.model.features import build
    from rlpredictor.model.train import fit
    from rlpredictor.predictor import Predictor

    ds, _ = build(league_db)
    model = fit(ds.X, ds.y)
    model.shrink = {"beta": 0.3, "k": 5.0, "gamma": 0.0}
    model.metrics["confidence"] = {"cuts": [0.2, 0.5], "levels": {
        "High": {"acc": 0.78, "n": 1200}, "Medium": {"acc": 0.74, "n": 1100},
        "Low": {"acc": 0.68, "n": 1000}}}
    out = cli.render(Predictor(league_db, model).predict("Bravo", "Echo", on=date(2025, 1, 1)))
    assert "Confidence: " in out and "backtest: right" in out and "not who wins" in out
