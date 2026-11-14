from fastapi.testclient import TestClient

from rlpredictor import api, config
from rlpredictor.model.features import build
from rlpredictor.model.train import fit


def test_api_predict(league_db, tmp_path, monkeypatch):
    ds, _ = build(league_db)
    model_path = tmp_path / "model.joblib"
    fit(ds.X, ds.y).save(model_path)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    api._state.clear()
    api._state["conn"] = league_db

    with TestClient(api.app) as c:
        api._state["conn"] = league_db
        assert c.get("/health").json()["model"] is True
        assert c.get("/teams", params={"q": "alp"}).json()[0]["name"] == "Alpha"
        r = c.post("/predict", json={"team1": "Echo", "team2": "Alpha", "best_of": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["winner"] == "Alpha"
        assert {f["name"] for f in body["factors"]} >= {"Rating", "Head-to-head"}
        assert "Liquipedia" in body["attribution"]
        assert c.post("/predict", json={"team1": "Nobody", "team2": "Alpha"}).status_code == 400
        assert len(c.get("/events/upcoming", params={"page": "Upcoming Cup"}).json()) == 1
