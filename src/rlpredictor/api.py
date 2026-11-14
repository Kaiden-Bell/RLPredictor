"""FastAPI JSON API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from threading import Lock

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rlpredictor import config
from rlpredictor.data import db
from rlpredictor.model.features import FEATURES

_state: dict = {}
_lock = Lock()

def get_predictor():
    with _lock:
        if "predictor" not in _state:
            from rlpredictor.model.train import GameModel
            from rlpredictor.predictor import Predictor

            if not config.MODEL_PATH.exists():
                raise HTTPException(503, "No trained model. Run `rlpredict train`.")
            _state["predictor"] = Predictor(_state.get("conn") or db.connect(), GameModel.load())
        return _state["predictor"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.setdefault("conn", db.connect())
    yield
    _state.clear()

app = FastAPI(title="RLPredictor", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

class PredictRequest(BaseModel):
    team1: str
    team2: str
    best_of: int = Field(5, description="1, 3, 5 or 7")
    on: date | None = None
    roster1: list[str] | None = None
    roster2: list[str] | None = None

@app.get("/health")
def health():
    return {"ok": True, "model": config.MODEL_PATH.exists()}

@app.get("/teams")
def teams(q: str = Query(..., min_length=1)):
    conn = _state.get("conn") or db.connect()
    rows = db.find_team(conn, q)
    return [{"id": r["id"], "name": r["name"], "region": r["region"],
             "last_played": r["last_played"] if "last_played" in r.keys() else None} for r in rows]

@app.post("/predict")
def predict(req: PredictRequest):
    from rlpredictor.predictor import PredictError

    p = get_predictor()
    try:
        return p.predict(req.team1, req.team2, best_of=req.best_of, on=req.on,
                         roster1=req.roster1, roster2=req.roster2).to_dict()
    except PredictError as e:
        raise HTTPException(400, {"error": str(e), "candidates": e.candidates})

@app.get("/events/upcoming")
def upcoming(page: str = Query(..., description="Liquipedia page title or URL")):
    return get_predictor().upcoming(page)

@app.get("/model")
def model_info():
    p = get_predictor()
    m = p.model
    return {
        "trained_through": m.trained_through,
        "n_games": m.n_games,
        "factors": m.metrics.get("kept_groups"),
        "weights": {f: w for f, w in zip(FEATURES, m.weights.tolist()) if w},
        "backtest": m.metrics.get("backtest"),
        "attribution": config.ATTRIBUTION,
    }