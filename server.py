"""
server.py — FastAPI webserver for RLPredictor.

Stateless compute layer: all data lives in per-session temp directories.
Users upload/download their own .db files (includes model weights).
"""

import os
import sys
import uuid
import json
import shutil
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

SESSION_DIR = Path(tempfile.gettempdir()) / "rlpredictor_sessions"
SESSION_DIR.mkdir(exist_ok=True)

# In-memory session store: session_id -> {db_path, bc_api_key, match_data, ...}
sessions: dict[str, dict] = {}


def get_session(request: Request) -> dict:
    """Get or raise for a valid session from cookie."""
    sid = request.cookies.get("rlp_session")
    if not sid or sid not in sessions:
        raise HTTPException(status_code=401, detail="No active session. Please create one first.")
    return sessions[sid]


def get_session_db_path(session: dict) -> Path:
    return session["db_path"]


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("RLPredictor server starting...")
    yield
    # Cleanup all session temp dirs on shutdown
    for sid, sess in sessions.items():
        sess_dir = sess.get("dir")
        if sess_dir and Path(sess_dir).exists():
            shutil.rmtree(sess_dir, ignore_errors=True)
    print("RLPredictor server shut down. All sessions cleaned up.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RLPredictor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionNewRequest(BaseModel):
    bc_api_key: str


class ScrapeRequest(BaseModel):
    url: str
    sections: Optional[list[str]] = None


class MatchLoadRequest(BaseModel):
    match_index: int


class PredictRequest(BaseModel):
    player: str
    stat: str
    threshold: float
    over: bool = True
    num_games: Optional[int] = None


class PropLine(BaseModel):
    Goals: Optional[float] = None
    Saves: Optional[float] = None
    Demos: Optional[float] = None


class TeamLines(BaseModel):
    team_name: str
    players: dict[str, PropLine]


class PropSheetRequest(BaseModel):
    num_games: Optional[int] = None
    lines: dict[str, TeamLines]  # "team1" and "team2"


class TrainRequest(BaseModel):
    epochs: int = 200
    lr: float = 0.001


# ---------------------------------------------------------------------------
# Helper: override DB_PATH for a session
# ---------------------------------------------------------------------------

def _override_db_path(session: dict):
    """Temporarily patch database.py to use the session's DB path."""
    import utils.database as db_mod
    db_mod.DB_PATH = session["db_path"]


# ---------------------------------------------------------------------------
# Routes: Landing page
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Routes: Session management
# ---------------------------------------------------------------------------

@app.post("/api/session/new")
async def session_new(response: Response, bc_api_key: str = Form(...),
                      db_file: Optional[UploadFile] = File(None)):
    """Create a new session with a BC API key and optionally upload a .db file."""
    sid = str(uuid.uuid4())
    sess_dir = SESSION_DIR / sid
    sess_dir.mkdir(parents=True)
    db_path = sess_dir / "predictor.db"

    if db_file and db_file.filename:
        # Save uploaded DB
        content = await db_file.read()
        db_path.write_bytes(content)
    
    # Initialize the DB schema (creates tables if they don't exist)
    from utils.database import initialize_database, import_ids_json
    initialize_database(db_path)

    # If fresh DB, import ids.json for player ID mappings
    ids_json = Path(__file__).parent / "data" / "ids.json"
    if ids_json.exists():
        from utils.database import get_connection
        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
        if count == 0:
            import_ids_json(str(ids_json), conn)
        conn.close()

    sessions[sid] = {
        "id": sid,
        "dir": str(sess_dir),
        "db_path": db_path,
        "bc_api_key": bc_api_key,
        "matchups": None,
        "match_data": None,
    }

    response.set_cookie("rlp_session", sid, httponly=True, samesite="lax")
    
    has_model = False
    from utils.database import get_connection
    conn = get_connection(db_path)
    model_count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    replay_count = conn.execute("SELECT COUNT(*) FROM replays").fetchone()[0]
    player_count = conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0]
    conn.close()
    has_model = model_count > 0

    return {
        "session_id": sid,
        "db_uploaded": db_file is not None and db_file.filename is not None,
        "has_model": has_model,
        "cached_replays": replay_count,
        "player_ids": player_count,
    }


@app.get("/api/session/download")
async def session_download(request: Request):
    """Download the session's .db file (includes model weights, cached data, etc)."""
    session = get_session(request)
    db_path = get_session_db_path(session)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="No database found in session.")
    return FileResponse(
        str(db_path),
        media_type="application/octet-stream",
        filename="rlpredictor.db",
    )


@app.delete("/api/session")
async def session_delete(request: Request, response: Response):
    """End the session, clean up temp files."""
    sid = request.cookies.get("rlp_session")
    if sid and sid in sessions:
        sess = sessions.pop(sid)
        sess_dir = sess.get("dir")
        if sess_dir and Path(sess_dir).exists():
            shutil.rmtree(sess_dir, ignore_errors=True)
    response.delete_cookie("rlp_session")
    return {"status": "session ended"}


# ---------------------------------------------------------------------------
# Routes: Tournament scraping
# ---------------------------------------------------------------------------

@app.post("/api/scrape")
async def scrape_tournament(req: ScrapeRequest, request: Request):
    """Scrape a Liquipedia tournament URL and return matchups."""
    session = get_session(request)
    _override_db_path(session)

    def _scrape():
        from scrapers.playoff_scraper import scrape
        df = scrape(req.url, sections=req.sections)
        return df

    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _scrape)

    matchups = df.to_dict("records")
    # Convert player lists from strings back to lists if needed
    for m in matchups:
        for key in ("team1_players", "team2_players"):
            val = m.get(key)
            if isinstance(val, str):
                try:
                    m[key] = json.loads(val)
                except Exception:
                    m[key] = []

    session["matchups"] = matchups
    
    # Filter to only concrete matchups
    concrete = [m for m in matchups if m.get("team1") and m.get("team2")]

    return {
        "total": len(matchups),
        "concrete": len(concrete),
        "matchups": concrete,
    }


# ---------------------------------------------------------------------------
# Routes: Match loading (THE BIG CALL)
# ---------------------------------------------------------------------------

@app.post("/api/match/load")
async def match_load(req: MatchLoadRequest, request: Request):
    """Load all data for a specific matchup (H2H, generic, momentum, sentiment)."""
    session = get_session(request)
    _override_db_path(session)

    matchups = session.get("matchups")
    if not matchups:
        raise HTTPException(status_code=400, detail="No matchups loaded. Scrape a tournament first.")

    concrete = [m for m in matchups if m.get("team1") and m.get("team2")]
    if req.match_index < 0 or req.match_index >= len(concrete):
        raise HTTPException(status_code=400, detail=f"Match index out of range (0-{len(concrete)-1}).")

    match = concrete[req.match_index]
    bc_key = session["bc_api_key"]
    db_path = get_session_db_path(session)

    def _load():
        from scrapers.h2h_ballchasing import Ballchasing, load_player_id_map, resolve_ids, getH2HStats
        from stats import replayStats, rankedActivity
        from sentiment import get_player_sentiment

        os.environ["BALLCHASING_API_KEY"] = bc_key
        bc = Ballchasing(key=bc_key)

        t1, t2 = match["team1"], match["team2"]
        r1 = match.get("team1_players") or []
        r2 = match.get("team2_players") or []

        idMap = load_player_id_map()
        ids1 = resolve_ids(r1, idMap)
        ids2 = resolve_ids(r2, idMap)
        all_ids = ids1 + ids2

        logs = []

        # 1. Generic stats
        gen_df = replayStats(bc, all_ids, logs)

        # 2. Ranked momentum
        momentum = rankedActivity(bc, all_ids, logs)

        # 3. H2H
        h2h_df, h2h_logs = getH2HStats(t1, t2, r1, r2, bc)
        logs.extend(h2h_logs)

        # 4. Sentiment for each player name
        sentiment = {}
        all_names = list(set((r1 or []) + (r2 or [])))
        for name in all_names:
            sentiment[name] = get_player_sentiment(name)

        # Roster confidence
        confident_h2h = False
        if not h2h_df.empty:
            h2h_players = set(h2h_df["Player"].str.lower().unique())
            expected = set(str(p).lower() for p in (r1 + r2))
            confident_h2h = len(h2h_players & expected) >= 4

        # Build available player names
        av_names = set()
        if not gen_df.empty:
            av_names.update(gen_df["Player"].dropna().unique())
        if not h2h_df.empty:
            av_names.update(h2h_df["Player"].dropna().unique())

        return {
            "team1": t1,
            "team2": t2,
            "rosters": {"team1": r1, "team2": r2},
            "h2h_stats": h2h_df.to_dict("records") if not h2h_df.empty else [],
            "h2h_games": len(h2h_df),
            "gen_stats": gen_df.to_dict("records") if not gen_df.empty else [],
            "gen_games": len(gen_df),
            "momentum": momentum,
            "sentiment": sentiment,
            "h2h_confident": confident_h2h,
            "available_players": sorted(av_names),
            "logs": logs[:20],
            # Store DataFrames in session for predictions
            "_h2h_df": h2h_df,
            "_gen_df": gen_df,
            "_momentum": momentum,
            "_sentiment": sentiment,
            "_idMap": idMap,
        }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _load)

    # Store internal data in session, remove from response
    session["match_data"] = {
        "match": match,
        "h2h_df": result.pop("_h2h_df"),
        "gen_df": result.pop("_gen_df"),
        "momentum": result.pop("_momentum"),
        "sentiment": result.pop("_sentiment"),
        "idMap": result.pop("_idMap"),
        "h2h_confident": result["h2h_confident"],
    }

    return result


@app.post("/api/match/load/stream")
async def match_load_stream(req: MatchLoadRequest, request: Request):
    """SSE endpoint: load match data with streaming progress updates."""
    session = get_session(request)
    _override_db_path(session)

    matchups = session.get("matchups")
    if not matchups:
        raise HTTPException(status_code=400, detail="No matchups loaded. Scrape a tournament first.")

    concrete = [m for m in matchups if m.get("team1") and m.get("team2")]
    if req.match_index < 0 or req.match_index >= len(concrete):
        raise HTTPException(status_code=400, detail=f"Match index out of range (0-{len(concrete)-1}).")

    match = concrete[req.match_index]
    bc_key = session["bc_api_key"]
    db_path = get_session_db_path(session)

    import queue
    progress_q = queue.Queue()

    def _load_with_progress():
        import time
        from scrapers.h2h_ballchasing import Ballchasing, load_player_id_map, resolve_ids, getH2HStats
        from stats import replayStats, rankedActivity
        from sentiment import get_player_sentiment

        os.environ["BALLCHASING_API_KEY"] = bc_key
        bc = Ballchasing(key=bc_key)

        t1, t2 = match["team1"], match["team2"]
        r1 = match.get("team1_players") or []
        r2 = match.get("team2_players") or []

        progress_q.put(f"Resolving player IDs for {len(r1)+len(r2)} players...")
        idMap = load_player_id_map()
        ids1 = resolve_ids(r1, idMap)
        ids2 = resolve_ids(r2, idMap)
        all_ids = ids1 + ids2
        progress_q.put(f"Found {len(all_ids)} platform IDs")

        logs = []

        progress_q.put(f"Fetching generic stats for {len(all_ids)} players...")
        gen_df = replayStats(bc, all_ids, logs)
        progress_q.put(f"Generic: {len(gen_df)} replays loaded")

        progress_q.put("Fetching ranked momentum data...")
        momentum = rankedActivity(bc, all_ids, logs)
        progress_q.put(f"Momentum data for {len(momentum)} players")

        progress_q.put(f"Fetching H2H: {t1} vs {t2}...")
        h2h_df, h2h_logs = getH2HStats(t1, t2, r1, r2, bc)
        logs.extend(h2h_logs)
        progress_q.put(f"H2H: {len(h2h_df)} game records")

        all_names = list(set((r1 or []) + (r2 or [])))
        progress_q.put(f"Analyzing sentiment for {len(all_names)} players...")
        sentiment = {}
        for name in all_names:
            sentiment[name] = get_player_sentiment(name)
        progress_q.put("Sentiment analysis complete")

        # Roster confidence
        confident_h2h = False
        if not h2h_df.empty:
            h2h_players = set(h2h_df["Player"].str.lower().unique())
            expected = set(str(p).lower() for p in (r1 + r2))
            confident_h2h = len(h2h_players & expected) >= 4

        av_names = set()
        if not gen_df.empty:
            av_names.update(gen_df["Player"].dropna().unique())
        if not h2h_df.empty:
            av_names.update(h2h_df["Player"].dropna().unique())

        result = {
            "team1": t1,
            "team2": t2,
            "rosters": {"team1": r1, "team2": r2},
            "h2h_stats": h2h_df.to_dict("records") if not h2h_df.empty else [],
            "h2h_games": len(h2h_df),
            "gen_stats": gen_df.to_dict("records") if not gen_df.empty else [],
            "gen_games": len(gen_df),
            "momentum": momentum,
            "sentiment": sentiment,
            "h2h_confident": confident_h2h,
            "available_players": sorted(av_names),
            "logs": logs[:20],
        }

        # Store internal data in session
        session["match_data"] = {
            "match": match,
            "h2h_df": h2h_df,
            "gen_df": gen_df,
            "momentum": momentum,
            "sentiment": sentiment,
            "idMap": idMap,
            "h2h_confident": confident_h2h,
        }

        progress_q.put(("__COMPLETE__", result))

    async def event_generator():
        import threading
        thread = threading.Thread(target=_load_with_progress, daemon=True)
        thread.start()

        while True:
            try:
                msg = progress_q.get(timeout=0.5)
            except Exception:
                if not thread.is_alive():
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Worker thread died unexpectedly'})}\n\n"
                    break
                continue

            if isinstance(msg, tuple) and msg[0] == "__COMPLETE__":
                yield f"data: {json.dumps({'type': 'complete', 'data': msg[1]})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'type': 'progress', 'message': str(msg)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Routes: Predictions
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict_single(req: PredictRequest, request: Request):
    """Run a single O/U prediction."""
    session = get_session(request)
    _override_db_path(session)
    md = session.get("match_data")
    if not md:
        raise HTTPException(status_code=400, detail="No match loaded. Load a match first.")

    db_path = get_session_db_path(session)

    def _predict():
        from utils.database import load_model_from_db
        from features import extract_features
        from model import predict as nn_predict

        h2h_df = md["h2h_df"]
        gen_df = md["gen_df"]
        momentum = md["momentum"]
        sentiment = md["sentiment"]
        idMap = md["idMap"]
        confident = md["h2h_confident"]
        match = md["match"]

        # Resolve player
        player = req.player
        stat = req.stat
        threshold = req.threshold

        # Find player momentum
        p_canon = player.strip().lower()
        aliases = idMap.get("aliases", {})
        players_table = idMap.get("players", {})
        resolved = aliases.get(p_canon, p_canon)
        player_ids = players_table.get(resolved, [])
        p_momentum = None
        for pid in player_ids:
            mdata = momentum.get(pid)
            if mdata and mdata.get("games", 0) > 0:
                p_momentum = mdata
                break

        sent_data = sentiment.get(player, {"score": 0.0, "status": "Unknown"})

        # Load model from session DB
        nn_model, model_meta = load_model_from_db(db_path=db_path)

        per_game_thresh = threshold / req.num_games if req.num_games and req.num_games > 1 else threshold

        feat_vec = extract_features(
            player_name=player,
            stat_name=stat,
            threshold=per_game_thresh,
            h2h_df=h2h_df,
            gen_df=gen_df,
            momentum_data=p_momentum,
            sentiment_data=sent_data,
            confident_h2h=confident,
            playlist_type=1,
        )

        result = {
            "player": player,
            "stat": stat,
            "threshold": threshold,
            "model_used": False,
        }

        if nn_model:
            nn_prob = nn_predict(nn_model, feat_vec)
            display_prob = nn_prob if req.over else (1.0 - nn_prob)
            result["model_used"] = True
        else:
            # Heuristic fallback
            display_prob = None
            if not gen_df.empty and player in gen_df["Player"].values:
                p_data = gen_df[gen_df["Player"] == player]
                if not p_data.empty and stat in p_data.columns:
                    vals = p_data[stat]
                    hits = (vals > threshold).sum() if req.over else (vals < threshold).sum()
                    display_prob = (hits / len(p_data))

        if display_prob is not None:
            if display_prob >= 0.5:
                pick = "OVER" if req.over else "UNDER"
                pick_prob = display_prob
            else:
                pick = "UNDER" if req.over else "OVER"
                pick_prob = 1.0 - display_prob

            confidence = "High" if abs(pick_prob - 0.5) > 0.2 else "Medium" if abs(pick_prob - 0.5) > 0.1 else "Low"
            result.update({
                "pick": pick,
                "probability": round(pick_prob, 4),
                "confidence": confidence,
            })
        else:
            result.update({"pick": "INSUFFICIENT DATA", "probability": None, "confidence": None})

        # Reasoning
        reasoning = {}
        if not h2h_df.empty and player in h2h_df["Player"].values:
            p_data = h2h_df[h2h_df["Player"] == player]
            if stat in p_data.columns:
                reasoning["h2h_avg"] = round(p_data[stat].mean(), 1)
                reasoning["h2h_hit_rate"] = f"{(p_data[stat] > threshold).mean():.0%}"
                reasoning["h2h_games"] = len(p_data)

        if not gen_df.empty and player in gen_df["Player"].values:
            p_data = gen_df[gen_df["Player"] == player]
            if stat in p_data.columns:
                reasoning["gen_avg"] = round(p_data[stat].mean(), 1)
                reasoning["gen_games"] = len(p_data)

        if p_momentum:
            g = p_momentum["games"]
            label = f"{'High' if g >= 20 else 'Moderate' if g >= 5 else 'Cold'} ({g} games, {p_momentum['win_rate']}% WR)"
            reasoning["momentum"] = label

        reasoning["sentiment"] = f"{sent_data.get('score', 0):+.2f} ({sent_data.get('status', 'N/A')})"
        result["reasoning"] = reasoning

        # Log prediction for verification
        import csv
        from datetime import datetime
        try:
            with open("data/prediction_log.csv", "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.now().strftime("%Y-%m-%d"),
                    player, stat, threshold,
                    "OVER" if prob >= 0.5 else "UNDER",
                    "", "High" if abs(prob-0.5)>0.2 else "Low"
                ])
        except Exception:
            pass

        return result

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _predict)


# ---------------------------------------------------------------------------
# Routes: Prop Sheet (evaluate all lines at once)
# ---------------------------------------------------------------------------

@app.post("/api/predict/sheet")
async def predict_sheet(req: PropSheetRequest, request: Request):
    """Evaluate all prop lines and return best pick per team."""
    session = get_session(request)
    _override_db_path(session)
    md = session.get("match_data")
    if not md:
        raise HTTPException(status_code=400, detail="No match loaded. Load a match first.")

    db_path = get_session_db_path(session)

    def _evaluate():
        from utils.database import load_model_from_db
        from features import extract_features
        from model import predict as nn_predict

        h2h_df = md["h2h_df"]
        gen_df = md["gen_df"]
        momentum = md["momentum"]
        sentiment = md["sentiment"]
        idMap = md["idMap"]
        confident = md["h2h_confident"]

        nn_model, _ = load_model_from_db(db_path=db_path)

        all_picks = []

        for team_key, team_data in req.lines.items():
            for player_name, lines in team_data.players.items():
                # Resolve momentum
                p_canon = player_name.strip().lower()
                aliases = idMap.get("aliases", {})
                players_table = idMap.get("players", {})
                resolved = aliases.get(p_canon, p_canon)
                player_ids = players_table.get(resolved, [])
                p_momentum = None
                for pid in player_ids:
                    mdata = momentum.get(pid)
                    if mdata and mdata.get("games", 0) > 0:
                        p_momentum = mdata
                        break

                sent_data = sentiment.get(player_name, {"score": 0.0, "status": "Unknown"})

                for stat, threshold in lines.model_dump(exclude_none=True).items():
                    per_game_thresh = threshold / req.num_games if req.num_games and req.num_games > 1 else threshold

                    feat_vec = extract_features(
                        player_name=player_name,
                        stat_name=stat,
                        threshold=per_game_thresh,
                        h2h_df=h2h_df,
                        gen_df=gen_df,
                        momentum_data=p_momentum,
                        sentiment_data=sent_data,
                        confident_h2h=confident,
                        playlist_type=1,
                    )

                    if nn_model:
                        nn_prob = nn_predict(nn_model, feat_vec)
                        # Pick the side (over or under) with higher probability
                        if nn_prob >= 0.5:
                            pick = "OVER"
                            prob = float(nn_prob)
                        else:
                            pick = "UNDER"
                            prob = float(1.0 - nn_prob)
                    else:
                        # Heuristic
                        prob = 0.5
                        pick = "INSUFFICIENT DATA"
                        if not gen_df.empty and player_name in gen_df["Player"].values:
                            p_data = gen_df[gen_df["Player"] == player_name]
                            if not p_data.empty and stat in p_data.columns:
                                hit_rate = (p_data[stat] > threshold).mean()
                                if hit_rate >= 0.5:
                                    pick = "OVER"
                                    prob = float(hit_rate)
                                else:
                                    pick = "UNDER"
                                    prob = float(1.0 - hit_rate)

                    confidence_val = abs(prob - 0.5)
                    confidence = "High" if confidence_val > 0.2 else "Medium" if confidence_val > 0.1 else "Low"

                    # Build reasoning
                    reasoning = {}
                    if not h2h_df.empty and player_name in h2h_df["Player"].values:
                        p_data = h2h_df[h2h_df["Player"] == player_name]
                        if stat in p_data.columns:
                            reasoning["h2h_avg"] = round(p_data[stat].mean(), 1)
                            reasoning["h2h_hit_rate"] = f"{(p_data[stat] > threshold).mean():.0%}"
                    if not gen_df.empty and player_name in gen_df["Player"].values:
                        p_data = gen_df[gen_df["Player"] == player_name]
                        if stat in p_data.columns:
                            reasoning["gen_avg"] = round(p_data[stat].mean(), 1)
                    if p_momentum:
                        g = p_momentum["games"]
                        reasoning["momentum"] = f"{'High' if g >= 20 else 'Moderate' if g >= 5 else 'Cold'} ({g} games)"
                    reasoning["sentiment"] = f"{sent_data.get('score', 0):+.2f} ({sent_data.get('status', 'N/A')})"

                    all_picks.append({
                        "team": team_key,
                        "team_name": team_data.team_name,
                        "player": player_name,
                        "stat": stat,
                        "threshold": threshold,
                        "pick": pick,
                        "probability": round(prob, 4),
                        "confidence": confidence,
                        "reasoning": reasoning,
                    })

        # Sort by confidence (highest probability = furthest from 0.5)
        all_picks.sort(key=lambda x: abs(x["probability"] - 0.5), reverse=True)

        # Find best pick per team
        best_picks = {}
        for p in all_picks:
            team = p["team"]
            if team not in best_picks:
                best_picks[team] = p

        # Log predictions for verification
        import csv
        from datetime import datetime
        try:
            with open("data/prediction_log.csv", "a", newline="") as f:
                w = csv.writer(f)
                dt = datetime.now().strftime("%Y-%m-%d")
                for p in all_picks:
                    w.writerow([
                        dt, p["player"], p["stat"], p["threshold"],
                        p["pick"], "", p["confidence"]
                    ])
        except Exception:
            pass

        return {
            "best_picks": best_picks,
            "full_breakdown": all_picks,
            "model_used": nn_model is not None,
            "total_lines_evaluated": len(all_picks),
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _evaluate)


# ---------------------------------------------------------------------------
# Routes: Training
# ---------------------------------------------------------------------------

@app.post("/api/train")
async def train_model(req: TrainRequest, request: Request):
    """Train the neural network from session DB data."""
    session = get_session(request)
    _override_db_path(session)
    db_path = get_session_db_path(session)

    def _train():
        from features import build_training_data
        from model import train_model as do_train
        from utils.database import save_model_to_db
        import numpy as np

        features, labels, meta = build_training_data()

        if len(features) == 0:
            return {"status": "error", "message": "No training data. Load some matches first to build cache."}

        model, history = do_train(
            features, labels,
            epochs=req.epochs,
            lr=req.lr,
            verbose=True,
        )

        best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
        save_model_to_db(model, epochs=req.epochs, val_acc=best_val_acc,
                         samples=len(features), db_path=db_path)

        return {
            "status": "complete",
            "samples": len(features),
            "best_val_acc": f"{best_val_acc:.1%}",
            "epochs_run": len(history["train_loss"]),
            "message": "Model trained and saved to session DB. Download your DB to keep it.",
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _train)


# ---------------------------------------------------------------------------
# Routes: Session info
# ---------------------------------------------------------------------------

@app.get("/api/session/info")
async def session_info(request: Request):
    """Get current session status."""
    session = get_session(request)
    db_path = get_session_db_path(session)

    from utils.database import get_connection
    conn = get_connection(db_path)
    info = {
        "session_id": session["id"],
        "has_matchups": session.get("matchups") is not None,
        "has_match_loaded": session.get("match_data") is not None,
        "cached_replays": conn.execute("SELECT COUNT(*) FROM replays").fetchone()[0],
        "player_ids": conn.execute("SELECT COUNT(*) FROM player_ids").fetchone()[0],
        "has_model": conn.execute("SELECT COUNT(*) FROM models").fetchone()[0] > 0,
    }
    conn.close()
    return info


# ---------------------------------------------------------------------------
# Routes: Verify
# ---------------------------------------------------------------------------

@app.post("/api/verify")
async def verify_predictions_api(request: Request):
    """Run verification against logged predictions."""
    import subprocess
    import os
    
    if not os.path.exists("data/prediction_log.csv"):
        return {"logs": [], "total": 0, "wins": 0, "win_rate": 0}
        
    try:
        subprocess.run(["python3", "verify_predictions.py"], check=False)
    except Exception as e:
        pass
        
    import csv
    logs = []
    wins = 0
    total = 0
    try:
        with open("data/prediction_log.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 9 and row[8] in ("WIN", "LOSS", "PUSH"):
                    total += 1
                    if row[8] == "WIN": wins += 1
                    logs.append({
                        "date": row[0],
                        "player": row[1],
                        "stat": row[2],
                        "prediction": f"{row[4]} {row[3]}",
                        "actual": row[7],
                        "result": row[8]
                    })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    logs.reverse()
    
    return {
        "logs": logs[:100], 
        "total": total,
        "wins": wins,
        "win_rate": round(wins/total*100, 1) if total > 0 else 0
    }



# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
