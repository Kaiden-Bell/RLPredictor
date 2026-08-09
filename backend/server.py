"""
Author: Kaiden Bell
Date (Coded): 2026-06-01
File Function:
- Description: FastAPI server bridging the frontend React dashboard to the Python
  scraping pipeline (playoff_scraper) and future PyTorch prediction routes.
- Usage: uvicorn server:app --reload --port 8000
"""

import os
import sys
import hashlib
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
from pydantic import BaseModel

import pandas as pd

# Ensure the backend package is importable
sys.path.insert(0, os.path.dirname(__file__))

from scrapers.playoff_scraper import scrape_playoffs
from utils.database import initialize_database

load_dotenv()


# ──────────────────────────────────────────────────────────────────────
# Playwright Lifespan (Phase 1)
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage a persistent headless Chromium instance across the app lifetime."""
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    app.state.playwright = pw
    app.state.browser = browser
    print("[STARTUP] Playwright browser launched")
    yield
    await browser.close()
    await pw.stop()
    print("[SHUTDOWN] Playwright browser closed")


app = FastAPI(
    title="RLPredictor API",
    description="Backend API for the RLPredictor Rocket League esports analytics platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Vite/Express frontend dev server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIQUIPEDIA_RL_PREFIX = "https://liquipedia.net/rocketleague/"


# ──────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    url: str
    sections: Optional[list[str]] = None


class TeamOut(BaseModel):
    id: str
    name: str
    shortName: str
    logo: str
    color: str


class BracketMatchOut(BaseModel):
    id: str
    matchIndex: int
    team1: Optional[TeamOut] = None
    team2: Optional[TeamOut] = None
    score1: Optional[int] = None
    score2: Optional[int] = None
    winnerId: Optional[str] = None
    status: str
    roundIndex: int
    section: str = ""
    round: str = ""
    bestOf: int = 5


class PlayerOut(BaseModel):
    id: str
    name: str
    role: str


class RosterOut(BaseModel):
    teamId: str
    active: list[PlayerOut]
    substitutes: list[PlayerOut]


class TournamentResponse(BaseModel):
    name: str
    url: str
    game: str
    bracketMatches: list[BracketMatchOut]
    rosters: dict[str, RosterOut]
    matchCount: int
    sections: list[str] = []


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

# Deterministic team colour palette based on team name hash
TEAM_COLORS = [
    "#ef4444", "#eab308", "#3b82f6", "#f97316", "#a855f7",
    "#22c55e", "#06b6d4", "#ec4899", "#14b8a6", "#f59e0b",
]


def _team_id(name: str) -> str:
    """Generate a short deterministic id from a team name."""
    return hashlib.md5(name.encode()).hexdigest()[:6]


def _team_color(name: str) -> str:
    """Pick a colour from the palette based on team name hash."""
    idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(TEAM_COLORS)
    return TEAM_COLORS[idx]


def _short_name(name: str) -> str:
    """Derive a short display name from a full team name."""
    # Common patterns: "Team Vitality" -> "Vitality", "G2 Esports" -> "G2"
    parts = name.split()
    if len(parts) >= 2 and parts[0].lower() in ("team", "club"):
        return " ".join(parts[1:])
    return parts[0] if parts else name


def _build_team(name: str) -> TeamOut:
    """Create a TeamOut object from a raw team name string."""
    tid = _team_id(name)
    return TeamOut(
        id=tid,
        name=name,
        shortName=_short_name(name),
        logo=_short_name(name),
        color=_team_color(name),
    )


def _match_status(s1: str, s2: str) -> str:
    """Determine bracket match status from score strings."""
    if s1 and s2:
        return "completed"
    return "scheduled"


def _determine_winner(t1_id: str, t2_id: str, s1: str, s2: str) -> Optional[str]:
    """Determine the winner id from scores."""
    if not s1 or not s2:
        return None
    if int(s1) > int(s2):
        return t1_id
    elif int(s2) > int(s1):
        return t2_id
    return None


def _infer_round_index(section: str, round_label: str) -> int:
    """
    Map scraped section/round labels to a numeric round index.
    0 = early rounds, higher = later rounds.
    """
    combined = f"{section} {round_label}".lower()
    if "final" in combined and "semi" not in combined and "quarter" not in combined:
        return 3
    if "semi" in combined:
        return 2
    if "quarter" in combined:
        return 1
    if "round 3" in combined or "round 4" in combined:
        return 1
    return 0


def _build_tournament_data(url: str, df: pd.DataFrame) -> dict:
    """
    Transform the raw scraper DataFrame into the TournamentData shape
    expected by the React frontend.
    """
    # Extract a readable tournament name from the URL slug
    slug_parts = url.rstrip("/").split("/")
    name_parts = [p.replace("_", " ") for p in slug_parts[-3:] if p]
    tournament_name = " — ".join(name_parts) if name_parts else "RLCS Tournament"

    bracket_matches: list[dict] = []
    rosters: dict[str, dict] = {}
    seen_teams: dict[str, TeamOut] = {}
    match_index_counters: dict[int, int] = {}

    for idx, row in df.iterrows():
        t1_name = row.get("team1")
        t2_name = row.get("team2")
        s1 = str(row.get("team1_score", "")).strip()
        s2 = str(row.get("team2_score", "")).strip()
        section = row.get("section", "")
        round_label = row.get("round", "")

        # Build team objects
        team1 = None
        team2 = None
        if t1_name and not pd.isna(t1_name):
            if t1_name not in seen_teams:
                seen_teams[t1_name] = _build_team(t1_name)
            team1 = seen_teams[t1_name]

        if t2_name and not pd.isna(t2_name):
            if t2_name not in seen_teams:
                seen_teams[t2_name] = _build_team(t2_name)
            team2 = seen_teams[t2_name]

        round_idx = _infer_round_index(section, round_label)

        # Track match index per round
        if round_idx not in match_index_counters:
            match_index_counters[round_idx] = 0
        m_idx = match_index_counters[round_idx]
        match_index_counters[round_idx] += 1

        status = _match_status(s1, s2)
        winner_id = None
        if team1 and team2:
            winner_id = _determine_winner(team1.id, team2.id, s1, s2)

        best_of = int(row.get("best_of", 5))

        bracket_matches.append({
            "id": f"m{idx}",
            "matchIndex": m_idx,
            "team1": team1.model_dump() if team1 else None,
            "team2": team2.model_dump() if team2 else None,
            "score1": int(s1) if s1.isdigit() else None,
            "score2": int(s2) if s2.isdigit() else None,
            "winnerId": winner_id,
            "status": status,
            "roundIndex": round_idx,
            "section": section,
            "round": round_label,
            "bestOf": best_of,
        })

        # Build rosters from scraped player lists
        for side, team_obj in [("team1", team1), ("team2", team2)]:
            if not team_obj:
                continue
            if team_obj.id in rosters:
                continue
            players_raw = row.get(f"{side}_players", [])
            if isinstance(players_raw, str):
                try:
                    import ast
                    players_raw = ast.literal_eval(players_raw)
                except Exception:
                    players_raw = []
            if not isinstance(players_raw, list):
                players_raw = []

            active = []
            for pi, pname in enumerate(players_raw[:3]):
                active.append({
                    "id": f"{team_obj.id}-p{pi}",
                    "name": str(pname),
                    "role": "Core Starter",
                })
            subs = []
            for si, sname in enumerate(players_raw[3:]):
                subs.append({
                    "id": f"{team_obj.id}-sub{si}",
                    "name": str(sname),
                    "role": "Substitute",
                })

            rosters[team_obj.id] = {
                "teamId": team_obj.id,
                "active": active,
                "substitutes": subs,
            }

    # Collect unique section names in order of first appearance
    seen_sections: list[str] = []
    for m in bracket_matches:
        s = m["section"]
        if s and s not in seen_sections:
            seen_sections.append(s)

    return {
        "name": tournament_name,
        "url": url,
        "game": "Rocket League",
        "bracketMatches": bracket_matches,
        "rosters": rosters,
        "matchCount": len(bracket_matches),
        "sections": seen_sections,
    }


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Basic health probe."""
    return {"status": "ok", "service": "rlpredictor-api"}


@app.post("/api/scrape", response_model=TournamentResponse)
async def scrape_tournament(req: ScrapeRequest):
    """
    Scrapes a Liquipedia Rocket League tournament page and returns
    structured bracket + roster data for the frontend dashboard.
    """
    url = req.url.strip()

    # Strict RL-only validation
    if not url.lower().startswith(LIQUIPEDIA_RL_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=(
                "RLPredictor exclusively scrapes Rocket League tournaments. "
                f"URLs must start with {LIQUIPEDIA_RL_PREFIX}"
            ),
        )

    try:
        initialize_database()
        df = await scrape_playoffs(url, sections=req.sections, browser=app.state.browser)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Scraping failed: {str(e)}",
        )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No bracket data found on this page. Make sure the URL points to a tournament with visible brackets.",
        )

    return _build_tournament_data(url, df)


@app.post("/api/scrape/light")
async def scrape_tournament_light(req: ScrapeRequest):
    """
    A lighter scrape endpoint that uses requests + BeautifulSoup only
    (no Selenium), for pages that don't require JS rendering.
    Falls back to the full scrape if the light parse yields no results.
    """
    url = req.url.strip()

    if not url.lower().startswith(LIQUIPEDIA_RL_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"URLs must start with {LIQUIPEDIA_RL_PREFIX}",
        )

    # Try a quick requests-based fetch first
    try:
        from scrapers.playoff_scraper import fetch_html, nearest_sect, round_map, get_team_name, is_placeholder
        import requests as req_lib

        soup = fetch_html(url)
        brackets = soup.find_all("div", class_="brkts-bracket")
        if not brackets:
            raise ValueError("No brackets found via light scrape, falling back to Playwright.")

        rows = []
        for b in brackets:
            section = nearest_sect(b)
            rmap = round_map(b)
            for m in b.find_all("div", class_="brkts-match"):
                ops = m.select(".brkts-opponent-entry")
                if len(ops) < 2:
                    continue
                t1 = get_team_name(ops[0])
                t2 = get_team_name(ops[1])
                s1_el = ops[0].select_one(".brkts-opponent-score-inner")
                s2_el = ops[1].select_one(".brkts-opponent-score-inner")
                s1 = s1_el.get_text(strip=True) if s1_el else ""
                s2 = s2_el.get_text(strip=True) if s2_el else ""
                s1 = s1 if s1.isdigit() else ""
                s2 = s2 if s2.isdigit() else ""

                rows.append({
                    "section": section,
                    "round": rmap.get(id(m)) or "",
                    "best_of": 5,
                    "team1": t1,
                    "team2": t2,
                    "team1_score": s1,
                    "team2_score": s2,
                    "team1_players": [],
                    "team2_players": [],
                })

        if not rows:
            raise ValueError("Light scrape found brackets but no matches.")

        df = pd.DataFrame(rows)
        return _build_tournament_data(url, df)

    except Exception as light_err:
        print(f"[LIGHT SCRAPE] Failed: {light_err} — falling back to full Playwright scrape.")
        # Fall through to full scrape
        try:
            initialize_database()
            df = await scrape_playoffs(url, sections=req.sections, browser=app.state.browser)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

        if df.empty:
            raise HTTPException(status_code=404, detail="No bracket data found.")

        return _build_tournament_data(url, df)


class ManualMatchupRequest(BaseModel):
    team1_name: str
    team2_name: str


@app.post("/api/roster/lookup")
async def lookup_rosters(req: ManualMatchupRequest):
    """
    Look up rosters for two teams by name. Uses cached rosters when
    available, otherwise fetches from Liquipedia team pages.
    """
    from scrapers.playoff_scraper import (
        get_team_url,
        fetch_html,
        extract_roster,
        is_placeholder,
    )
    from utils.database import get_cached_roster, cache_roster

    results = {}
    sess = None

    for side, name in [("team1", req.team1_name), ("team2", req.team2_name)]:
        name = name.strip()
        if not name or is_placeholder(name):
            results[side] = {"name": name, "players": [], "error": "Invalid or empty team name"}
            continue

        team_url = get_team_url(name)
        cached = get_cached_roster(team_url)
        if cached is not None:
            results[side] = {"name": name, "players": cached, "source": "cache"}
            continue

        try:
            if sess is None:
                import requests as req_lib
                sess = req_lib.Session()
            ts = fetch_html(team_url, session=sess)
            roster = extract_roster(ts)
            cache_roster(team_url, roster)
            results[side] = {"name": name, "players": roster, "source": "scraped"}
        except Exception as e:
            results[side] = {"name": name, "players": [], "error": str(e)}

    # Build team objects
    t1 = _build_team(req.team1_name.strip()) if req.team1_name.strip() else None
    t2 = _build_team(req.team2_name.strip()) if req.team2_name.strip() else None

    rosters_out = {}
    for side, team_obj in [("team1", t1), ("team2", t2)]:
        if not team_obj:
            continue
        players_raw = results.get(side, {}).get("players", [])
        active = [
            {"id": f"{team_obj.id}-p{i}", "name": str(p), "role": "Core Starter"}
            for i, p in enumerate(players_raw[:3])
        ]
        subs = [
            {"id": f"{team_obj.id}-sub{i}", "name": str(p), "role": "Substitute"}
            for i, p in enumerate(players_raw[3:])
        ]
        rosters_out[team_obj.id] = {
            "teamId": team_obj.id,
            "active": active,
            "substitutes": subs,
        }

    return {
        "team1": t1.model_dump() if t1 else None,
        "team2": t2.model_dump() if t2 else None,
        "rosters": rosters_out,
        "lookupDetails": results,
    }
