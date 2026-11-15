"""Paths and environment configuration."""

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

import os
import re



ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("RLPREDICTOR_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("RLPREDICTOR_DB", DATA_DIR / "predictor.db"))
MODEL_PATH = Path(os.environ.get("RLPREDICTOR_MODEL", DATA_DIR / "model.joblib"))

LIQUIPEDIA_API = "https://liquipedia.net/rocketleague/api.php"
BALLCHASING_API = "https://ballchasing.com/api"

# Liquipedia API terms: at most 1 request / 2 s, gzip.
LIQUIPEDIA_MIN_INTERVAL = 2.1
# Ballchasing regular accounts: 2 calls/s, 500/hour.
BALLCHASING_MIN_INTERVAL = 0.6

# Liquipedia page-title prefixes ingested per season. The parser is generic match2, so
# other circuits can be added here.
# No trailing slash: the season root page itself ("Rocket League Championship Series/2026")
# is the World Championship.
DEFAULT_SEASON_PREFIXES = [
    "Rocket League Championship Series/{season}",
    "Esports World Cup/{season}",
    "Rocket League Championship Series/Club Championship/{season}",
    "NEO/Coupe de France Slash/{season}",
    "Webedia/Coupe de France Slash/{season}",
]
# Season roots that are an event of their own (Worlds) rather than the parent event of
# their subpages: "…/2026/Additional Content" must not inherit Worlds as its event root.
STANDALONE_ROOTS = re.compile(r"^Rocket League Championship Series/\d{4}$")

ATTRIBUTION = (
    "Match data from Liquipedia (liquipedia.net/rocketleague), licensed CC-BY-SA 3.0. "
    "Player stats from ballchasing.com. Community sentiment from r/RocketLeagueEsports."
)

class ConfigError(RuntimeError):
    pass

def liquipedia_user_agent() -> str:
    ua = os.environ.get("LIQUIPEDIA_USER_AGENT", "").strip()
    if not ua:
        raise ConfigError(
            "LIQUIPEDIA_USER_AGENT is not set. Liquipedia requires a descriptive User-Agent "
            "with contact info, e.g.\n"
            '  LIQUIPEDIA_USER_AGENT="RLPredictor/0.2 (you@example.com)"\n'
            "Add it to .env in the project root."
        )
    return ua

def ballchasing_api_key() -> str | None:
    return os.environ.get("BALLCHASING_API_KEY", "").strip() or None


def reddit_credentials() -> tuple[str, str, str]:
    cid = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    sec = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    ua = os.environ.get("REDDIT_USER_AGENT", "").strip()
    if not (cid and sec and ua):
        raise ConfigError(
            "Reddit sentiment needs REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT.\n"
            "Create a 'script' app at https://www.reddit.com/prefs/apps and add them to .env, e.g.\n"
            '  REDDIT_USER_AGENT="script:rlpredictor:0.2 (by /u/your_username)"')
    return cid, sec, ua
