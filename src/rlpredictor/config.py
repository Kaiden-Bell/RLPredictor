"""Paths and environment configuration."""

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

import os



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
DEFAULT_SEASON_PREFIXES = [
    "Rocket League Championship Series/{season}/",
    "Esports World Cup/{season}",
]

ATTRIBUTION = (
    "Match data from Liquipedia (liquipedia.net/rocketleague), licensed CC-BY-SA 3.0. "
    "Player stats from ballchasing.com."
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
