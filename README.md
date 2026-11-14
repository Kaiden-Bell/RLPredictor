# RLPredictor

Explainable match-outcome predictions for Rocket League esports.

Ask "Vitality vs FURIA, Bo7?" and get the winner, the probability, the likely score, and **why**: each factor's contribution in percentage points, with the evidence behind it.

```
$ rlpredict predict Vitality Furia --bo 7

Team Vitality  vs  FURIA   (Bo7, as of 2026-11-14)
  Team Vitality: zen, ExoTiiK, stizzy  [Esports World Cup/2026]
  FURIA: yANXNZ, Lostt., swiftt.  [Esports World Cup/2026]

Prediction: Team Vitality wins  (88%)   most likely 4-1
Series length: sweep 25%  dominant 31%  competitive 25%  decider 18%

Why:
  • Rating            +33.1 pts → Team Vitality
      Team Vitality rated 2118 vs FURIA 1970 (roster-weighted Elo); cross-region matchup
  • Head-to-head      +1.9 pts → Team Vitality
      Team Vitality 1–0 in series (3–1 in games) over the last 12 months; last met Aug 2026
  • Roster stability  +0.5 pts → Team Vitality
      Team Vitality: current trio has 70 series together; FURIA: current trio has 54 series together
```

## How it works

- **Data:** RLCS and Esports World Cup results, per-game scores and event rosters come from the **Liquipedia MediaWiki API**. It's keyless and allowed under their API terms; rendered pages are never scraped. Optional per-player series stats come from **Ballchasing**, using the group id Liquipedia links on every series.
- **Ratings:** player-level Elo, updated per game. A team's strength is the mean of its three listed players, so roster changes count immediately.
- **Model:** logistic regression on game outcomes, using team1 − team2 differences in rating, head-to-head, recent form, roster stability and player value.
  - Factors are only kept if they improve the out-of-sample backtest.
  - Per-game probability converts exactly to a Bo3/5/7 score distribution.
- **No leakage:** features are computed by replaying history in date order. A test checks that changing future results never changes past features.

Walk-forward backtest (train on everything before each month, predict that month) on 15,480 RLCS series from 2024–26:

| Model | Log-loss | Brier | Accuracy |
|---|---|---|---|
| Coin flip | 0.693 | 0.250 | 50.0% |
| Elo only | 0.520 | 0.175 | 72.8% |
| **Full model** | **0.518** | **0.174** | **73.2%** |

Run `rlpredict backtest` for the current numbers and per-factor ablations. See [DESIGN.MD](DESIGN.MD) for the full design.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"      # or: python -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                        # set LIQUIPEDIA_USER_AGENT (with your contact info)
```

| Variable | Required | Purpose |
|---|---|---|
| `LIQUIPEDIA_USER_AGENT` | yes | Liquipedia's API terms require a descriptive User-Agent with contact info |
| `BALLCHASING_API_KEY` | no | Turns on the player-value factor (token from ballchasing.com/upload) |

## Usage

```bash
rlpredict ingest                      # RLCS 2024–2026 + EWC from Liquipedia (~15 min first run, incremental after)
rlpredict enrich --budget 400         # optional: Ballchasing player stats (resumable; ~500 calls/hour limit)
rlpredict train                       # backtest, select factors, fit, save data/model.joblib
rlpredict predict Vitality Furia --bo 7
rlpredict predict "Karmine Corp" "Team Falcons" --roster1 "Vatira,Atow.,zen"   # what-if lineup
rlpredict event "<Liquipedia event page title or URL>"   # predictions for upcoming matches
rlpredict teams vit                   # search team names
rlpredict status
rlpredict serve                       # JSON API on http://127.0.0.1:8000
```

### API

| Route | Description |
|---|---|
| `GET /health` | Liveness, plus whether a trained model exists |
| `GET /teams?q=` | Team search |
| `POST /predict` | Body `{team1, team2, best_of, roster1?, roster2?, on?}` → winner, probabilities, score distribution, factors |
| `GET /events/upcoming?page=` | Predictions for unfinished matches on an event page |
| `GET /model` | Training window, active factors, weights, backtest metrics |

### Tests

```bash
pytest
```

## Data attribution

Match data comes from [Liquipedia](https://liquipedia.net/rocketleague), licensed [CC-BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Player stats come from [ballchasing.com](https://ballchasing.com). Every prediction includes this attribution.

The original over/under prop predictor is archived on the `legacy/ou-predictor` branch.
