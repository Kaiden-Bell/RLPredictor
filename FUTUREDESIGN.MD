# The Esports Prediction Framework (`esports-predictor`) — Future Open-Source Architecture

> [!NOTE]
> This document outlines the long-term vision to refactor **RLPredictor** into a universal, open-source **Esports Prediction Framework**. While the immediate focus is building the Rocket League Match Predictor application ([DESIGN.MD](file:///home/kbell/Documents/Codes/Projects/RLPredictor/DESIGN.MD)), this architecture provides the roadmap for scaling into a multi-game community platform.

---

## 1. Vision & Open-Source Strategy

Most developers don't contribute to open-source projects just to help a random maintainer — they contribute because they think:  
**"I can easily add support for my favorite game!"**

By decoupling the core prediction, simulation, evaluation, and explainability pipelines from game-specific data scraping and feature engineering, `esports-predictor` lowers the barrier to entry for community contributors.

```text
esports-predictor/
├── core/
│   ├── training/        # Game-agnostic PyTorch/ML training loops
│   ├── inference/       # Match inference & Monte Carlo series simulation engine
│   ├── datasets/        # Standardized schema, DB caching, and data pipelines
│   ├── evaluation/      # Backtesting, accuracy tracking, and metric logging
│   └── explainability/  # Natural language narrative & logic generators
│
├── games/
│   ├── rocketleague/    # RocketLeagueAdapter (Ballchasing, Liquipedia, Kickoffs, 2s Momentum)
│   ├── valorant/        # ValorantAdapter (VLR.gg, economy, agent comps)
│   ├── cs2/             # CS2Adapter (HLTV, map vetoes, weapon rounds)
│   ├── dota2/           # Dota2Adapter (OpenDota, hero drafting, net worth)
│   └── lol/             # LeagueOfLegendsAdapter (Gol.gg, gold delta, objectives)
│
└── app/                 # Web UI & FastAPI server
```

---

## 2. Standardized `GameAdapter` Interface

Contributors will be able to add support for any esports title by writing a single game adapter module implementing the `GameAdapter` abstract class:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class GameAdapter(ABC):
    """
    Abstract Base Class for game-specific adapters in the Esports Prediction Framework.
    """

    @abstractmethod
    def scrape_tournament(self, tournament_url: str) -> Dict[str, Any]:
        """Scrapes tournament bracket structure and active matchups."""
        pass

    @abstractmethod
    def build_team_features(self, team1: str, team2: str) -> np.ndarray:
        """Extracts normalized feature vector for a team matchup."""
        pass

    @abstractmethod
    def extract_telemetry_reasons(self, team1: str, team2: str) -> Dict[str, Any]:
        """Returns game-specific reasoning metrics (e.g. Kickoffs for RL, Economy/Vetoes for CS2)."""
        pass

    @abstractmethod
    def get_indepth_player_stats(self, match_id: str) -> List[Dict[str, Any]]:
        """Optional drill-down player statistics for in-depth mode."""
        pass
```

---

## 3. Core Engine Services

1. **Universal Series Simulator**: A game-agnostic Monte Carlo engine that accepts single-game win probability $P$ and simulates Best-of-3, Best-of-5, and Best-of-7 series to generate score distributions and sweep probabilities.
2. **Unified Explainability Pipeline**: Converts game-specific telemetry metrics (e.g. Kickoff conversions in RL, Pistol round win rates in CS2, First Blood % in LoL) into standardized human-readable logic cards.
3. **Automated Backtesting Framework**: Evaluates adapter accuracy over historical game databases and logs validation metrics.
