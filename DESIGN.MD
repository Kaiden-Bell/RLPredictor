# RLPredictor — System & Architectural Design

## 1. Vision & Core Objective

**RLPredictor** is an accessible, high-impact Rocket League Esports **Full-Team Match Prediction Engine & Web Application**.

Instead of focusing on niche sports betting lines (e.g. *"Will Zen get Over 2.5 Demos?"*), RLPredictor answers the exact questions every RLCS fan, caster, and viewer asks before a match:
1. **Which team wins the series?**
2. **How long will the series last?** (e.g. a dominant 4–0 sweep or a 7-game thriller?)
3. **If an upset is predicted, WHY does the underdog win?**

By fusing quantitative replay telemetry (kickoff conversions, H2H history), team momentum (ranked 2v2 grind, tournament form), and Reddit community sentiment, RLPredictor outputs data-backed, human-readable match breakdowns that anyone can enjoy.

> [!NOTE]
> For the long-term open-source multi-game framework design (`esports-predictor`), see [FUTUREDESIGN.MD](file:///home/kbell/Documents/Codes/Projects/RLPredictor/FUTUREDESIGN.MD).

---

## 2. User Experience & Application Workflow

### 2.1 Bracket-to-Match UI Flow

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│ 1. Paste Bracket URL   │ ───► │ 2. Select Match        │ ───► │ 3. Match Outcome Card  │
│ (Liquipedia RLCS link) │      │ (Click Match in UI)    │      │ (Winner %, Score, Why) │
└────────────────────────┘      └────────────────────────┘      └───────────┬────────────┘
                                                                            │
                                                                 (Optional Drill-down)
                                                                            ▼
                                                                ┌────────────────────────┐
                                                                │ 4. In-Depth O/U Mode   │
                                                                │ (Player Stat Lines)    │
                                                                └────────────────────────┘
```

1. **Tournament Bracket Scraper**: The user inputs a Liquipedia RLCS event URL (or selects a cached tournament). The app parses the bracket structure and displays an interactive match bracket.
2. **Match Selection**: Clicking any match in the bracket triggers immediate AI prediction.
3. **Match Outcome Card**:
   - **Winner & Confidence**: Projected winning team and win probability (e.g. *Team Vitality 64% vs. Karmine Corp 36%*).
   - **Series Score Forecast**: Exact series score prediction for the match format (e.g. *4–2 in a Bo7*).
   - **Series Duration Distribution**: Monte Carlo probabilities for series outcome categories:
     - Full Sweep (4–0 / 3–0)
     - Dominant (4–1 / 3–1)
     - Competitive (4–2)
     - Game 7 Thriller (4–3)
   - **Upset Alert Badge**: Highlighted whenever an underdog is modeled to win despite historical odds.
   - **3-Pillar "Why They Win" Breakdown**:
     - *Kickoff Conversion & H2H History*: Head-to-head win-rate and goal conversion rate directly off kickoffs in past encounters.
     - *Team Momentum*: Recent tournament performance and 14-day ranked 2v2 grind levels.
     - *Reddit Community Sentiment*: Scraped Reddit match-thread sentiment and fan expectation index.
4. **In-Depth O/U Mode (Optional)**: Users can click *"View In-Depth Player Stats"* to expand detailed player-level Over/Under stat predictions (retained from legacy CLI engine for power users).

---

## 3. Data Telemetry & Scrapers

### 3.1 Data Pipelines
- **Liquipedia Scraper**: Parses tournament bracket structures, active team matchups, rosters, and historical head-to-head match histories.
- **Ballchasing Replay Telemetry**:
  - Extracts H2H replay data for kickoff goal conversions (goals scored within 10 seconds of a kickoff) and initial possession control rates.
  - Queries ranked 2v2 activity for player momentum (games played, win rate, average score over the last 14 days).
- **Reddit Sentiment Scraper**: Uses NLTK VADER sentiment analysis on `/r/RocketLeagueEsports` threads and match discussions to measure public sentiment.

---

## 4. Machine Learning & Series Simulation Engine

### 4.1 Team Matchup Model
- **Input Feature Vector**: 14-dimensional normalized vector comparing Team 1 vs. Team 2:
  - H2H Win Rate Delta & H2H Kickoff Goal Conversion Delta.
  - Recent Tournament Form Delta (Goals/Game, Shooting %, Saves/Game).
  - Combined Ranked 2v2 Momentum Delta (Games played, Win %, Avg score).
  - Reddit Community Sentiment Differential.
- **Classifier**: PyTorch Neural Network / Gradient-boosted Classifier outputting single-game win probability $P(\text{Team 1 Win})$.

### 4.2 Series Simulator (Monte Carlo)
- Runs $N = 10,000$ simulated series using single-game win probability $P$ for Best-of-5 and Best-of-7 formats.
- Outputs exact probability distribution across series scores:
  - **Best of 7**: 4-0, 4-1, 4-2, 4-3, 3-4, 2-4, 1-4, 0-4.
  - **Series Duration**: Sweep % vs. Game 7 Thriller %.
