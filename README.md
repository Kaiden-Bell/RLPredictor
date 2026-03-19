# 🚀 RLPredictor

RLPredictor is an advanced Machine Learning prediction engine for Rocket League Esports. It ingests historical player statistics, recent ranked 2v2 momentum, head-to-head match data, and public sentiment to output high-confidence predictions on whether a player will go **Over or Under** specific statistical thresholds (e.g., Goals, Shots, Saves, Demos) in a given series.

## 🌟 Key Features

- **Neural Network Prediction Engine:** Uses a custom PyTorch Multi-Layer Perceptron (MLP) trained on 13 engineered features to output the raw mathematical probability of a player hitting their over/under target.
- **Automated Data Scraping:** Integrates with Liquipedia to fetch active tournament matchups and rosters.
- **Deep Replay Analysis:** Communicates with the Ballchasing API to pull hundreds of recent RLCS/scrim replays and ranked 2v2 grinds to calculate a player's true baseline over chronological lookbacks.
- **Sentiment Analysis:** Scrapes `/r/RocketLeagueEsports` using NLTK VADER to determine if crowd sentiment and public momentum align with the mathematical data.
- **Conversational CLI:** A natural language chat interface that answers questions like *"Will Zen get over 2.5 demos in 3 games?"* and provides human-readable logic justifying its Neural Net predictions.

## ⚙️ How It Works

1. **Scraping Liquipedia:** Specify a tournament URL, and the core script (`main.py`) will automatically fetch all available matchups and rosters.
2. **Feature Engineering:** Behind the scenes, the predictor generates a 13-dimensional feature array for every player, evaluating variables like:
   - Head-to-Head (H2H) averages against the specific opposing roster.
   - General statistical variance (Standard Deviation) and historical hit rates.
   - Ranked 2v2 grind momentum (matches played, avg score, and win rate over the last 14 days).
   - Scaled Reddit sentiment scores.
3. **Inference & Logic Generation:** The PyTorch model calculates a raw $P(\text{Over})$ expectation. The conversational engine then scales per-game averages over the specified series length, fusing the neural projection with human-readable heuristics to give strong betting advice.
4. **Self-Supervised Learning:** Running `train.py` continuously scales the neural network. The script dynamically constructs training data by rolling through your local `.bc_cache.json` replay history, utilizing previous games to predict subsequent target games without manual labels.

## 🛠️ Usage

Ensure your root `.env` file contains your `BALLCHASING_API_KEY`.

### 1. Main Pipeline & Interface
Boot the core scraper and dive into interactive mode. You can extract Head-to-Head stats, dump feature vectors, or interact with the AI Predictor Chat.
```bash
python main.py "https://liquipedia.net/rocketleague/..." --mode chat
```
*Tip: You can pre-select a matchup using `--match "Team Name"` or parse specific segments of the bracket using `--section playoff`.*

### 2. The Prediction Chat
When using `--mode chat`, the bot will fetch the latest context for the two teams and await your question.
```text
Query: Will LJ get over 4 saves in 3 games?
```
The neural predictor responds with historical Generic hit rates, Head-to-Head data, scaled multi-game projections, Ranked Momentum gradients, Reddit sentiment feedback, and the **AI Probability & Confidence Suggestion**.

### 3. Training the Model
To train the neural network locally across your actively cached Ballchasing archive:
```bash
python train.py --epochs 200 --lr 0.001
```

## 📦 Requirements

- `Python 3.10+`
- `torch`
- `pandas`
- `numpy`
- `nltk`
- `requests`
- `python-dotenv`

Install required dependencies natively:
```bash
pip install -r requirements.txt
```
