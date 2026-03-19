import requests
import nltk
import urllib3

urllib3.disable_warnings()

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # download silently
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_player_sentiment(player_name: str) -> dict:
    url = f"https://www.reddit.com/r/RocketLeagueEsports/search.json?q={player_name}&restrict_sr=1&sort=new&limit=15"
    headers = {"User-Agent": "RLPredictorBot/1.0 by kbell"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return {"score": 0.0, "status": "Reddit API Error", "count": 0}
            
        data = r.json()
        posts = data.get("data", {}).get("children", [])
        
        texts = []
        for p in posts:
            pdata = p.get("data", {})
            title = pdata.get("title", "")
            selftext = pdata.get("selftext", "")
            texts.append(title + " " + selftext)
            
        if not texts:
            return {"score": 0.0, "status": "No recent posts", "count": 0}
            
        total_score = 0
        for t in texts:
            sentiment = sia.polarity_scores(t)
            total_score += sentiment["compound"]
            
        avg_score = total_score / len(texts)
        
        if avg_score > 0.15:
            status = "Positive"
        elif avg_score < -0.15:
            status = "Negative"
        else:
            status = "Neutral"
            
        return {"score": avg_score, "status": status, "count": len(texts)}
        
    except Exception as e:
        return {"score": 0.0, "status": "Error", "count": 0}
