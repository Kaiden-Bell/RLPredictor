"""
Author: Kaiden Bell
Date (Coded): (I'll update this part)
File Function:
- Description: Reddit RocketLeagueEsports community sentiment analyzer utilizing NLTK VADER.
- Usage: Imported by chat.py and features.py to calculate public player sentiment metrics.
"""

import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import urllib3


urllib3.disable_warnings()


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


sia = SentimentIntensityAnalyzer()


def get_player_sentiment(player_names: list) -> dict:
    """
    Description:
        Searches Reddit RocketLeagueEsports subreddit posts and analyzes player sentiment.
    Arguments:
        player_names: List of aliases for a single player.
    Returns:
        Dictionary containing average compound score, status, and post count.
    """
    if isinstance(player_names, str): player_names = [player_names]
    
    search_terms = " OR ".join([f'"{name}"' for name in player_names[:3]])
    url = f"https://www.reddit.com/r/RocketLeagueEsports/search.json?q={search_terms}&restrict_sr=1&sort=new&limit=15"
    headers = {"User-Agent": "RLPredictorBot/1.0 by kbell"}
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return {"score": 0.0, "status": "Reddit API Error", "count": 0}
            
        data = r.json()
        posts = data.get("data", {}).get("children", [])
        
        texts = []
        for p in posts:
            pdata = p.get("data", {})
            title = pdata.get("title", "")
            selftext = pdata.get("selftext", "")
            texts.append(title + " " + selftext)
            
        if not texts: return {"score": 0.0, "status": "No recent posts", "count": 0}
            
        total_score = 0
        for t in texts:
            sentiment = sia.polarity_scores(t)
            total_score += sentiment["compound"]
            
        avg_score = total_score / len(texts)
        
        if avg_score > 0.15: status = "Positive"
        elif avg_score < -0.15: status = "Negative"
        else: status = "Neutral"
            
        return {"score": avg_score, "status": status, "count": len(texts)}
        
    except Exception:
        return {"score": 0.0, "status": "Error", "count": 0}
