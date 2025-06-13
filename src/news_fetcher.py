# src/news_fetcher.py
import requests
from typing import List, Dict, Any
from config.settings import Settings

class NewsFetcher:
    BASE_URL = "https://newsapi.org/v2/top-headlines"
    
    def __init__(self):
        self.api_key = Settings.NEWS_API_KEY
        self.sources = Settings.NEWS_SOURCES
        print(f"DEBUG: NEWS_API_KEY = {getattr(Settings, 'NEWS_API_KEY', 'Not found')}")
        
    def fetch_top_news(self, count: int = 5) -> List[Dict[str, Any]]:
        """Fetch top news articles from configured sources."""
        if not self.api_key:
            raise ValueError("News API key not configured")
            
        params = {
            "sources": self.sources,
            "apiKey": self.api_key,
            "pageSize": count
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return self._process_articles(articles)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process raw articles to extract relevant information."""
        processed = []
        for article in articles:
            processed.append({
                "title": article.get("title", "No title"),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "image_url": article.get("urlToImage", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", "")
            })
        return processed