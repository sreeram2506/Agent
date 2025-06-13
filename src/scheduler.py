# src/scheduler.py
import schedule
import time
from datetime import datetime
import os
import sys
from typing import Optional
from pathlib import Path
from src.image_generator import ImageGenerator

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.news_fetcher import NewsFetcher
from src.content_processor import ContentProcessor
from src.instagram_poster import InstagramPoster
from config.settings import Settings

class NewsToInstagramScheduler:
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.content_processor = ContentProcessor()
        self.image_generator = ImageGenerator()
        # self.instagram_poster = InstagramPoster()
        self.logo_path = '../assets/logo/shared image.jpeg'
        self.posted_articles = set()  # Simple in-memory storage of posted articles
        
    def _get_article_key(self, article: dict) -> str:
        """Generate a unique key for an article to avoid duplicates."""
        return f"{article.get('source', '')}_{article.get('title', '')}"
        
    def create_and_post(self, image_path: Optional[str] = '../assets/news_thumbnail.jpg'):
        """Fetch news, process, create image, and post to Instagram."""
        print(f"\n{datetime.now()}: Running scheduled post creation...")
        
        try:
            # Fetch latest news
            articles = self.news_fetcher.fetch_top_news(count=5)  # Get 5, we'll pick the first new one
            
            if not articles:
                print("No articles found to post.")
                return
                
            # Find first article that hasn't been posted yet
            article_to_post = None
            for article in articles:
                article_key = self._get_article_key(article)
                if article_key not in self.posted_articles:
                    article_to_post = article
                    self.posted_articles.add(article_key)
                    break
                    
            if not article_to_post:
                print("No new articles to post.")
                return
                
            print(f"Processing article: {article_to_post['title'][:50]}...")
            
            # Generate content
            article_to_post['category'] = 'general'
            caption = self.content_processor.generate_instagram_caption(article_to_post)
            print("Generated caption.", article_to_post)
            
            # Create image
            image_path = self.image_generator.create_news_thumbnail(article_to_post, image_path)
            print("Created image.", image_path)
            if not image_path or not os.path.exists(image_path):
                print("Failed to create image.")
                return
            print(f"Created image: {image_path}")
            
            # Post to Instagram
            print("Posting to Instagram...")
            # success = self.instagram_poster.post_to_instagram(image_path, caption)
            # if success:
            #     print("Successfully posted to Instagram!")
            # else:
            #     print("Failed to post to Instagram.")
                # Remove from posted articles so we can retry
                # self.posted_articles.discard(self._get_article_key(article_to_post))
                
        except Exception as e:
            print(f"Error in create_and_post: {e}")
            import traceback
            traceback.print_exc()
            
    def start_scheduler(self, run_once: bool = False):
        """Start the scheduling loop.
        
        Args:
            run_once: If True, run the job once and exit. Useful for testing.
        """
        if run_once:
            print("Running once...")
            self.create_and_post(image_path='../assets/news_thumbnail.jpg')
            return
            
        # Schedule the job
        schedule.every(Settings.POST_FREQUENCY_HOURS).hours.do(self.create_and_post)
        
        # Run immediately the first time
        self.create_and_post(image_path=None)
        
        # Then run on schedule
        print(f"Scheduler started. Will post every {Settings.POST_FREQUENCY_HOURS} hours.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")