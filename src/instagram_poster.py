# src/instagram_poster.py
import os
import time
import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ChallengeRequired, ClientError
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstagramPoster:
    def __init__(self):
        self.client = Client()
        self.username = Settings.INSTAGRAM_USERNAME
        self.password = Settings.INSTAGRAM_PASSWORD
        self.assets_dir = Path(Settings.ASSETS_DIR)
        self.session_file = self.assets_dir / "instagram_session.json"
        self.analytics_file = self.assets_dir / "instagram_analytics.json"
        
        # Initialize client settings
        self.client.delay_range = [1, 3]  # More natural delay between actions
        self.client.request_timeout = 30  # Increased timeout
        
        # Load analytics
        self.analytics = self._load_analytics()
        
        # Create assets directory if it doesn't exist
        self.assets_dir.mkdir(exist_ok=True)
    
    def _load_analytics(self) -> Dict:
        """Load analytics data from file."""
        try:
            if self.analytics_file.exists():
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading analytics: {e}")
        return {
            "posts": [],
            "best_posting_times": [],
            "engagement_rates": [],
            "hashtag_performance": {}
        }
    
    def _save_analytics(self) -> None:
        """Save analytics data to file."""
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.analytics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving analytics: {e}")

    def _update_analytics(self, post_data: Dict) -> None:
        """Update analytics with new post data."""
        self.analytics["posts"].append({
            "timestamp": datetime.now().isoformat(),
            "post_id": post_data.get("pk", ""),
            "like_count": post_data.get("like_count", 0),
            "comment_count": post_data.get("comment_count", 0),
            "caption": post_data.get("caption_text", "")[:100],
            "hashtags": self._extract_hashtags(post_data.get("caption_text", ""))
        })
        self._calculate_engagement_rate()
        self._save_analytics()
    
    def _extract_hashtags(self, caption: str) -> List[str]:
        """Extract hashtags from caption."""
        import re
        return re.findall(r'#(\w+)', caption.lower())
    
    def _calculate_engagement_rate(self) -> None:
        """Calculate engagement rate from recent posts."""
        recent_posts = [p for p in self.analytics["posts"] 
                       if datetime.fromisoformat(p["timestamp"]) > datetime.now() - timedelta(days=30)]
        
        if not recent_posts:
            return
            
        total_engagement = sum(p.get("like_count", 0) + p.get("comment_count", 0) 
                              for p in recent_posts)
        avg_engagement = total_engagement / len(recent_posts)
        self.analytics["engagement_rates"].append({
            "date": datetime.now().isoformat(),
            "rate": avg_engagement
        })
        
    def _get_optimal_posting_time(self) -> datetime:
        """Determine the best time to post based on analytics."""
        if not self.analytics.get("best_posting_times"):
            # Default to business hours if no analytics
            hour = random.choice([9, 12, 15, 18, 20])  # Common engagement hours
        else:
            # Find hour with most engagement
            hours = [datetime.fromisoformat(t).hour 
                    for t in self.analytics["best_posting_times"]]
            hour = max(set(hours), key=hours.count) if hours else 12
            
        # Schedule for next occurrence of best hour
        now = datetime.now()
        post_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if post_time < now:
            post_time += timedelta(days=1)
            
        return post_time

    def login(self) -> bool:
        """Login to Instagram with session management and enhanced error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Configure client settings
                self.client.set_locale('en_US')
                self.client.set_country(0)  # 0 for international
                self.client.set_country_code(91)  # India
                
                # Try to load previous session
                if self.session_file.exists():
                    try:
                        self.client.load_settings(self.session_file)
                        # Test if session is still valid
                        self.client.get_timeline_feed()
                        logger.info("Successfully logged in using saved session")
                        return True
                    except (LoginRequired, ChallengeRequired) as e:
                        logger.warning(f"Session expired: {e}")
                        os.remove(self.session_file)  # Remove invalid session
                
                # New login required
                logger.info(f"Attempting login as {self.username} (attempt {attempt + 1}/{max_retries})")
                time.sleep(random.uniform(2, 5))  # Human-like delay
                
                # Login with 2FA support
                login_result = self.client.login(
                    username=self.username,
                    password=self.password,
                    verification_code=input("Enter 2FA code (if enabled): ") if attempt > 0 else None
                )
                
                if login_result:
                    self.client.dump_settings(self.session_file)
                    logger.info("Successfully logged in and saved session")
                    return True
                    
            except ChallengeRequired as e:
                logger.warning(f"Challenge required: {e}")
                try:
                    challenge_code = input("Enter challenge code from email/SMS: ")
                    self.client.challenge_resolve(challenge_code)
                    self.client.dump_settings(self.session_file)
                    return True
                except Exception as challenge_error:
                    logger.error(f"Challenge resolution failed: {challenge_error}")
            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("Max login attempts reached")
                    return False
                time.sleep(min(2 ** attempt, 60))  # Exponential backoff
        
        return False
    
    def _optimize_caption(self, caption: str, category: str = None) -> str:
        """Optimize caption for better engagement."""
        # Ensure caption is within limits
        max_length = 2200
        if len(caption) > max_length:
            truncated = caption[:max_length-3].rsplit('\n', 1)[0] + '...'
            if len(truncated) < max_length * 0.8:
                truncated = caption[:max_length-3] + '...'
            caption = truncated
        
        # Add relevant hashtags if not present
        hashtags = self._generate_hashtags(caption, category)
        if hashtags and not any(tag in caption for tag in hashtags):
            caption += f"\n\n{hashtags}"
            
        return caption
    
    def _generate_hashtags(self, caption: str, category: str = None) -> str:
        """Generate relevant hashtags based on content and category."""
        # Category-based hashtags
        category_hashtags = {
            'tech': ['#TechNews', '#DigitalWorld', '#Innovation', '#FutureTech'],
            'business': ['#BusinessNews', '#StartupLife', '#Entrepreneurship'],
            'general': ['#NewsUpdate', '#StayInformed', '#DailyNews']
        }
        
        # Content-based hashtags
        content = caption.lower()
        tags = []
        
        # Add category tags
        if category and category in category_hashtags:
            tags.extend(category_hashtags[category])
        
        # Add trending tags
        tags.extend(['#BreakingNews', '#Viral', '#Trending'])
        
        # Ensure unique and limit to 20 hashtags
        return ' '.join(list(dict.fromkeys(tags))[:20])

    def post_to_instagram(
        self, 
        image_path: str, 
        caption: str,
        category: str = None,
        schedule: bool = False
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Post an image to Instagram with enhanced features.
        
        Args:
            image_path: Path to the image file
            caption: Post caption
            category: Content category for hashtag optimization
            schedule: Whether to schedule for optimal time
            
        Returns:
            Tuple of (success: bool, post_data: Optional[Dict])
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False, None
            
        try:
            # Ensure we're logged in
            if not self.client.user_id and not self.login():
                return False, None
            
            # Optimize caption
            caption = self._optimize_caption(caption, category)
            
            # Schedule post if requested
            if schedule:
                post_time = self._get_optimal_posting_time()
                logger.info(f"Optimal posting time: {post_time.strftime('%Y-%m-%d %H:%M')}")
                # Note: Actual scheduling would require Instagram Business API
                # For now, we'll proceed with immediate posting
            
            # Upload the photo with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    media = self.client.photo_upload(
                        path=image_path,
                        caption=caption[:2200],
                        extra_data={
                            "custom_accessibility_caption": "News update",
                            "disable_comments": False,
                        }
                    )
                    
                    # Update analytics
                    self._update_analytics(media.dict())
                    logger.info(f"Successfully posted to Instagram: {media.pk}")
                    return True, media.dict()
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(5)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Error posting to Instagram: {e}")
            # Try to re-login once on failure
            try:
                logger.info("Attempting to re-login...")
                if self.login():
                    return self.post_to_instagram(image_path, caption, category, schedule)
            except Exception as retry_error:
                logger.error(f"Re-login failed: {retry_error}")
        
        return False, None