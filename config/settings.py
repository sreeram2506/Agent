# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # News API
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    NEWS_SOURCES = "techcrunch,engadget,wired,bbc-news,cnn"

    # Hugging Face
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Instagram
    INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME")
    INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD")
    
    # App Settings
    POST_FREQUENCY_HOURS = 6  # Post every 6 hours
    IMAGE_SIZE = (1080, 1080)  # Instagram square post size
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Image Generation
    IMAGE_EFFECTS = {
        'shadow': True,
        'stroke': True,
        'rounded_corners': True,
        'blur_radius': 3,
        'opacity': 0.7
    }
    
    # Font Settings
    FONT_PATHS = {
        'regular': str(BASE_DIR / 'assets/fonts/Montserrat-Regular.ttf'),
        'bold': str(BASE_DIR / 'assets/fonts/Montserrat-Bold.ttf'),
        'light': str(BASE_DIR / 'assets/fonts/Montserrat-Light.ttf'),
    }
    
    # Color Palettes
    COLOR_PALETTES = {
        'tech': ['#2c3e50', '#3498db', '#1abc9c', '#e74c3c'],
        'business': ['#2c3e50', '#9b59b6', '#3498db', '#e74c3c'],
        'general': ['#2c3e50', '#e67e22', '#f1c40f', '#e74c3c']
    }
    
    # Instagram Posting
    OPTIMAL_POSTING_HOURS = [9, 12, 15, 18, 20]  # 9AM, 12PM, 3PM, 6PM, 8PM
    MAX_HASHTAGS = 20
    CAPTION_MAX_LENGTH = 2200
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create necessary directories
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        os.makedirs(cls.ASSETS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.ASSETS_DIR, 'fonts'), exist_ok=True)
        os.makedirs(os.path.join(cls.ASSETS_DIR, 'images'), exist_ok=True)

# Initialize settings
settings = Settings()
settings.setup_directories()