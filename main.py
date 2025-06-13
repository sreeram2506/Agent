# main.py
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from src.scheduler import NewsToInstagramScheduler

def check_environment():
    """Check that all required environment variables are set."""
    load_dotenv()  # Load .env file if it exists
    
    required_vars = [
        'NEWS_API_KEY',
        'OPENAI_API_KEY',
        'INSTAGRAM_USERNAME',
        'INSTAGRAM_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("Error: The following required environment variables are missing:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease create a .env file with these variables or set them in your environment.")
        return False
    
    return True

def main():
    """Main entry point for the application."""
    if not check_environment():
        sys.exit(1)
    
    # Create assets directory if it doesn't exist
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Check if we should run once (for testing)
    run_once = "--run-once" in sys.argv
    
    # Start the scheduler
    scheduler = NewsToInstagramScheduler()
    scheduler.start_scheduler(run_once=run_once)

if __name__ == "__main__":
    main()