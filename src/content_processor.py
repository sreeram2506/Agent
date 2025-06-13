# src/content_processor.py

from typing import Dict
import logging
import openai
from config.settings import Settings

logger = logging.getLogger(__name__)

class ContentProcessor:
    def __init__(self):
        openai.api_key = Settings.OPENAI_API_KEY

    def generate_instagram_caption(self, article: Dict, tone: str = "educational") -> str:
        """
        Generate an Instagram caption for a news article based on the specified tone.
        Supported tones: 'educational', 'engaging' (default: 'educational')
        """
        title = article.get("title")
        description = article.get("description", "")
        source = article.get("source", "NewsSource").lower().replace(" ", "")
        print(f"Article: {article}")
        if not title:
            return "Check out this breaking news! 🗞️ #news #update"

        # Tone-specific prompt design
        if tone == "educational":
            prompt = (
                "You're an educational content creator for Instagram who breaks down current events clearly.\n"
                "Create an informative, easy-to-digest caption about this news article.\n"
                "Start with a clear hook (e.g., surprising fact or bold statement).\n"
                "Explain the context in 1–2 short sentences.\n"
                "Add one key insight to help people understand *why this matters*.\n"
                "Close with a reflective question or CTA.\n"
                "Use emojis strategically (lightly) and include 5–7 educational/relevant hashtags.\n"
                "Keep it under 2,200 characters.\n\n"
                f"Title: {title}\n"
                f"Description: {description}\n\n"
                "Instagram Caption:"
            )
            system_role = "You are an educational content creator specializing in simplifying news topics."
        else:
            prompt = (
                "You're a social media expert. Create an engaging Instagram caption for this news article.\n"
                "Include a bold hook, relevant emojis, and 5–7 hashtags.\n"
                "Keep it under 2,200 characters.\n\n"
                f"Title: {title}\n"
                f"Description: {description}\n\n"
                "Caption:"
            )
            system_role = "You are a social media expert who crafts engaging Instagram captions."

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"{title}\n\n#news #update #{source}"

    def summarize_article(self, article: Dict) -> str:
        """
        Generate a concise 2–3 sentence summary of the article.
        """
        title = article.get("title", "Untitled")
        description = article.get("description", "")

        if not description:
            return title

        prompt = (
            "Summarize this news article in 2–3 short, clear sentences:\n\n"
            f"Title: {title}\n"
            f"Content: {description}\n\n"
            "Summary:"
        )

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes news articles concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            logger.error(f"Error summarizing article: {e}")
            return (description[:200] + "...") if description else "Summary not available."
