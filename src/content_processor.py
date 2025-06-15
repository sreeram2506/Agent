import logging
import requests
from typing import Dict
from config.settings import Settings

logger = logging.getLogger(__name__)

# Choose a suitable model from Hugging Face (instruction-tuned)
HF_MODEL_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HEADERS = {"Authorization": f"Bearer {Settings.HUGGINGFACE_API_KEY}"}


class ContentProcessor:
    def __init__(self):
        self.api_url = HF_MODEL_URL
        self.headers = HEADERS

    def _query_model(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            generated = response.json()
            print("Generated", generated)

            # Some models return a list of dicts; others return a string
            if isinstance(generated, list):
                return generated[0]["generated_text"].strip()
            elif isinstance(generated, dict) and "generated_text" in generated:
                return generated["generated_text"].strip()
            else:
                logger.warning(f"Unexpected response format: {generated}")
                return "⚠️ Could not parse model response."
        except Exception as e:
            logger.error(f"Error calling Hugging Face model: {e}")
            return "⚠️ Error generating content."

    def generate_instagram_caption(self, article: Dict, tone: str = "educational") -> str:
        title = article.get("title", "Untitled")
        description = article.get("description", "")
        source = article.get("source", "NewsSource").lower().replace(" ", "")
        print(f"Article: {article}")
        if not title:
            return "Check out this breaking news! 🗞️ #news #update"

        if tone == "educational":
            prompt = (
                "You're an educational Instagram content creator who simplifies current news.\n"
                f"Title: {title}\n"
                f"Description: {description}\n\n"
                "Write a caption with a hook, 1–2 lines of context, and one key insight. End with a question or call to action. Use light emojis and 5–7 educational hashtags."
            )
        else:
            prompt = (
                "You're a social media expert.\n"
                f"Title: {title}\n"
                f"Description: {description}\n\n"
                "Write an engaging Instagram caption with a bold hook, a few emojis, and 5–7 relevant hashtags. Max 2200 characters."
            )

        result = self._query_model(prompt, max_tokens=200, temperature=0.7)
        return result or f"{title}\n\n#news #update #{source}"

    def summarize_article(self, article: Dict) -> str:
        title = article.get("title", "Untitled")
        description = article.get("description", "")
        if not description:
            return title

        prompt = (
            f"Title: {title}\n"
            f"Description: {description}\n\n"
            "Summarize this news article in 2–3 short, clear sentences."
        )

        return self._query_model(prompt, max_tokens=100, temperature=0.3)
