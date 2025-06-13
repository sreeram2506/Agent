import os
import io
import requests
import logging
import hashlib
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
from typing import Optional, List, Tuple, Union
from io import BytesIO
from urllib.parse import urlparse
import mimetypes
from dataclasses import dataclass
from config.settings import Settings

@dataclass
class ThumbnailSettings:
    title: str
    subtitle: str = ""
    category: str = "general"
    image_url: str = ""
    prompt: str = ""
    logo_path: str = ""
    width: int = 1080
    height: int = 1080

class ImageGenerator:
    def __init__(self, logo_path: str = ""):
        self.image_size = Settings.IMAGE_SIZE
        self.assets_dir = Settings.ASSETS_DIR
        self.fonts_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fonts')
        self.effects = Settings.IMAGE_EFFECTS
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.fonts_dir, exist_ok=True)
        self.logo_path = logo_path

        # Modern color palettes
        self.palettes = {
            'tech': ['#0a192f', '#64ffda', '#00ddeb', '#ff6b6b'],  # Deep navy, cyan, red
            'business': ['#1a1a1a', '#f4c430', '#0077b6', '#ff4757'],  # Black, gold, blue
            'general': ['#2d3436', '#a8e6ce', '#ff7675', '#d8a7b1']  # Dark gray, mint, coral
        }

    def _generate_ai_image(self, prompt: str) -> Optional[Image.Image]:
        """Generate an image using Stability AI API"""
        try:
            from stability_sdk import client
            import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

            stability_api = client.StabilityInference(
                key=os.getenv('STABILITY_KEY'),  # Get API key from environment
                verbose=True,
            )

            answers = stability_api.generate(
                prompt=prompt,
                width=1024,
                height=1024,
                samples=1,
            )

            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        print("Content warning, try a different prompt")
                        continue
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        return Image.open(io.BytesIO(artifact.binary))
                        
        except Exception as e:
            print(f"Error generating image via API: {e}")
            import traceback
            traceback.print_exc()
        
        return None

    def _download_image(self, url: str, prompt: str = None) -> Optional[Image.Image]:
        """Download an image from URL or generate one using AI if URL is not provided."""
        if not url and prompt:
            print("Generating AI image...")
            return self._generate_ai_image(prompt)
            
        if not url:
            return None
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                parsed = urlparse(url)
                if not mimetypes.guess_type(parsed.path)[0] or 'image' not in mimetypes.guess_type(parsed.path)[0]:
                    logging.warning(f"URL does not point to an image: {url}")
                    return None
            
            image = Image.open(BytesIO(response.content))
            
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image from {url}: {e}")
        except Exception as e:
            logging.error(f"Error processing image from {url}: {e}")
            
        return None

    def _get_font(self, size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font with the specified size and style."""
        try:
            if bold and italic:
                font_path = os.path.join(self.fonts_dir, 'Montserrat-BoldItalic.ttf')
            elif bold:
                font_path = os.path.join(self.fonts_dir, 'Montserrat-Bold.ttf')
            elif italic:
                font_path = os.path.join(self.fonts_dir, 'Montserrat-Italic.ttf')
            else:
                font_path = os.path.join(self.fonts_dir, 'Montserrat-Regular.ttf')
                
            return ImageFont.truetype(font_path, size)
        except IOError:
            # Fallback to default font if custom font not found
            return ImageFont.load_default()

    def _add_text_with_effects(self, draw, text: str, position: Tuple[int, int], 
                             font: ImageFont.FreeTypeFont, fill: str, 
                             stroke: Optional[Tuple[int, str]] = None, 
                             shadow: bool = True, anchor: str = None,
                             gradient: Optional[Tuple[str, str]] = None) -> None:
        """Draw text with optional effects like shadow, stroke, and gradient."""
        text = unicodedata.normalize('NFKD', str(text)).encode('latin-1', 'ignore').decode('latin-1')
        x, y = position
        
        # Create a temporary image for text with effects
        temp_img = Image.new('RGBA', (draw.im.size[0], draw.im.size[1] * 2), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Add shadow effect
        if shadow:
            shadow_offset = 3
            for ox, oy in [(shadow_offset, shadow_offset), (-shadow_offset, shadow_offset), 
                          (shadow_offset, -shadow_offset), (-shadow_offset, -shadow_offset)]:
                temp_draw.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0, 100), anchor=anchor)
        
        # Add stroke effect
        if stroke:
            stroke_width, stroke_color = stroke
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        temp_draw.text((x + dx, y + dy), text, font=font, fill=stroke_color, anchor=anchor)
        
        # Add gradient text
        if gradient:
            from_col, to_col = gradient
            # Create gradient mask
            width, height = draw.textsize(text, font=font)
            gradient = Image.new('RGBA', (width, height))
            for x_px in range(width):
                ratio = x_px / width
                r = int((1 - ratio) * int(from_col[1:3], 16) + ratio * int(to_col[1:3], 16))
                g = int((1 - ratio) * int(from_col[3:5], 16) + ratio * int(to_col[3:5], 16))
                b = int((1 - ratio) * int(from_col[5:7], 16) + ratio * int(to_col[5:7], 16))
                for y_px in range(height):
                    gradient.putpixel((x_px, y_px), (r, g, b, 255))
            
            # Create text mask
            text_mask = Image.new('L', (width, height), 0)
            text_draw = ImageDraw.Draw(text_mask)
            text_draw.text((0, 0), text, font=font, fill=255)
            
            # Apply gradient to text
            gradient.putalpha(text_mask)
            
            # Paste the gradient text
            temp_img.paste(gradient, (x - width//2 if anchor == 'mm' else x, 
                                    y - height//2 if anchor == 'mm' else y), 
                          gradient)
        else:
            temp_draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
        
        # Composite the text with effects onto the original image
        draw.bitmap((0, 0), temp_img.split()[-1].point(lambda p: p > 0 and 255), fill=None)

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
        """Wrap text to fit within max_width."""
        lines = []
        words = text.split()
        
        while words:
            line = []
            while words:
                test_line = ' '.join(line + [words[0]])
                if font.getlength(test_line) <= max_width:
                    line.append(words.pop(0))
                else:
                    break
            lines.append(' '.join(line))
            
        return '\n'.join(lines)

    def _create_gradient_mask(self, size: Tuple[int, int], direction: str = 'bottom', 
                            start_alpha: int = 200, end_alpha: int = 0) -> Image.Image:
        """Create a gradient mask for overlay effects."""
        width, height = size
        gradient = Image.new('L', (1, height), color=0)
        
        for y in range(height):
            if direction == 'bottom':
                alpha = int(start_alpha + (end_alpha - start_alpha) * (y / height))
            else:  # top
                alpha = int(end_alpha + (start_alpha - end_alpha) * (y / height))
            gradient.putpixel((0, y), alpha)
            
        return gradient.resize((width, height))

    def _add_watermark(self, image: Image.Image, logo_path: str) -> Image.Image:
        """Add a watermark logo to the image."""
        try:
            logo = Image.open(logo_path).convert('RGBA')
            # Resize logo to be 10% of the image width
            logo_size = int(image.width * 0.1)
            logo.thumbnail((logo_size, logo_size), Image.Resampling.LANCZOS)
            
            # Create a transparent layer for the logo
            logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            logo_layer.paste(logo, (image.width - logo.width - 20, 20), logo)
            
            # Composite the logo onto the image
            return Image.alpha_composite(image.convert('RGBA'), logo_layer)
        except Exception as e:
            logging.error(f"Error adding watermark: {e}")
            return image

    def create_news_thumbnail(self, article: dict, image_path: Optional[str] = None) -> Optional[Image.Image]:
        """Create a news thumbnail with the given settings."""
        print(f"Creating thumbnail for {article}")
        try:
            # Create base image with background color
            palette = self.palettes.get(article['category'].lower(), self.palettes['general'])
            bg_color = palette[0]
            primary_color = palette[1]
            accent_color = palette[2]
            image = Image.new('RGB', self.image_size, bg_color)
            
            draw = ImageDraw.Draw(image)
            
            # Download or generate the main image
            main_image = self._download_image(article['image_url'], article['title'])
            
            if main_image:
                # Resize and position the main image
                main_image = ImageOps.fit(main_image, self.image_size, 
                                        method=Image.Resampling.LANCZOS)
                
                # Apply effects
                if 'blur' in self.effects:
                    main_image = main_image.filter(ImageFilter.GaussianBlur(2))
                if 'contrast' in self.effects:
                    enhancer = ImageEnhance.Contrast(main_image)
                    main_image = enhancer.enhance(1.2)
                
                # Create overlay
                overlay = Image.new('RGBA', self.image_size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # Add gradient overlay
                gradient = self._create_gradient_mask((self.image_size[0], self.image_size[1]//2), 
                                                     'bottom', 150, 0)
                overlay.paste((0, 0, 0), (0, self.image_size[1]//2), gradient)
                
                # Composite the image and overlay
                image.paste(main_image, (0, 0))
                image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(image)
            
            # Add title
            title_font = self._get_font(72, bold=True)
            title = self._wrap_text(article['title'], title_font, self.image_size[0] - 100)
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_height = title_bbox[3] - title_bbox[1]
            
            # Add subtitle if exists
            subtitle_y = self.image_size[1] - 100
            if article.get('subtitle'):
                subtitle_font = self._get_font(36)
                subtitle = self._wrap_text(article['subtitle'], subtitle_font, self.image_size[0] - 200)
                subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
                subtitle_height = subtitle_bbox[3] - subtitle_bbox[1]
                subtitle_y = self.image_size[1] - 100 - subtitle_height - 20
                
                # Draw subtitle with effects
                self._add_text_with_effects(
                    draw, subtitle, 
                    (self.image_size[0] // 2, subtitle_y),
                    subtitle_font, 
                    fill=primary_color,
                    stroke=(2, bg_color),
                    anchor='mm'
                )
            
            # Draw title with effects
            self._add_text_with_effects(
                draw, title,
                (self.image_size[0] // 2, subtitle_y - 30),
                title_font,
                fill='#ffffff',
                stroke=(3, bg_color),
                anchor='mm'
            )
            
            # Add category tag
            category_font = self._get_font(24, bold=True)
            category_text = article['category'].upper()
            category_bbox = draw.textbbox((0, 0), category_text, font=category_font)
            category_width = category_bbox[2] - category_bbox[0] + 30
            category_height = category_bbox[3] - category_bbox[1] + 10
            
            # Draw category background
            draw.rounded_rectangle(
                [(40, 40), (40 + category_width, 40 + category_height)],
                radius=15,
                fill=accent_color
            )
            
            # Draw category text
            self._add_text_with_effects(
                draw, category_text,
                (40 + category_width//2, 40 + category_height//2),
                category_font,
                fill='#ffffff',
                anchor='mm'
            )
            
            # Add watermark if logo path is provided
            # if self.logo_path:
            #     image = self._add_watermark(image, self.logo_path)
            
            return image
            
        except Exception as e:
            logging.error(f"Error creating thumbnail: {e}")
            import traceback
            traceback.print_exc()
            return None
