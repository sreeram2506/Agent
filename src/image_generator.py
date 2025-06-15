import os
import io
import requests
import logging
import hashlib
import unicodedata
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
from typing import Optional, List, Tuple, Union
from io import BytesIO
from urllib.parse import urlparse
import mimetypes
from dataclasses import dataclass
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
import torch
from typing import Optional
from PIL import Image
import logging
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None  # Will be loaded on first use
        
        # Modern color palettes
        self.palettes = {
            'tech': ['#0a192f', '#64ffda', '#00ddeb', '#ff6b6b'],  # Deep navy, cyan, red
            'business': ['#1a1a1a', '#f4c430', '#0077b6', '#ff4757'],  # Black, gold, blue
            'general': ['#2d3436', '#a8e6ce', '#ff7675', '#d8a7b1']  # Dark gray, mint, coral
        }

    def _load_ai_model(self):
        """Load the AI model if not already loaded"""
        if self.pipe is None:
            print("Loading Kandinsky 2.2 model...")
            try:
                self.pipe = AutoPipelineForText2Image.from_pretrained(
                    "kandinsky-community/kandinsky-2-2-decoder",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_safetensors=True
                )
                
                if self.device == "cuda":
                    self.pipe = self.pipe.to("cuda")
                    self.pipe.enable_model_cpu_offload()
                    self.pipe.enable_xformers_memory_efficient_attention()
                    
                print("Model loaded successfully")
                
            except Exception as e:
                logging.error(f"Failed to load Kandinsky model: {e}")
                # Fallback to a simpler model if the main one fails
                try:
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        safety_checker=None
                    )
                    if self.device == "cuda":
                        self.pipe = self.pipe.to("cuda")
                    print("Falling back to Stable Diffusion 1.5")
                except Exception as e2:
                    logging.error(f"Failed to load fallback model: {e2}")
                    raise

    def _generate_ai_image(self, prompt: str, article: dict = None) -> Optional[Image.Image]:
        """Generate an image using AI with enhanced prompt engineering"""
        try:
            self._load_ai_model()
            
            if not prompt and article:
                prompt = article.get('title', 'News illustration')
            
            # Create a more detailed prompt based on article content
            if article:
                enhanced_prompt = (
                    f"Professional news illustration of {article.get('title', '')}. "
                    f"{article.get('description', '')} "
                    "High quality, detailed, professional digital art, trending on artstation, 8k, "
                    "sharp focus, cinematic lighting, highly detailed, intricate"
                )
            else:
                enhanced_prompt = (
                    f"{prompt}, professional illustration, high quality, detailed, "
                    "trending on artstation, 8k, vibrant colors, sharp focus, "
                    "cinematic lighting, highly detailed, intricate"
                )
            
            # Common negative prompt
            negative_prompt = (
                "low quality, blurry, distorted, text, watermark, signature, "
                "ugly, disfigured, low resolution, bad anatomy, extra limbs, "
                "poorly drawn face, mutation, deformed, blurry, bad proportions, "
                "extra fingers, duplicate, cropped, jpeg artifacts, out of frame"
            )
            
            print(f"Generating AI image with prompt: {enhanced_prompt}")
            
            with torch.inference_mode():
                # Try to generate with higher quality settings
                try:
                    # First try with Kandinsky 2.2
                    image = self.pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        width=1024,
                        height=1024,
                        num_inference_steps=50,
                        guidance_scale=8.0,
                        num_images_per_prompt=1,
                    ).images[0]
                except Exception as e:
                    print(f"Error with Kandinsky, falling back to simpler settings: {e}")
                    # Fallback to simpler settings
                    image = self.pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=negative_prompt,
                        width=768,
                        height=768,
                        num_inference_steps=30,
                        guidance_scale=7.5
                    ).images[0]
                
                # Apply some post-processing
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.2)
                
                return image.convert('RGB')
            
        except Exception as e:
            logging.error(f"Error generating AI image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _download_image(self, url: str, prompt: str = None) -> Optional[Image.Image]:
        """Download an image from URL or generate one using AI if URL is not provided."""
        if not url and prompt:
            print("No image URL provided, generating AI image...")
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
                    logging.warning(f"URL does not point to an image, generating AI image instead: {url}")
                    return self._generate_ai_image(prompt or "A beautiful landscape")
            
            image = Image.open(BytesIO(response.content))
            
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image from {url}, falling back to AI generation: {e}")
            return self._generate_ai_image(prompt or "A beautiful landscape")
            
        except Exception as e:
            logging.error(f"Error processing image from {url}, falling back to AI generation: {e}")
            return self._generate_ai_image(prompt or "A beautiful landscape")

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
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
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
                # Use font.getbbox() for modern Pillow versions (9.0.0+)
                try:
                    # First try the modern approach with getbbox
                    bbox = font.getbbox(test_line)
                    text_width = bbox[2] - bbox[0]  # right - left
                except AttributeError:
                    # Fall back to getsize if getbbox doesn't exist
                    try:
                        text_width = font.getsize(test_line)[0]
                    except AttributeError:
                        # If both methods fail, raise a more descriptive error
                        raise RuntimeError("Your Pillow version doesn't support any known text measurement methods. Please upgrade Pillow to the latest version.")
                
                if text_width <= max_width:
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

    def create_news_thumbnail(self, article: dict, image_path: Optional[str] = None) -> Optional[str]:
        """Create a news thumbnail with the given settings and return the file path."""
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
            title_font = self._get_font(48, bold=True)
            title = self._wrap_text(article['title'], title_font, self.image_size[0] - 100)
            
            # Calculate text position
            _, _, _, text_height = draw.textbbox((0, 0), title, font=title_font)
            text_y = self.image_size[1] - 300 - text_height
            
            # Add text with effects
            self._add_text_with_effects(
                draw, title,
                (self.image_size[0] // 2, text_y),
                title_font,
                fill=primary_color,
                stroke=(2, bg_color),
                anchor='mm'
            )
            
            # Add source and date
            source_font = self._get_font(24)
            source_text = f"via {article['source']} • {article['published_at'][:10]}"
            self._add_text_with_effects(
                draw, source_text,
                (self.image_size[0] // 2, self.image_size[1] - 50),
                source_font,
                fill='#ffffff',
                anchor='mm'
            )
            
            # Save the image
            if not image_path:
                # Create assets directory if it doesn't exist
                assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
                os.makedirs(assets_dir, exist_ok=True)
                image_path = os.path.join(assets_dir, 'news_thumbnail.jpg')
            else:
                # Ensure the directory of the provided path exists
                os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
            
            image.save(image_path, 'JPEG', quality=90)
            print(f"Created image at {image_path}")
            return image_path
            
        except Exception as e:
            logging.error(f"Error creating thumbnail: {e}")
            import traceback
            traceback.print_exc()
            return None
