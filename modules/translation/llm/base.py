from typing import Any
import numpy as np
from abc import abstractmethod
import base64
import os
import imkit as imk
from PIL import Image, ImageDraw, ImageFont

from ..base import LLMTranslation, REJECT_INSTRUCTION, ANNOTATED_PROMPT, HYPHEN_INSTRUCTION
from ...utils.textblock import TextBlock
from ...utils.translator_utils import get_raw_text, set_texts_from_json, normalize_quotes


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    
    def __init__(self):
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.model = None
        self.img_as_llm_input = False
        self.temperature = None
        self.top_p = None
        self.max_tokens = None
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        llm_settings = settings.get_llm_settings()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.img_as_llm_input = llm_settings.get('image_input_enabled', True)
        self.temperature = llm_settings.get('temperature', 1)
        self.top_p = llm_settings.get('top_p', 0.95)
        self.max_tokens = llm_settings.get('max_tokens', 5000)

        self.system_prompt_enabled = llm_settings.get('system_prompt_enabled', False)
        self.system_prompt = llm_settings.get('system_prompt', '')
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str, image_format: str = ".webp") -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            image_format: Original image format/extension (e.g. ".png", ".webp", ".jpg")
            
        Returns:
            List of updated TextBlock objects with translations
        """
        # Store format for use in _perform_translation
        self.current_image_format = image_format
        entire_raw_text = get_raw_text(blk_list)
        
        if self.system_prompt_enabled and self.system_prompt:
             try:
                system_prompt = self.system_prompt.format(source_lang=self.source_lang, target_lang=self.target_lang)
             except KeyError:
                 # Fallback if user messed up placeholders, just use as is or try to use default if really broken? 
                 # We will use as is to respect user's "hardcoded" input if they didn't want placeholders.
                 system_prompt = self.system_prompt
             # Append REJECT instruction to custom prompts
             system_prompt = f"{system_prompt}\n{REJECT_INSTRUCTION}"
        else:
             system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)

        system_prompt = f"{system_prompt}\n{HYPHEN_INSTRUCTION}"

        user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"
        
        # Annotate image and append visual-reference instructions when image is sent to the LLM
        annotated_image = None
        if self.img_as_llm_input and image is not None:
            annotated_image = self._annotate_image(image, blk_list)
            user_prompt = f"{user_prompt}\n\n{ANNOTATED_PROMPT}"
        
        # Retry: 1 intento con imagen, 1 intento sin imagen
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # En el segundo intento (índice 1), intentar sin imagen
                if attempt == 1:
                    print(f"Translation attempt {attempt + 1}: Attempting without image")
                    entire_translated_text = self._perform_translation(user_prompt, system_prompt, None)
                else:
                    entire_translated_text = self._perform_translation(user_prompt, system_prompt, annotated_image if annotated_image is not None else image)
                
                set_texts_from_json(blk_list, entire_translated_text)
                normalize_quotes(blk_list)
                # Si tiene éxito, salir del loop
                break
            except (ValueError, Exception) as e:
                last_error = e
                error_msg = str(e)
                
                # Si es el último intento, lanzar la excepción
                if attempt == max_retries - 1:
                    raise
                
                # Log del reintento
                print(f"Translation attempt {attempt + 1} failed: {error_msg}")

                # Check for fatal errors that shouldn't trigger a retry
                if ("accounts failed" in error_msg and "Gemini Web Error" in error_msg) or "CAMBIO DE MODELO" in error_msg:
                    print("Fatal error detected. Stopping retries.")
                    raise

                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
            
        return blk_list
    
    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass

    def _annotate_image(self, image: np.ndarray, blk_list: list) -> np.ndarray:
        """
        Return a copy of the image with each text block annotated with a red
        bounding box and a "BLOCK N" label.

        Uses bubble_xyxy when available, otherwise falls back to xyxy.
        """
        if image is None:
            return image

        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            img_u8 = image.astype(np.uint8)
        else:
            img_u8 = image.copy()

        if len(img_u8.shape) == 2:
            pil_img = Image.fromarray(img_u8, mode='L').convert('RGB')
        else:
            pil_img = Image.fromarray(img_u8)

        draw = ImageDraw.Draw(pil_img)
        img_h, img_w = image.shape[:2]
        lw = max(2, round((img_w + img_h) / 2 * 0.004))  # adaptive line width
        font_size = max(14, round((img_w + img_h) / 2 * 0.022))

        # Try to load the project's bold font; fall back to PIL default
        font_path = os.path.join(
            os.path.dirname(__file__),       # llm/
            '..', '..', '..', 'font',        # -> project root/font/
            'AnimeAce3BB_Bold.otf'
        )
        font_path = os.path.normpath(font_path)
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        RED = (220, 30, 30)
        WHITE = (255, 255, 255)

        for i, blk in enumerate(blk_list):
            # Prefer bubble bbox for the visual frame
            bbox = blk.bubble_xyxy if blk.bubble_xyxy is not None else blk.xyxy
            if bbox is None:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Draw bounding box
            for offset in range(lw):
                draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=RED)

            # Draw label background + text
            label = f"BLOCK {i}"
            try:
                # Pillow >= 10.0 uses getbbox
                tb = font.getbbox(label)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except AttributeError:
                tw, th = font.getsize(label)

            pad = 3
            lx1 = x1
            ly1 = max(0, y1 - th - pad * 2)
            draw.rectangle([lx1, ly1, lx1 + tw + pad * 2, ly1 + th + pad * 2], fill=RED)
            draw.text((lx1 + pad, ly1 + pad), label, fill=WHITE, font=font)

        return np.array(pil_img)

    def encode_image(self, image: np.ndarray, ext=".jpg"):
        """
        Encode CV2/numpy image directly to base64 string using cv2.imencode.
        
        Args:
            image: Numpy array representing the image
            ext: Extension/format to encode the image as (".png" by default for higher quality)
                
        Returns:
            Tuple of (Base64 encoded string, mime_type)
        """
        # Direct encoding from numpy/cv2 format to bytes
        buffer = imk.encode_image(image, ext.lstrip('.'))
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Map extension to mime type
        mime_types = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext.lower(), f"image/{ext[1:].lower()}")
        
        return img_str, mime_type