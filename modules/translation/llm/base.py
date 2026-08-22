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

        Uses xyxy (text bbox) for a tight annotation around the actual text.
        """
        if image is None:
            return image

        if image.dtype != np.uint8:
            img_u8 = image.astype(np.uint8)
        else:
            img_u8 = image.copy()

        is_color = img_u8.ndim == 3 and img_u8.shape[2] >= 3

        # Pipeline images are BGR (OpenCV); PIL expects RGB — swap channels
        if is_color:
            pil_img = Image.fromarray(img_u8[..., :3][..., ::-1])
        else:
            pil_img = Image.fromarray(img_u8).convert('RGB')

        draw = ImageDraw.Draw(pil_img)
        img_h, img_w = image.shape[:2]
        lw = max(2, round((img_w + img_h) / 2 * 0.004))
        font_size = max(12, round((img_w + img_h) / 2 * 0.018))

        font_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'font', 'AnimeAce3BB_Bold.otf'
        ))
        try:
            font = ImageFont.truetype(font_path, size=font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

        RED    = (220, 30, 30)
        YELLOW = (255, 255, 0)
        BLACK  = (0, 0, 0)

        GAP = 4  # minimum pixel gap between labels

        def _rects_overlap(a, b):
            """Return True if rectangles (x1,y1,x2,y2) a and b overlap or are within GAP pixels."""
            return (a[0] - GAP < b[2] and a[2] + GAP > b[0] and
                    a[1] - GAP < b[3] and a[3] + GAP > b[1])

        placed_labels = []  # list of (lx1, ly1, lx2, ly2) already drawn

        # Pre-compute bboxes and distance thresholds for adaptive font sizing
        avg_dim = (img_w + img_h) / 2
        ISOLATED = avg_dim * 0.18   # beyond this → isolated → scale up
        CLOSE    = avg_dim * 0.06   # below this → crowded → scale down

        all_bboxes = []
        for _blk in blk_list:
            if _blk.xyxy is not None:
                all_bboxes.append((int(_blk.xyxy[0]), int(_blk.xyxy[1]),
                                   int(_blk.xyxy[2]), int(_blk.xyxy[3])))
            else:
                all_bboxes.append(None)

        def _bbox_gap(a, b):
            """Pixel gap between two bboxes (0 if overlapping)."""
            dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
            return (dx * dx + dy * dy) ** 0.5

        for i, blk in enumerate(blk_list):
            # Use text bbox (xyxy) — tighter than the full speech bubble
            bbox = blk.xyxy
            if bbox is None:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_w = x2 - x1

            # Draw bounding box (multi-pixel outline)
            for offset in range(lw):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=RED
                )

            # Adaptive font size based on proximity to nearest other block
            this_b = all_bboxes[i]
            if this_b is not None and len(all_bboxes) > 1:
                dists = [_bbox_gap(this_b, b) for j, b in enumerate(all_bboxes)
                         if b is not None and j != i]
                min_dist = min(dists) if dists else float('inf')
            else:
                min_dist = float('inf')

            if min_dist > ISOLATED:
                # Isolated — scale up, max 2×
                t = min(1.0, (min_dist - ISOLATED) / ISOLATED)
                local_font_size = min(int(font_size * 2), int(font_size * (1.0 + t)))
            elif min_dist < CLOSE:
                # Crowded — scale down toward 0.6×
                t = min_dist / CLOSE
                local_font_size = max(8, int(font_size * (0.6 + 0.4 * t)))
            else:
                local_font_size = font_size

            # Measure label, shrinking font until it fits within bbox_w
            label = f"BLOCK {i}"
            blk_font_size = local_font_size
            try:
                blk_font = ImageFont.truetype(font_path, size=blk_font_size)
            except (IOError, OSError):
                blk_font = ImageFont.load_default()

            def _measure(f, lbl):
                try:
                    tb = f.getbbox(lbl)
                    return tb[0], tb[1], tb[2] - tb[0], tb[3] - tb[1]
                except AttributeError:
                    w, h = f.getsize(lbl)
                    return 0, 0, w, h

            t_left, t_top, tw, th = _measure(blk_font, label)

            MIN_FONT = 8
            while tw > bbox_w and blk_font_size > MIN_FONT:
                blk_font_size = max(MIN_FONT, blk_font_size - 2)
                try:
                    blk_font = ImageFont.truetype(font_path, size=blk_font_size)
                except (IOError, OSError):
                    blk_font = ImageFont.load_default()
                t_left, t_top, tw, th = _measure(blk_font, label)

            pad = 3
            box_w = tw + pad * 2
            box_h = th + pad * 2

            # Candidate positions in priority order:
            # above-left, above-right, below-left, below-right, inside top-left (last resort)
            candidates = [
                (x1,                   max(0, y1 - box_h)),              # above-left
                (max(0, x2 - box_w),   max(0, y1 - box_h)),              # above-right
                (x1,                   min(img_h - box_h, y2)),          # below-left
                (max(0, x2 - box_w),   min(img_h - box_h, y2)),          # below-right
                (x1,                   y1),                              # inside top-left (fallback)
            ]

            # Occupied zones = already-placed labels + all OTHER blocks' bboxes
            occupied = list(placed_labels)
            for j, other in enumerate(blk_list):
                if j == i or other.xyxy is None:
                    continue
                ox1, oy1, ox2, oy2 = (int(other.xyxy[0]), int(other.xyxy[1]),
                                       int(other.xyxy[2]), int(other.xyxy[3]))
                occupied.append((ox1, oy1, ox2, oy2))

            lx1, ly1 = candidates[-1]  # default: inside fallback
            for cx, cy in candidates:
                cx = max(0, min(cx, img_w - box_w))
                cy = max(0, min(cy, img_h - box_h))
                candidate_rect = (cx, cy, cx + box_w, cy + box_h)
                if not any(_rects_overlap(candidate_rect, p) for p in occupied):
                    lx1, ly1 = cx, cy
                    break

            lx2 = min(lx1 + box_w, img_w)
            ly2 = min(ly1 + box_h, img_h)
            placed_labels.append((lx1, ly1, lx2, ly2))

            # Label background
            draw.rectangle([lx1, ly1, lx2, ly2], fill=RED)

            # Text with black outline for maximum readability
            tx = lx1 + pad - t_left
            ty = ly1 + pad - t_top
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                draw.text((tx + dx, ty + dy), label, fill=BLACK, font=blk_font)
            draw.text((tx, ty), label, fill=YELLOW, font=blk_font)


        # Convert back from RGB to BGR for the rest of the pipeline
        result = np.array(pil_img)
        if is_color:
            result = result[..., ::-1].copy()

        return result

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