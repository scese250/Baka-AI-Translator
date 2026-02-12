import numpy as np
import os
import base64
import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import imkit as imk
from .textblock import TextBlock, sort_textblock_rectangles
from modules.detection.utils.geometry import does_rectangle_fit, is_mostly_contained
from modules.detection.utils.content import get_inpaint_bboxes
from modules.inpainting.lama import LaMa
from modules.inpainting.mi_gan import MIGAN
from modules.inpainting.aot import AOT
from modules.inpainting.schema import Config
from app.ui.messages import Messages



language_codes = {
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese": "zh",
    "Simplified Chinese": "zh-CN",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "Russian": "ru",
    "French": "fr",
    "German": "de",
    "Dutch": "nl",
    "Spanish": "es",
    "Italian": "it",
    "Turkish": "tr",
    "Polish": "pl",
    "Portuguese": "pt",
    "Brazilian Portuguese": "pt-br",
    "Thai": "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Hungarian": "hu",
    "Finnish": "fi",
    "Arabic": "ar",
    "Czech": "cs",
    "Persian": "fa",
    "Romanian": "ro",
    "Mongolian": "mn",
    }

def get_layout_direction(language: str) -> Qt.LayoutDirection:
    return Qt.LayoutDirection.RightToLeft if language == 'Arabic' else Qt.LayoutDirection.LeftToRight


inpaint_map = {
    "LaMa": LaMa,
    "MI-GAN": MIGAN,
    "AOT": AOT,
}

def get_config(settings_page):
    strategy_settings = settings_page.get_hd_strategy_settings()
    if strategy_settings['strategy'] == settings_page.ui.tr("Resize"):
        config = Config(hd_strategy="Resize", hd_strategy_resize_limit = strategy_settings['resize_limit'])
    elif strategy_settings['strategy'] == settings_page.ui.tr("Crop"):
        config = Config(hd_strategy="Crop", hd_strategy_crop_margin = strategy_settings['crop_margin'],
                        hd_strategy_crop_trigger_size = strategy_settings['crop_trigger_size'])
    else:
        config = Config(hd_strategy="Original")

    return config

def is_close(value1, value2, tolerance=2):
    return abs(value1 - value2) <= tolerance

def get_language_code(lng: str):
    lng_cd = language_codes.get(lng, None)
    return lng_cd

def rgba2hex(rgba_list):
    r,g,b,a = [int(num) for num in rgba_list]
    return "#{:02x}{:02x}{:02x}{:02x}".format(r, g, b, a)

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def lists_to_blk_list(blk_list: list[TextBlock], texts_bboxes: list, texts_string: list):  
    group = list(zip(texts_bboxes, texts_string))  

    for blk in blk_list:
        blk_entries = []
        
        for line, text in group:
            if does_rectangle_fit(blk.xyxy, line):
                blk_entries.append((line, text)) 
            elif is_mostly_contained(blk.xyxy, line, 0.5):
                blk_entries.append((line, text)) 

        # Sort and join text entries
        sorted_entries = sort_textblock_rectangles(blk_entries, blk.source_lang_direction)
        if blk.source_lang in ['ja', 'zh']:
            blk.text = ''.join(text for bbox, text in sorted_entries)
        else:
            blk.text = ' '.join(text for bbox, text in sorted_entries)

    return blk_list


def apply_solid_fill_for_uniform_bubbles(
    image: np.ndarray, 
    blk_list: list[TextBlock], 
    mask_dilation: int = 5
) -> tuple[np.ndarray, np.ndarray, list[TextBlock]]:
    """
    For text blocks with uniform backgrounds (white or black),
    apply a solid color fill instead of expensive AI inpainting.
    
    Uses a simple and robust approach: fills the text bounding box
    with the detected background color. Fast and predictable.
    """
    h, w = image.shape[:2]
    solid_fill_mask = np.zeros((h, w), dtype=np.uint8)
    modified_image = image.copy()
    remaining_blocks = []
    
    for blk in blk_list:
        # Skip blocks with no text
        if not blk.text and not blk.translation:
            continue
        
        # Get text bounding box
        tx1, ty1, tx2, ty2 = [int(v) for v in blk.xyxy]
        tx1 = max(0, tx1)
        ty1 = max(0, ty1)
        tx2 = min(w, tx2)
        ty2 = min(h, ty2)
        
        if tx2 <= tx1 or ty2 <= ty1:
            remaining_blocks.append(blk)
            continue
        
        # Sample the area around text to determine fill color
        margin = 10
        bx1 = max(0, tx1 - margin)
        by1 = max(0, ty1 - margin)
        bx2 = min(w, tx2 + margin)
        by2 = min(h, ty2 + margin)
        
        region = image[by1:by2, bx1:bx2]
        if region.size == 0:
            remaining_blocks.append(blk)
            continue
        
        # Convert to grayscale for analysis
        if len(region.shape) == 3:
            gray = np.mean(region, axis=2)
        else:
            gray = region
        
        # Use percentile to detect dominant background color
        p95 = np.percentile(gray, 95)
        p05 = np.percentile(gray, 5)
        p50 = np.percentile(gray, 50)
        
        fill_color = None
        
        # White bubble: 95th percentile is very bright (>240) with high contrast
        if p95 > 240 and (p95 - p05) > 80:
            fill_color = (255, 255, 255)
        # Black bubble: 5th percentile is very dark and median is dark
        elif p05 < 20 and p50 < 60 and (p95 - p05) > 80:
            fill_color = (0, 0, 0)
        # Uniform gray/color: low variance (tight distribution)
        elif abs(p95 - p05) < 35:
            fill_color = (int(p50), int(p50), int(p50))
        
        if fill_color is None:
            # Complex background - needs AI inpainting
            remaining_blocks.append(blk)
            continue
        
        # Apply solid fill to the text bounding box with padding
        pad = mask_dilation + 3
        fill_x1 = max(0, tx1 - pad)
        fill_y1 = max(0, ty1 - pad)
        fill_x2 = min(w, tx2 + pad)
        fill_y2 = min(h, ty2 + pad)
        
        # Apply solid fill
        modified_image[fill_y1:fill_y2, fill_x1:fill_x2] = fill_color
        
        # Mark this area in the solid fill mask
        solid_fill_mask[fill_y1:fill_y2, fill_x1:fill_x2] = 255
    
    return modified_image, solid_fill_mask, remaining_blocks



def generate_mask(img: np.ndarray, blk_list: list[TextBlock], default_padding: int = 5) -> np.ndarray:
    """
    Generate a mask by fitting a merged shape around each block's inpaint bboxes,
    then dilating that shape according to padding logic.
    """
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    LONG_EDGE = 2048

    for blk in blk_list:
        # Skip blocks with no text and no translation
        if not blk.text and not blk.translation:
            continue
        
        bboxes = get_inpaint_bboxes(blk.xyxy, img)
        blk.inpaint_bboxes = bboxes
        if bboxes is None or len(bboxes) == 0:
            continue

        # 1) Compute tight per-block ROI
        xs = [x for x1, _, x2, _ in bboxes for x in (x1, x2)]
        ys = [y for _, y1, _, y2 in bboxes for y in (y1, y2)]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))
        roi_w, roi_h = max_x - min_x + 1, max_y - min_y + 1

        # 2) Down-sample factor to limit mask size
        ds = max(1.0, max(roi_w, roi_h) / LONG_EDGE)
        mw, mh = int(roi_w / ds) + 2, int(roi_h / ds) + 2
        pad_offset = 1

        # 3) Paint bboxes into small mask with padding offset
        small = np.zeros((mh, mw), dtype=np.uint8)
        for x1, y1, x2, y2 in bboxes:
            x1i = int((x1 - min_x) / ds) + pad_offset
            y1i = int((y1 - min_y) / ds) + pad_offset
            x2i = int((x2 - min_x) / ds) + pad_offset
            y2i = int((y2 - min_y) / ds) + pad_offset
            small = imk.rectangle(small, (x1i, y1i), (x2i, y2i), 255, -1)

        # 4) Close small mask to bridge gaps
        # Reduced KSIZE to avoid merging disparate text lines into a single blob
        KSIZE = 3
        kernel = imk.get_structuring_element(imk.MORPH_RECT, (KSIZE, KSIZE))
        closed = imk.morphology_ex(small, imk.MORPH_CLOSE, kernel)

        # 5) Extract all contours
        contours, _ = imk.find_contours(closed)
        if not contours:
            continue

        # 6) Merge contours: collect valid polygons in full image coords
        polys = []
        for cnt in contours:
            pts = cnt.squeeze(1)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            pts_f = (pts.astype(np.float32) - pad_offset) * ds
            pts_f[:, 0] += min_x
            pts_f[:, 1] += min_y
            polys.append(pts_f.astype(np.int32))
        if not polys:
            continue

        # 7) Create per-block mask and fill all polygons
        

        
        block_mask = np.zeros((h, w), dtype=np.uint8)
        block_mask = imk.fill_poly(block_mask, polys, 255)

        # 8) Determine dilation kernel size
        # Use strictly the user's padding setting. 
        # If the user wants more dilation, they can increase the setting in the UI.
        if default_padding > 0:
            dil_kernel = imk.get_structuring_element(imk.MORPH_RECT, (default_padding, default_padding))
            dilated = imk.dilate(block_mask, dil_kernel, iterations=1)
        else:
            dilated = block_mask

        # 10) Combine with global mask
        mask = np.bitwise_or(mask, dilated)

    return mask

def validate_ocr(main_page, source_lang):
    settings_page = main_page.settings_page
    tr = settings_page.ui.tr
    settings = settings_page.get_all_settings()
    credentials = settings.get('credentials', {})
    source_lang_en = main_page.lang_mapping.get(source_lang, source_lang)
    ocr_tool = settings['tools']['ocr']

    # Helper to check authentication or credential
    def has_access(service, key_field):
        return bool(credentials.get(service, {}).get(key_field))
    # Helper to check authentication or presence of multiple credential fields
    def has_all_credentials(service, keys):
        creds = credentials.get(service, {})
        return all(creds.get(k) for k in keys)

    # Microsoft OCR: needs api_key_ocr and endpoint
    if ocr_tool == tr("Microsoft OCR"):
        service = tr("Microsoft Azure")
        if not has_all_credentials(service, ['api_key_ocr', 'endpoint']):
            Messages.show_signup_or_credentials_error(main_page)
            return False

    # Google Cloud Vision
    elif ocr_tool == tr("Google Cloud Vision"):
        service = tr("Google Cloud")
        if not has_access(service, 'api_key'):
            Messages.show_signup_or_credentials_error(main_page)
            return False

    # GPT-based OCR
    elif ocr_tool == tr('GPT-4.1-mini'):
        service = tr('Open AI GPT')
        if not has_access(service, 'api_key'):
            Messages.show_signup_or_credentials_error(main_page)
            return False

    return True


def validate_translator(main_page, source_lang, target_lang):
    settings_page = main_page.settings_page
    tr = settings_page.ui.tr
    settings = settings_page.get_all_settings()
    credentials = settings.get('credentials', {})
    translator_tool = settings['tools']['translator']

    def has_access(service, key_field):
        return bool(credentials.get(service, {}).get(key_field))

    # Credential checks
    if translator_tool == tr("DeepL"):
        service = tr("DeepL")
        if not has_access(service, 'api_key'):
            Messages.show_signup_or_credentials_error(main_page)
            return False
        
    elif translator_tool == tr("Microsoft Translator"):
        service = tr("Microsoft Azure")
        if not has_access(service, 'api_key_translator'):
            Messages.show_signup_or_credentials_error(main_page)
            return False
        
    elif translator_tool == tr("Yandex"):
        service = tr("Yandex")
        if not has_access(service, 'api_key'):
            Messages.show_signup_or_credentials_error(main_page)
            return False
        
    elif "GPT" in translator_tool:
        service = tr('Open AI GPT')
        if not has_access(service, 'api_key'):
            Messages.show_signup_or_credentials_error(main_page)
            return False
        
    elif "Gemini" in translator_tool:
        service = tr('Google Gemini')
        browser_name = credentials.get(service, {}).get('browser', 'Firefox')
        
        # For Gemini, we just verify that a browser is configured.
        # Actual cookie validation happens at runtime to avoid issues with locked browser databases.
        # The user should use "Grab Cookies" button in Settings > Credentials to verify cookies manually.
        if not browser_name:
            Messages.show_gemini_cookies_error(main_page, "Firefox")
            return False
        
        
    elif "Claude" in translator_tool:
        service = tr('Anthropic Claude')
        if not has_access(service, 'api_key'):
            Messages.show_signup_or_credentials_error(main_page)
            return False

    # Unsupported target languages by service
    unsupported = {
        tr("DeepL"): [
            main_page.tr('Thai'),
            main_page.tr('Vietnamese')
        ],
        tr("Google Translate"): [
            main_page.tr('Brazilian Portuguese')
        ]
    }
    unsupported_langs = unsupported.get(translator_tool, [])
    if target_lang in unsupported_langs:
        Messages.show_translator_language_not_supported(main_page)
        return False

    return True

def font_selected(main_page):
    if not main_page.render_settings().font_family:
        Messages.select_font_error(main_page)
        return False
    return True

def validate_settings(main_page, source_lang, target_lang):
    if not validate_ocr(main_page, source_lang):
        return False
    if not validate_translator(main_page, source_lang, target_lang):
        return False
    if not font_selected(main_page):
        return False
    
    return True

def is_directory_empty(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # If any file is found, the directory is not empty
        if files:
            return False
    # If no files are found, check if there are any subdirectories
    for root, dirs, files in os.walk(directory):
        if dirs:
            # Recursively check subdirectories
            for dir in dirs:
                if not is_directory_empty(os.path.join(root, dir)):
                    return False
    return True

def get_smart_text_color(detected_rgb: tuple, setting_color: QColor, outline_width: float = 0.0) -> QColor:
    """
    Determines the best text color to use based on the detected color from the image
    and the user's preferred setting color. Prevents invisible text (e.g. white on white).
    
    CRITICAL: If an outline is present, we TRUST the user's color choice because the outline 
    provides necessary contrast. We disable smart correction in this case.
    """
    # If user has an outline enabled, trust their color choice. Outline guarantees contrast.
    if outline_width > 0:
        return setting_color

    if not detected_rgb:
        return setting_color

    try:
        detected_color = QColor(*detected_rgb)
        if not detected_color.isValid():
            return setting_color

        def get_luma(c):
            return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        
        det_luma = get_luma(detected_color)
        set_luma = get_luma(setting_color)
        
        # If detected is Light (likely on Dark BG) and Setting is Dark
        # e.g. White text on Black BG, but user setting is Black
        if det_luma > 140 and set_luma < 100:
            return detected_color
        
        # If detected is Dark (likely on Light BG) and Setting is Light
        # e.g. Black text on White BG, but user setting is White
        elif det_luma < 100 and set_luma > 140:
            return detected_color
            
    except Exception:
        pass
        
    return setting_color
