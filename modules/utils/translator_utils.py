import base64
import json
import re
import jieba
import janome.tokenizer
import numpy as np
from pythainlp.tokenize import word_tokenize
from .textblock import TextBlock
import imkit as imk


MODEL_MAP = {
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "Gemini-3.0-Pro": "gemini-3.0-pro"
}

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        # Reemplazar ♡ por ♥ en el texto de entrada
        text = blk.text.replace("♡", "♥") if blk.text else blk.text
        rw_txts_dict[block_key] = text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json

def get_raw_translation(blk_list: list[TextBlock]):
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_translations_dict[block_key] = blk.translation
    
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json

def set_texts_from_json(blk_list: list[TextBlock], json_string: str):
    # Try to clean common formatting issues
    # 1. Remove markdown code blocks if present
    cleaned = re.sub(r'```json\s*', '', json_string)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = cleaned.strip()
    
    # 2. Extract JSON object using regex
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        json_string = match.group(0)
        try:
            translation_dict = json.loads(json_string)
            
            for idx, blk in enumerate(blk_list):
                block_key = f"block_{idx}"
                if block_key in translation_dict:
                    translation = translation_dict[block_key]
                    # Handle REJECT from AI
                    if translation and translation.strip().upper() == "REJECT":
                        blk.translation = ""
                        blk.rejected = True
                    else:
                        # Reemplazar ♡ por ♥ en la traducción de salida
                        blk.translation = translation.replace("♡", "♥") if translation else translation
                else:
                    print(f"Warning: {block_key} not found in JSON string.")
                    
        except json.JSONDecodeError as e:
            # Provide detailed error information
            error_msg = f"JSON parsing failed at line {e.lineno}, column {e.colno}: {e.msg}"
            print(f"\nERROR: {error_msg}")
            print(f"Problematic JSON (first 1000 chars):\n{json_string[:1000]}")
            
            # Re-raise with more context
            raise json.JSONDecodeError(
                f"{e.msg}\n\nProblematic JSON snippet:\n{json_string[:500]}",
                e.doc,
                e.pos
            ) from e
    else:
        error_msg = "No JSON found in the input string."
        print(f"\nERROR: {error_msg}")
        print(f"Raw response (first 1000 chars):\n{cleaned[:1000]}")
        raise ValueError(f"{error_msg}\n\nRaw response snippet:\n{cleaned[:500]}")

def set_upper_case(blk_list: list[TextBlock], upper_case: bool):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper() 
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def normalize_quotes(blk_list: list[TextBlock]):
    """
    Post-translation filter: replaces Spanish guillemets and curly quotes
    with straight double quotes in all translated blocks.
    
    Replacements:
      « » → "
      \u201c \u201d → "
    """
    for blk in blk_list:
        if not blk.translation:
            continue
        blk.translation = (
            blk.translation
            .replace('\u00ab', '"')   # «
            .replace('\u00bb', '"')   # »
            .replace('\u201c', '"')   # "
            .replace('\u201d', '"')   # "
        )


def filter_rejected_blocks(blk_list: list[TextBlock]) -> int:
    """
    Post-translation filter: marks blocks as rejected if their translation
    is 'REJECT' (from AI) or consists entirely of punctuation/symbols.
    Rejected blocks will not be inpainted or rendered.
    
    Returns the count of rejected blocks.
    """
    # Punctuation-only regex: matches strings that are ONLY whitespace + common punctuation
    punct_pattern = re.compile(r'^[\s.,\'"!?;:\-_=+*#@&()\[\]{}|/\\~^`]+$')
    rejected_count = 0
    
    for blk in blk_list:
        if blk.rejected:
            # Already rejected (e.g. by set_texts_from_json)
            rejected_count += 1
            continue
        
        translation = blk.translation
        if not translation:
            continue
        
        stripped = translation.strip()
        
        # Check AI REJECT
        if stripped.upper() == "REJECT":
            blk.translation = ""
            blk.rejected = True
            rejected_count += 1
            continue
        
        # Check punctuation-only translation
        if punct_pattern.match(stripped):
            blk.translation = ""
            blk.rejected = True
            rejected_count += 1
            continue
    
    return rejected_count

def compress_repeated_chars(blk_list: list[TextBlock]):
    """
    Post-translation normalization: collapses runs of 4+ identical
    characters down to 3. For example:
      'aaaaaaaaaaaaaaah' -> 'aaah'
      'NOOOOOOOOO' -> 'NOOO'
      'jajajaja' -> unchanged (alternating pattern, not single repeated char)
    """
    repeat_pattern = re.compile(r'(.)\1{3,}')
    
    for blk in blk_list:
        if blk.rejected or not blk.translation:
            continue
        blk.translation = repeat_pattern.sub(r'\1\1\1', blk.translation)

def get_chinese_tokens(text):
    return list(jieba.cut(text, cut_all=False))

def get_japanese_tokens(text):
    tokenizer = janome.tokenizer.Tokenizer()
    return [token.surface for token in tokenizer.tokenize(text)]

def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        trg_lng_code_lower = trg_lng_cd.lower()
        seg_result = []

        if 'zh' in trg_lng_code_lower:
            seg_result = get_chinese_tokens(translation)

        elif 'ja' in trg_lng_code_lower:
            seg_result = get_japanese_tokens(translation)

        elif 'th' in trg_lng_code_lower:
            seg_result = word_tokenize(translation)

        if seg_result:
            blk.translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
        else:
            # apply casing/formatting for this single block when no segmentation is done
            if translation is None:
                continue
            if upper_case and not translation.isupper():
                blk.translation = translation.upper()
            elif not upper_case and translation.isupper():
                blk.translation = translation.lower().capitalize()
            else:
                blk.translation = translation

def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)
