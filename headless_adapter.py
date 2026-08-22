"""
Headless adapter for Baka-AI-Translator.
Provides a fake 'main_page' object that mimics ComicTranslate 
so that BatchProcessor + ComicTranslatePipeline can run without PySide6 GUI.

This script is placed in the Baka-AI-Translator root and runs in its venv.
"""
import json
import os
import sys
import glob
import logging

logger = logging.getLogger(__name__)

# ─── Fake Signal ─────────────────────────────────────────────────────────────

class FakeSignal:
    """Mimics PySide6 Signal.emit() — calls registered callbacks or no-ops."""

    def __init__(self, name="", on_emit=None):
        self._name = name
        self._on_emit = on_emit
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def disconnect(self, callback=None):
        if callback:
            self._callbacks = [c for c in self._callbacks if c is not callback]
        else:
            self._callbacks = []

    def emit(self, *args, **kwargs):
        if self._on_emit:
            self._on_emit(*args, **kwargs)
        for cb in self._callbacks:
            try:
                cb(*args, **kwargs)
            except Exception:
                pass


class FakeWorker:
    """Fake current_worker that never cancels."""
    is_cancelled = False

    def cancel(self):
        self.is_cancelled = True


class FakeFileHandler:
    """Minimal FileHandler stub."""
    archive_info = []


# ─── Headless Settings Page ──────────────────────────────────────────────────

class HeadlessSettingsPage:
    """
    Reads settings directly from AppSettings (settings.json)
    and provides the same API as SettingsPage without any Qt widgets.
    """

    def __init__(self):
        from app.settings_manager import AppSettings
        self._settings = AppSettings.instance()
        self.ui = self  # Stub: self acts as ui for tr() calls

    def tr(self, text):
        """Identity translation — headless mode has no Qt i18n."""
        return text

    def get_tool_selection(self, tool_type):
        self._settings.beginGroup('tools')
        result = self._settings.value(tool_type, {
            'translator': 'Gemini-3.0-Flash',
            'ocr': 'Default', 
            'inpainter': 'LaMa (ONNX)',
            'detector': 'RT-DETR-V2',
        }.get(tool_type, ''))
        self._settings.endGroup()
        return result

    def is_gpu_enabled(self):
        self._settings.beginGroup('tools')
        val = self._settings.value('use_gpu', False, type=bool)
        self._settings.endGroup()
        return val

    def is_translate_first_enabled(self):
        self._settings.beginGroup('tools')
        val = self._settings.value('translate_first', False, type=bool)
        self._settings.endGroup()
        return val

    def get_llm_settings(self):
        self._settings.beginGroup('llm')
        result = {
            'extra_context': self._settings.value('extra_context', ''),
            'extra_context_enabled': self._settings.value('extra_context_enabled', True, type=bool),
            'system_prompt_enabled': self._settings.value('system_prompt_enabled', False, type=bool),
            'system_prompt': self._settings.value('system_prompt', ''),
            'image_input_enabled': self._settings.value('image_input_enabled', False, type=bool),
            'advanced_context_aware': self._settings.value('advanced_context_aware', False, type=bool),
            'context_session_enabled': self._settings.value('context_session_enabled', False, type=bool),
            'context_session_name': self._settings.value('context_session_name', ''),
            'gem_name': self._settings.value('gem_name', None),
            'analyze_textless_panels': self._settings.value('analyze_textless_panels', False, type=bool),
            'temperature': 1.0,
            'top_p': 0.95,
            'max_tokens': 65536,
        }
        self._settings.endGroup()
        return result

    def get_export_settings(self):
        self._settings.beginGroup('export')
        result = {
            'export_raw_text': self._settings.value('export_raw_text', False, type=bool),
            'export_translated_text': self._settings.value('export_translated_text', False, type=bool),
            'export_inpainted_image': self._settings.value('export_inpainted_image', False, type=bool),
            'export_location_mode': self._settings.value('export_location_mode', 'translated_folder'),
            'export_custom_path': self._settings.value('export_custom_path', ''),
            'image_format': self._settings.value('image_format', 'PNG'),
            'image_quality': self._settings.value('image_quality', 100, type=int),
            'save_as': {},
        }
        
        # Read save_as sub-group
        self._settings.beginGroup('save_as')
        for ext in ['.pdf', '.epub', '.cbr', '.cbz', '.cb7', '.cbt', '.zip', '.rar']:
            default_map = {
                '.pdf': 'pdf', '.epub': 'pdf', '.cbr': 'cbz',
                '.cbz': 'cbz', '.cb7': 'cb7', '.cbt': 'cbz',
                '.zip': 'zip', '.rar': 'zip'
            }
            result['save_as'][ext] = self._settings.value(ext, default_map.get(ext, ''))
        self._settings.endGroup()  # save_as
        self._settings.endGroup()  # export
        
        return result

    def get_credentials(self, service=""):
        from app.settings_manager import AppSettings
        settings = AppSettings.instance()
        settings.beginGroup('credentials')

        if service:
            creds = {'save_key': True}
            if service == "Microsoft Azure":
                creds.update({
                    'api_key_ocr': settings.value(f"{service}_api_key_ocr", ''),
                    'api_key_translator': settings.value(f"{service}_api_key_translator", ''),
                    'region_translator': settings.value(f"{service}_region_translator", ''),
                    'endpoint': settings.value(f"{service}_endpoint", ''),
                })
            elif service == "AIStudioToAPI":
                creds['base_url'] = settings.value(f"{service}_base_url", 'http://192.168.0.100:7860/v1/chat/completions')
                creds['api_key'] = settings.value(f"{service}_api_key", '123456')
            else:
                creds['api_key'] = settings.value(f"{service}_api_key", '')
            settings.endGroup()
            return creds

        # All services
        all_creds = {}
        known = ["OpenAI", "DeepL", "Google Gemini", "Microsoft Azure",
                 "Anthropic Claude", "Google Cloud", "Yandex", "AIStudioToAPI"]
        for s in known:
            all_creds[s] = self.get_credentials(s)
        settings.endGroup()
        return all_creds

    def get_hd_strategy_settings(self):
        self._settings.beginGroup('tools')
        self._settings.beginGroup('hd_strategy')
        strategy = self._settings.value('strategy', 'Resize')
        result = {'strategy': strategy}
        if strategy == 'Resize':
            result['resize_limit'] = self._settings.value('resize_limit', 960, type=int)
        elif strategy == 'Crop':
            result['crop_margin'] = self._settings.value('crop_margin', 512, type=int)
            result['crop_trigger_size'] = self._settings.value('crop_trigger_size', 512, type=int)
        self._settings.endGroup()  # hd_strategy
        self._settings.endGroup()  # tools
        return result

    def get_mask_dilation(self):
        self._settings.beginGroup('tools')
        val = self._settings.value('mask_dilation', 5, type=int)
        self._settings.endGroup()
        return val

    def get_all_settings(self):
        source_lang = self._settings.value('source_lang', 'Japanese')
        target_lang = self._settings.value('target_lang', 'Spanish')
        
        return {
            'language': self._settings.value('language', 'English'),
            'theme': 'Dark',
            'batch_threads': self._settings.value('batch_threads', 4, type=int),
            'source_lang': source_lang,
            'target_lang': target_lang,
            'tools': {
                'translator': self.get_tool_selection('translator'),
                'ocr': self.get_tool_selection('ocr'),
                'detector': self.get_tool_selection('detector'),
                'inpainter': self.get_tool_selection('inpainter'),
                'mask_dilation': self.get_mask_dilation(),
                'use_gpu': self.is_gpu_enabled(),
                'translate_first': self.is_translate_first_enabled(),
                'hd_strategy': self.get_hd_strategy_settings(),
            },
            'llm': self.get_llm_settings(),
            'export': self.get_export_settings(),
            'credentials': self.get_credentials(),
            'save_keys': True,
        }

    def get_min_font_size(self):
        self._settings.beginGroup('text_rendering')
        val = self._settings.value('min_font_size', 12, type=int)
        self._settings.endGroup()
        return val

    def get_max_font_size(self):
        self._settings.beginGroup('text_rendering')
        val = self._settings.value('max_font_size', 60, type=int)
        self._settings.endGroup()
        return val


# ─── HeadlessMainPage ────────────────────────────────────────────────────────

class HeadlessMainPage:
    """
    Minimum viable fake main_page for BatchProcessor.
    Provides:
    - image_files, image_states, image_data, image_patches
    - settings_page (HeadlessSettingsPage)
    - lang_mapping, button_to_alignment
    - All required signals (FakeSignal)
    - project_file, temp_dir, current_worker, file_handler
    - render_settings() method
    - blk_list
    """

    def __init__(self, image_files, source_lang="Japanese", target_lang="Spanish",
                 progress_callback=None):
        from PySide6.QtCore import Qt

        self.image_files = image_files
        self.curr_img_idx = 0
        self.blk_list = []
        self.image_states = {}
        self.image_data = {}
        self.image_patches = {}
        self.in_memory_history = {}
        self.project_file = None
        self.temp_dir = ""
        self.current_worker = FakeWorker()
        self.file_handler = FakeFileHandler()
        self.settings_page = HeadlessSettingsPage()
        self.webtoon_mode = False
        self.image_viewer = None

        # Build image states for every image
        for img_path in image_files:
            self.image_states[img_path] = {
                'source_lang': source_lang,
                'target_lang': target_lang,
                'skip': False,
                'viewer_state': {
                    'text_items_state': [],
                    'push_to_stack': False,
                },
            }

        # Language mapping (identity — settings uses English names)
        self.lang_mapping = {
            "English": "English", "Korean": "Korean", "Japanese": "Japanese",
            "French": "French", "Simplified Chinese": "Simplified Chinese",
            "Traditional Chinese": "Traditional Chinese", "Chinese": "Chinese",
            "Russian": "Russian", "German": "German", "Dutch": "Dutch",
            "Spanish": "Spanish", "Italian": "Italian", "Turkish": "Turkish",
            "Polish": "Polish", "Portuguese": "Portuguese",
            "Brazilian Portuguese": "Brazilian Portuguese",
            "Thai": "Thai", "Vietnamese": "Vietnamese",
            "Indonesian": "Indonesian", "Hungarian": "Hungarian",
            "Finnish": "Finnish", "Arabic": "Arabic", "Czech": "Czech",
            "Persian": "Persian", "Romanian": "Romanian", "Mongolian": "Mongolian",
        }
        self.reverse_lang_mapping = {v: k for k, v in self.lang_mapping.items()}

        self.button_to_alignment = {
            0: Qt.AlignmentFlag.AlignLeft,
            1: Qt.AlignmentFlag.AlignCenter,
            2: Qt.AlignmentFlag.AlignRight,
        }

        # Signals
        def _on_progress(index, total, step, steps, change_name):
            if progress_callback:
                progress_callback(index, total, step, steps)

        self.progress_update = FakeSignal("progress_update", on_emit=_on_progress)
        self.models_loaded = FakeSignal("models_loaded")
        self.image_skipped = FakeSignal("image_skipped")
        self.blk_rendered = FakeSignal("blk_rendered")
        self.image_processed = FakeSignal("image_processed")
        self.download_event = FakeSignal("download_event")

        # patches_processed must store patches in image_patches
        # so ImageSaveRenderer can apply them during save
        def _on_patches(patches, image_path):
            self.image_patches[image_path] = patches

        self.patches_processed = FakeSignal("patches_processed", on_emit=_on_patches)

    def render_settings(self):
        """Build TextRenderingSettings from saved settings."""
        from PySide6.QtCore import Qt
        from modules.rendering.render import TextRenderingSettings
        from modules.utils.pipeline_utils import get_layout_direction
        from app.settings_manager import AppSettings

        settings = AppSettings.instance()

        target_lang = settings.value('target_lang', 'Spanish')
        direction = get_layout_direction(target_lang)

        # Read text rendering settings
        settings.beginGroup('text_rendering')
        min_font_size = settings.value('min_font_size', 12, type=int)
        max_font_size = settings.value('max_font_size', 60, type=int)
        upper_case = settings.value('upper_case', False, type=bool)

        # Color overrides
        color_overrides = {}
        settings.beginGroup('color_overrides')
        for cls_name in settings.childGroups():
            settings.beginGroup(cls_name)
            color_overrides[cls_name] = {
                'text_color': settings.value('text_color', '#000000'),
                'outline_enabled': settings.value('outline_enabled', True, type=bool),
                'outline_color': settings.value('outline_color', '#FFFFFF'),
                'outline_width': settings.value('outline_width', '1.0'),
            }
            settings.endGroup()
        settings.endGroup()  # color_overrides

        # Font family (inside text_rendering group, outside color_overrides)
        font_family = settings.value('font_family', '')

        settings.endGroup()  # text_rendering

        return TextRenderingSettings(
            alignment_id=1,  # center
            font_family=font_family,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            color="#000000",
            upper_case=upper_case,
            outline=True,
            outline_color="#FFFFFF",
            outline_width="1.0",
            bold=False,
            italic=False,
            underline=False,
            line_spacing="1.0",
            direction=direction,
            color_overrides=color_overrides,
        )


def collect_images(input_dir):
    """Collect all image files from a directory, sorted alphabetically."""
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    files = []
    # Use glob.escape to handle brackets [ ] in directory names
    escaped_dir = glob.escape(input_dir)
    for ext in extensions:
        files.extend(glob.glob(os.path.join(escaped_dir, f'*{ext}')))
        files.extend(glob.glob(os.path.join(escaped_dir, f'*{ext.upper()}')))
    files = sorted(set(files))
    return files
