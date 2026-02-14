import os, shutil
from dataclasses import asdict, is_dataclass

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Signal, QTimer
from PySide6.QtGui import QFont, QFontDatabase

from .settings_ui import SettingsPageUI

# Dictionary to map old model names to the newest versions in settings
OCR_MIGRATIONS = {
    "GPT-4o":       "GPT-4.1-mini",
    "Gemini-2.5-Flash": "Gemini-2.0-Flash",
    "Gemini-2.5-Flash (Cookies)": "Gemini-3.0-Flash (Cookies)",
}

TRANSLATOR_MIGRATIONS = {
    "GPT-4o": "Gemini-3.0-Pro",
    "GPT-4o mini": "Gemini-3.0-Flash",
    "GPT-4.1": "Gemini-3.0-Pro",
    "GPT-4.1-mini": "Gemini-3.0-Flash",
    "DeepL": "Gemini-3.0-Pro",
    "Claude-4.5-Sonnet": "Gemini-3.0-Pro",
    "Claude-4.5-Haiku": "Gemini-3.0-Flash",
    "Claude-3-Opus": "Gemini-3.0-Pro",
    "Claude-4-Sonnet": "Gemini-3.0-Pro",
    "Claude-3-Haiku": "Gemini-3.0-Flash",
    "Claude-3.5-Haiku": "Gemini-3.0-Flash",
    "Gemini-2.0-Flash": "Gemini-3.0-Flash",
    "Gemini-2.0-Pro": "Gemini-3.0-Pro",
    "Gemini-2.5-Flash": "Gemini-3.0-Flash",
    "Gemini-2.5-Pro": "Gemini-3.0-Pro",
    "Yandex": "Gemini-3.0-Pro",
    "Google Translate": "Gemini-3.0-Pro",
    "Microsoft Translator": "Gemini-3.0-Pro",
    "Deepseek-v3": "Gemini-3.0-Pro",
    "Custom": "Gemini-3.0-Pro",
    "Gemini-3.0-Flash": "Gemini-3.0-Flash",
    "GeminiLocal-3-Flash": "AIStudio-3-Flash",
    "GeminiLocal-3-Pro": "AIStudio-3-Pro",
}

INPAINTER_MIGRATIONS = {
    "MI-GAN": "AOT",
}

class SettingsPage(QtWidgets.QWidget):
    theme_changed = Signal(str)
    font_imported = Signal(str)

    def __init__(self, parent=None):
        super(SettingsPage, self).__init__(parent)

        self.ui = SettingsPageUI(self)
        self._loading_settings = True

        # Debounced auto-save timer (500ms)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._do_auto_save)

        self._setup_connections()
        self._loading_settings = False

        # Use the Settings UI directly; inner content is scrollable on the
        # right side (see settings_ui.py). This keeps the left navbar fixed.
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.ui)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def _setup_connections(self):
        # Connect signals to slots
        self.ui.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        self.ui.lang_combo.currentTextChanged.connect(self.on_language_changed)
        self.ui.font_browser.sig_files_changed.connect(self.import_font)

        # Auto-save: connect all child widgets to debounced save
        for widget in self.ui.findChildren(QtWidgets.QWidget):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._schedule_save)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.stateChanged.connect(self._schedule_save)
            elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                widget.valueChanged.connect(self._schedule_save)
            elif isinstance(widget, QtWidgets.QRadioButton):
                widget.toggled.connect(self._schedule_save)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.textChanged.connect(self._schedule_save)

    def _schedule_save(self, *_args):
        """Restart the debounce timer; save_settings fires 500ms after last change."""
        if not self._loading_settings:
            self._save_timer.start()

    def _do_auto_save(self):
        """Called by the debounce timer to persist settings to disk."""
        if not self._loading_settings:
            self.save_settings()

    def on_theme_changed(self, theme: str):
        self.theme_changed.emit(theme)

    def get_language(self):
        return self.ui.lang_combo.currentText()
    
    def get_theme(self):
        return self.ui.theme_combo.currentText()

    def get_tool_selection(self, tool_type):
        tool_combos = {
            'translator': self.ui.translator_combo,
            'ocr': self.ui.ocr_combo,
            'inpainter': self.ui.inpainter_combo,
            'detector': self.ui.detector_combo
        }
        return tool_combos[tool_type].currentText()

    def is_gpu_enabled(self):
        return self.ui.use_gpu_checkbox.isChecked()

    def get_llm_settings(self):
        return {
            'extra_context': self.ui.extra_context.toPlainText(),
            'extra_context_enabled': self.ui.extra_context_enabled.isChecked(),
            'system_prompt_enabled': self.ui.system_prompt_enabled.isChecked(),
            'system_prompt': self.ui.system_prompt.toPlainText(),
            'image_input_enabled': self.ui.image_checkbox.isChecked(),
            'advanced_context_aware': self.ui.advanced_context_checkbox.isChecked(),
            'context_session_enabled': self.ui.context_session_checkbox.isChecked(),
            'context_session_name': self.ui.session_name_input.text().strip(),
            'gem_name': self.ui.gems_combo.currentData(),
            'analyze_textless_panels': self.ui.analyze_textless_checkbox.isChecked(),
            'temperature': 1.0,
            'top_p': 0.95,
            'max_tokens': 65536,
        }

    def get_export_settings(self):
        settings = {
            'export_raw_text': self.ui.raw_text_checkbox.isChecked(),
            'export_translated_text': self.ui.translated_text_checkbox.isChecked(),
            'export_inpainted_image': self.ui.inpainted_image_checkbox.isChecked(),
            'save_as': {}
        }
        for file_type in self.ui.from_file_types:
            settings['save_as'][f'.{file_type}'] = self.ui.export_widgets[f'.{file_type}_save_as'].currentText()
        
        settings['export_location_mode'] = 'custom' if self.ui.radio_custom.isChecked() else 'translated_folder'
        settings['export_custom_path'] = self.ui.custom_folder_browser.get_dayu_path()
        settings['image_format'] = self.ui.image_format_combo.currentText()
        settings['image_quality'] = self.ui.image_quality_spinbox.value()

        return settings

    def get_credentials(self, service: str = ""):
        # save_keys = self.ui.save_keys_checkbox.isChecked()
        save_keys = True # Always save keys now

        def _text_or_none(widget_key):
            w = self.ui.credential_widgets.get(widget_key)
            return w.text() if w is not None else None

        if service:
            creds = {'save_key': save_keys}
            if service == "Microsoft Azure":
                creds.update({
                    'api_key_ocr': _text_or_none("Microsoft Azure_api_key_ocr"),
                    'api_key_translator': _text_or_none("Microsoft Azure_api_key_translator"),
                    'region_translator': _text_or_none("Microsoft Azure_region"),
                    'endpoint': _text_or_none("Microsoft Azure_endpoint"),
                })
            elif service == "Custom":
                for field in ("api_key", "api_url", "model"):
                    creds[field] = _text_or_none(f"Custom_{field}")
            elif service == "Yandex":
                creds['api_key'] = _text_or_none("Yandex_api_key")
                creds['folder_id'] = _text_or_none("Yandex_folder_id")
            elif service == "AIStudioToAPI":
                creds['base_url'] = _text_or_none("AIStudioToAPI_base_url")
                creds['api_key'] = _text_or_none("AIStudioToAPI_api_key")
            # Google Gemini browser section removed
            else:
                creds['api_key'] = _text_or_none(f"{service}_api_key")

            return creds

        # no `service` passed → recurse over all known services
        return {s: self.get_credentials(s) for s in self.ui.credential_services}
        
    def get_hd_strategy_settings(self):
        strategy = self.ui.inpaint_strategy_combo.currentText()
        settings = {
            'strategy': strategy
        }

        if strategy == self.ui.tr("Resize"):
            settings['resize_limit'] = self.ui.resize_spinbox.value()
        elif strategy == self.ui.tr("Crop"):
            settings['crop_margin'] = self.ui.crop_margin_spinbox.value()
            settings['crop_trigger_size'] = self.ui.crop_trigger_spinbox.value()

        return settings

    def get_mask_dilation(self):
        return self.ui.dilation_spinbox.value()

    def get_all_settings(self):
        return {
            'language': self.get_language(),
            'theme': self.get_theme(),
            'batch_threads': int(self.ui.threads_combo.currentText()),
            'source_lang': self.ui.source_lang_combo.currentText(),
            'target_lang': self.ui.target_lang_combo.currentText(),
            'tools': {
                'translator': self.get_tool_selection('translator'),
                'ocr': self.get_tool_selection('ocr'),
                'detector': self.get_tool_selection('detector'),
                'inpainter': self.get_tool_selection('inpainter'),
                'mask_dilation': self.get_mask_dilation(),
                'use_gpu': self.is_gpu_enabled(),
                'hd_strategy': self.get_hd_strategy_settings()
            },
            'llm': self.get_llm_settings(),
            'export': self.get_export_settings(),
            'credentials': self.get_credentials(),
            'credentials': self.get_credentials(),
            'save_keys': True, # self.ui.save_keys_checkbox.isChecked(),
            'text_rendering': self.ui.text_rendering_page.get_settings(),
        }

    def import_font(self, file_paths: list[str]):

        file_paths = [f for f in file_paths 
                      if f.endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2"))]

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
        font_folder_path = os.path.join(project_root, 'resources', 'fonts')

        if not os.path.exists(font_folder_path):
            os.makedirs(font_folder_path)

        if file_paths:
            for file in file_paths:
                shutil.copy(file, font_folder_path)
                
            font_files = [os.path.join(font_folder_path, f) for f in os.listdir(font_folder_path) 
                      if f.endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2"))]
            
            font_families = []
            for font in font_files:
                font_family = self.add_font_family(font)
                font_families.append(font_family)
            
            if font_families:
                self.font_imported.emit(font_families[0])

    def select_color(self, outline = False):
        default_color = QtGui.QColor('#000000') if not outline else QtGui.QColor('#FFFFFF')
        color_dialog = QtWidgets.QColorDialog()
        color_dialog.setCurrentColor(default_color)
        
        if color_dialog.exec() == QtWidgets.QDialog.Accepted:
            color = color_dialog.selectedColor()
            if color.isValid():
                button = self.ui.color_button if not outline else self.ui.outline_color_button
                button.setStyleSheet(
                    f"background-color: {color.name()}; border: none; border-radius: 5px;"
                )
                button.setProperty('selected_color', color.name())

    # With the mappings, settings are saved with English values and loaded in the selected language
    def save_settings(self):
        from app.settings_manager import AppSettings
        settings = AppSettings.instance()
        all_settings = self.get_all_settings()

        def process_group(group_key, group_value, settings_obj):
            """Helper function to process a group and its nested values."""
            if is_dataclass(group_value):
                group_value = asdict(group_value)
            if isinstance(group_value, dict):
                settings_obj.beginGroup(group_key)
                for sub_key, sub_value in group_value.items():
                    process_group(sub_key, sub_value, settings_obj)
                settings_obj.endGroup()
            else:
                # Convert value to English using mappings if available
                mapped_value = self.ui.value_mappings.get(group_value, group_value)
                settings_obj.setValue(group_key, mapped_value)

        for key, value in all_settings.items():
            process_group(key, value, settings)

        # Handle Radio Button Saving manually
        settings.beginGroup('export')
        settings.setValue('export_location_mode', all_settings['export']['export_location_mode'])
        settings.setValue('export_custom_path', all_settings['export']['export_custom_path'])
        settings.endGroup()

        # Save credentials separately (Always save now)
        credentials = self.get_credentials()
        # save_keys = self.ui.save_keys_checkbox.isChecked()
        settings.beginGroup('credentials')
        settings.setValue('save_keys', True)
        # if save_keys:
        if True:
            for service, cred in credentials.items():
                translated_service = self.ui.value_mappings.get(service, service)
                if translated_service == "Microsoft Azure":
                    settings.setValue(f"{translated_service}_api_key_ocr", cred['api_key_ocr'])
                    settings.setValue(f"{translated_service}_api_key_translator", cred['api_key_translator'])
                    settings.setValue(f"{translated_service}_region_translator", cred['region_translator'])
                    settings.setValue(f"{translated_service}_endpoint", cred['endpoint'])
                elif translated_service == "Custom":
                    settings.setValue(f"{translated_service}_api_key", cred['api_key'])
                    settings.setValue(f"{translated_service}_api_url", cred['api_url'])
                    settings.setValue(f"{translated_service}_model", cred['model'])
                elif translated_service == "Yandex":
                    settings.setValue(f"{translated_service}_api_key", cred['api_key'])
                    settings.setValue(f"{translated_service}_folder_id", cred['folder_id'])
                elif translated_service == "AIStudioToAPI":
                    settings.setValue(f"{translated_service}_base_url", cred['base_url'])
                    settings.setValue(f"{translated_service}_api_key", cred['api_key'])
                # Google Gemini browser save logic removed
                else:
                    settings.setValue(f"{translated_service}_api_key", cred['api_key'])
        # else:
        #     settings.remove('credentials')  # Clear all credentials if save_keys is unchecked
        settings.endGroup()

        settings.sync()

    def load_settings(self):
        self._loading_settings = True
        from app.settings_manager import AppSettings
        settings = AppSettings.instance()

        # Load language
        language = settings.value('language', 'English')
        translated_language = self.ui.reverse_mappings.get(language, language)
        self.ui.lang_combo.setCurrentText(translated_language)

        # Load theme
        theme = settings.value('theme', 'Dark')
        translated_theme = self.ui.reverse_mappings.get(theme, theme)
        self.ui.theme_combo.setCurrentText(translated_theme)
        self.theme_changed.emit(translated_theme)

        # Load batch threads
        threads = str(settings.value('batch_threads', 4, type=int))
        self.ui.threads_combo.setCurrentText(threads)

        # Load source/target language
        source_lang = settings.value('source_lang', 'Japanese')
        self.ui.source_lang_combo.setCurrentText(source_lang)
        target_lang = settings.value('target_lang', 'English')
        self.ui.target_lang_combo.setCurrentText(target_lang)

        # Load tools settings
        settings.beginGroup('tools')
        raw_translator = settings.value('translator', 'Gemini-3.0-Flash')
        translator = TRANSLATOR_MIGRATIONS.get(raw_translator, raw_translator)
        translated_translator = self.ui.reverse_mappings.get(translator, translator)
        self.ui.translator_combo.setCurrentText(translated_translator)

        raw_ocr = settings.value('ocr', 'Default')
        ocr = OCR_MIGRATIONS.get(raw_ocr, raw_ocr)
        translated_ocr = self.ui.reverse_mappings.get(ocr, ocr)
        self.ui.ocr_combo.setCurrentText(translated_ocr)

        raw_inpainter = settings.value('inpainter', 'LaMa')
        inpainter = INPAINTER_MIGRATIONS.get(raw_inpainter, raw_inpainter)
        translated_inpainter = self.ui.reverse_mappings.get(inpainter, inpainter)
        self.ui.inpainter_combo.setCurrentText(translated_inpainter)

        detector = settings.value('detector', 'RT-DETR-V2')
        translated_detector = self.ui.reverse_mappings.get(detector, detector)
        self.ui.detector_combo.setCurrentText(translated_detector)

        self.ui.use_gpu_checkbox.setChecked(settings.value('use_gpu', False, type=bool))
        self.ui.dilation_spinbox.setValue(settings.value('mask_dilation', 5, type=int))

        # Load HD strategy settings
        settings.beginGroup('hd_strategy')
        strategy = settings.value('strategy', 'Resize')
        translated_strategy = self.ui.reverse_mappings.get(strategy, strategy)
        self.ui.inpaint_strategy_combo.setCurrentText(translated_strategy)
        if strategy == 'Resize':
            self.ui.resize_spinbox.setValue(settings.value('resize_limit', 960, type=int))
        elif strategy == 'Crop':
            self.ui.crop_margin_spinbox.setValue(settings.value('crop_margin', 512, type=int))
            self.ui.crop_trigger_spinbox.setValue(settings.value('crop_trigger_size', 512, type=int))
        settings.endGroup()  # hd_strategy
        settings.endGroup()  # tools

        # Load LLM settings
        settings.beginGroup('llm')
        self.ui.extra_context_enabled.setChecked(settings.value('extra_context_enabled', True, type=bool))
        self.ui.extra_context.setPlainText(settings.value('extra_context', ''))
        self.ui.extra_context.setEnabled(self.ui.extra_context_enabled.isChecked())
        self.ui.system_prompt_enabled.setChecked(settings.value('system_prompt_enabled', False, type=bool))
        self.ui.system_prompt.setPlainText(settings.value('system_prompt', ''))
        self.ui.image_checkbox.setChecked(settings.value('image_input_enabled', False, type=bool))
        self.ui.advanced_context_checkbox.setChecked(settings.value('advanced_context_aware', False, type=bool))
        self.ui.context_session_checkbox.setChecked(settings.value('context_session_enabled', False, type=bool))
        self.ui.session_name_input.setText(settings.value('context_session_name', ''))
        # Note: gem_name from combo is loaded by name, works across accounts
        saved_gem_name = settings.value('gem_name', None)
        if saved_gem_name:
            # Find index by data (gem name) and set it
            for i in range(self.ui.gems_combo.count()):
                if self.ui.gems_combo.itemData(i) == saved_gem_name:
                    self.ui.gems_combo.setCurrentIndex(i)
                    break
        self.ui.analyze_textless_checkbox.setChecked(settings.value('analyze_textless_panels', False, type=bool))
        settings.endGroup()

        # Load Text Rendering settings
        settings.beginGroup('text_rendering')
        # We read the whole group into a dict for the page to parse
        text_rendering_props = {}
        for key in settings.allKeys():
             text_rendering_props[key] = settings.value(key)
        
        # QSettings reading nested dicts is tricky, so we might need to handle color_overrides specifically
        # However, since we save it via process_group which flattens it, we need to read it back carefully.
        # But wait, QSettings flatten structure: text_rendering/color_overrides/text_bubble/text_color
        
        # Let's re-implement reading to handle the structure we defined in save_settings
        tr_settings = {
             'min_font_size': settings.value('min_font_size', 12, type=int),
             'max_font_size': settings.value('max_font_size', 60, type=int),
             'upper_case': settings.value('upper_case', False, type=bool),
             'color_overrides': {}
        }
        
        settings.beginGroup('color_overrides')
        for cls_name in settings.childGroups():
            settings.beginGroup(cls_name)
            tr_settings['color_overrides'][cls_name] = {
                'text_color': settings.value('text_color', '#000000'),
                'outline_enabled': settings.value('outline_enabled', True, type=bool),
                'outline_color': settings.value('outline_color', '#FFFFFF'),
                'outline_width': settings.value('outline_width', '1.0')
            }
            settings.endGroup()
        settings.endGroup() # color_overrides
        
        self.ui.text_rendering_page.set_settings(tr_settings)
        settings.endGroup() # text_rendering

        # Load export settings
        settings.beginGroup('export')
        self.ui.raw_text_checkbox.setChecked(settings.value('export_raw_text', False, type=bool))
        self.ui.translated_text_checkbox.setChecked(settings.value('export_translated_text', False, type=bool))
        self.ui.inpainted_image_checkbox.setChecked(settings.value('export_inpainted_image', False, type=bool))
        settings.beginGroup('save_as')
        
        # Default mappings for file format conversion
        default_save_as = {
            '.pdf': 'pdf',
            '.epub': 'pdf',
            '.cbr': 'cbz',
            '.cbz': 'cbz',
            '.cb7': 'cb7',
            '.cbt': 'cbz',
            '.zip': 'zip',
            '.rar': 'zip'
        }
        
        for file_type in self.ui.from_file_types:
            file_ext = f'.{file_type}'
            default_value = default_save_as.get(file_ext, file_type)
            self.ui.export_widgets[f'{file_ext}_save_as'].setCurrentText(settings.value(file_ext, default_value))
        settings.endGroup()  # save_as

        export_mode = settings.value('export_location_mode', 'translated_folder')
        if export_mode == 'custom':
            self.ui.radio_custom.setChecked(True)
        else:
            self.ui.radio_translated.setChecked(True) # Default
        
        self.ui.custom_folder_browser.set_dayu_path(settings.value('export_custom_path', ''))

        image_format = settings.value('image_format', 'PNG')
        self.ui.image_format_combo.setCurrentText(image_format)
        self.ui.image_quality_spinbox.setValue(settings.value('image_quality', 100, type=int))
        
        settings.endGroup()  # export

        # Load credentials
        settings.beginGroup('credentials')
        save_keys = settings.value('save_keys', True, type=bool)
        # self.ui.save_keys_checkbox.setChecked(save_keys)
        if save_keys:
            for service in self.ui.credential_services:
                translated_service = self.ui.value_mappings.get(service, service)
                if translated_service == "Microsoft Azure":
                    self.ui.credential_widgets["Microsoft Azure_api_key_ocr"].setText(settings.value(f"{translated_service}_api_key_ocr", ''))
                    self.ui.credential_widgets["Microsoft Azure_api_key_translator"].setText(settings.value(f"{translated_service}_api_key_translator", ''))
                    self.ui.credential_widgets["Microsoft Azure_region"].setText(settings.value(f"{translated_service}_region_translator", ''))
                    self.ui.credential_widgets["Microsoft Azure_endpoint"].setText(settings.value(f"{translated_service}_endpoint", ''))
                elif translated_service == "Custom":
                    self.ui.credential_widgets[f"{translated_service}_api_key"].setText(settings.value(f"{translated_service}_api_key", ''))
                    self.ui.credential_widgets[f"{translated_service}_api_url"].setText(settings.value(f"{translated_service}_api_url", ''))
                    self.ui.credential_widgets[f"{translated_service}_model"].setText(settings.value(f"{translated_service}_model", ''))
                elif translated_service == "Yandex":
                    self.ui.credential_widgets[f"{translated_service}_api_key"].setText(settings.value(f"{translated_service}_api_key", ''))
                    self.ui.credential_widgets[f"{translated_service}_folder_id"].setText(settings.value(f"{translated_service}_folder_id", ''))
                elif translated_service == "AIStudioToAPI":
                    self.ui.credential_widgets[f"{translated_service}_base_url"].setText(settings.value(f"{translated_service}_base_url", 'http://localhost:7860/v1/chat/completions'))
                    self.ui.credential_widgets[f"{translated_service}_api_key"].setText(settings.value(f"{translated_service}_api_key", '123456'))
                elif translated_service == "Google Gemini":
                    pass # Handled via Cookies.txt
                else:
                    self.ui.credential_widgets[f"{translated_service}_api_key"].setText(settings.value(f"{translated_service}_api_key", ''))
        settings.endGroup()

        self._loading_settings = False

    def on_language_changed(self, new_language):
        if not self._loading_settings:  
            self.show_restart_dialog()

    def show_restart_dialog(self):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle(self.tr("Restart Required"))
        msg_box.setText(self.tr("Please restart the application for the language changes to take effect."))
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec()

    def get_min_font_size(self):
        return int(self.ui.min_font_spinbox.value())
    
    def get_max_font_size(self):
        return int(self.ui.max_font_spinbox.value())

    def add_font_family(self, font_input: str) -> QFont:
        # Check if font_input is a file path
        if os.path.splitext(font_input)[1].lower() in [".ttf", ".ttc", ".otf", ".woff", ".woff2"]:
            font_id = QFontDatabase.addApplicationFont(font_input)
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    return font_families[0]
        
        # If not a file path or loading failed, treat as font family name
        return font_input



