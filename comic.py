import os, sys
import logging
from PySide6.QtGui import QIcon, QFont
from PySide6.QtCore import QTranslator, QLocale
from PySide6.QtWidgets import QApplication  
from app.settings_manager import AppSettings
from controller import ComicTranslate
from app.translations import ct_translations
from app import icon_resource

def main():
    
    # Configure logging to file and console
    # Clear log file on each new session (write mode)
    log_file = os.path.join(os.path.dirname(__file__), 'log.txt')
    
    # Create file handler (overwrites on each session)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    
    # Silence noisy third-party loggers
    for noisy in ['httpx', 'httpcore', 'PIL', 'urllib3', 'asyncio']:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    # Also redirect print statements to log file
    class TeeOutput:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, text):
            for stream in self.streams:
                stream.write(text)
                stream.flush()  # Real-time output
        def flush(self):
            for stream in self.streams:
                stream.flush()
    
    log_file_stream = open(log_file, 'a', encoding='utf-8')
    sys.stdout = TeeOutput(sys.__stdout__, log_file_stream)
    sys.stderr = TeeOutput(sys.__stderr__, log_file_stream)
    
    if sys.platform == "win32":
        # Necessary Workaround to set Taskbar Icon on Windows
        import ctypes
        myappid = u'BakaAI.BakaAITranslator' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Initialise JSON settings (singleton)
    settings_path = os.path.join(os.path.dirname(__file__), 'settings.json')
    AppSettings.init(settings_path)

    # Create QApplication directly instead of using the context manager
    app = QApplication(sys.argv)
    
    # Set the application icon
    icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
    icon = QIcon(icon_path)
    app.setWindowIcon(icon)

    settings = AppSettings.instance()
    selected_language = settings.value('language', get_system_language())
    if selected_language != 'English':
        load_translation(app, selected_language)  

    ct = ComicTranslate()

    # Check for file arguments
    if len(sys.argv) > 1:
        project_file = sys.argv[1]
        if os.path.exists(project_file) and project_file.endswith(".ctpr"):
            ct.thread_load_project(project_file)

    ct.show()
    ct.setWindowIcon(icon)
    
    # Start the event loop
    sys.exit(app.exec())


def get_system_language():
    locale = QLocale.system().name()  # Returns something like "en_US" or "zh_CN"
    
    # Special handling for Chinese
    if locale.startswith('zh_'):
        if locale in ['zh_CN', 'zh_SG']:
            return '简体中文'
        elif locale in ['zh_TW', 'zh_HK']:
            return '繁體中文'
    
    # For other languages, we can still use the first part of the locale
    lang_code = locale.split('_')[0]
    
    # Map the system language code to your application's language names
    lang_map = {
        'en': 'English',
        'ko': '한국어',
        'fr': 'Français',
        'ja': '日本語',
        'ru': 'русский',
        'de': 'Deutsch',
        'nl': 'Nederlands',
        'es': 'Español',
        'it': 'Italiano',
        'tr': 'Türkçe'
    }
    
    return lang_map.get(lang_code, 'English')  # Default to English if not found

def load_translation(app, language: str):
    translator = QTranslator(app)
    lang_code = {
        'English': 'en',
        '한국어': 'ko',
        'Français': 'fr',
        '日本語': 'ja',
        '简体中文': 'zh_CN',
        '繁體中文': 'zh_TW',
        'русский': 'ru',
        'Deutsch': 'de',
        'Nederlands': 'nl',
        'Español': 'es',
        'Italiano': 'it',
        'Türkçe': 'tr'
    }.get(language, 'en')

    # Load the translation file
    # if translator.load(f"ct_{lang_code}", "app/translations/compiled"):
    #     app.installTranslator(translator)
    # else:
    #     print(f"Failed to load translation for {language}")

    if translator.load(f":/translations/ct_{lang_code}.qm"):
        app.installTranslator(translator)
    else:
        print(f"Failed to load translation for {language}")

if __name__ == "__main__":
    main()

