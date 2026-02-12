from PySide6 import QtWidgets
from .utils import create_title_and_combo, set_combo_box_width
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.divider import MDivider
from ..dayu_widgets.label import MLabel

class PersonalizationPage(QtWidgets.QWidget):
    def __init__(self, languages: list[str], themes: list[str], parent=None):
        super().__init__(parent)
        self.languages = languages
        self.themes = themes

        layout = QtWidgets.QVBoxLayout(self)

        language_widget, self.lang_combo = create_title_and_combo(self.tr("Language"), self.languages)
        set_combo_box_width(self.lang_combo, self.languages)
        theme_widget, self.theme_combo = create_title_and_combo(self.tr("Theme"), self.themes)
        set_combo_box_width(self.theme_combo, self.themes)

        # Thread Count Setting
        self.thread_values = [str(i) for i in range(1, 33)]
        threads_widget, self.threads_combo = create_title_and_combo(self.tr("Batch Threads"), self.thread_values)
        set_combo_box_width(self.threads_combo, self.thread_values)
        # Default to "4" (index 3)
        self.threads_combo.setCurrentIndex(3)
        self.threads_combo.setToolTip(self.tr("Number of parallel threads for batch processing. Higher values use more CPU/RAM."))

        # Source / Target language combos
        from app.ui.main_window import supported_source_languages, supported_target_languages

        layout.addWidget(language_widget)
        layout.addWidget(theme_widget)
        layout.addWidget(threads_widget)

        layout.addWidget(MDivider(self.tr("Source / Target Language")))

        lang_row = QtWidgets.QHBoxLayout()

        src_layout = QtWidgets.QVBoxLayout()
        src_label = MLabel(self.tr("Source Language"))
        self.source_lang_combo = MComboBox().small()
        self.source_lang_combo.setMaximumWidth(200)
        self.source_lang_combo.addItems([self.tr(l) for l in supported_source_languages])
        self.source_lang_combo.setToolTip(self.tr("Source Language"))
        src_layout.addWidget(src_label)
        src_layout.addWidget(self.source_lang_combo)

        tgt_layout = QtWidgets.QVBoxLayout()
        tgt_label = MLabel(self.tr("Target Language"))
        self.target_lang_combo = MComboBox().small()
        self.target_lang_combo.setMaximumWidth(200)
        self.target_lang_combo.addItems([self.tr(l) for l in supported_target_languages])
        self.target_lang_combo.setToolTip(self.tr("Target Language"))
        tgt_layout.addWidget(tgt_label)
        tgt_layout.addWidget(self.target_lang_combo)

        lang_row.addLayout(src_layout)
        lang_row.addLayout(tgt_layout)
        layout.addLayout(lang_row)

        layout.addStretch()