from PySide6 import QtWidgets
from PySide6.QtCore import Signal
from .utils import create_title_and_combo, set_combo_box_width
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.divider import MDivider
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.push_button import MPushButton


class PersonalizationPage(QtWidgets.QWidget):
    # Profile signals
    profile_save_requested = Signal(str)       # name
    profile_switch_requested = Signal(str)     # name
    profile_delete_requested = Signal(str)     # name
    profile_rename_requested = Signal(str, str)  # old, new

    def __init__(self, languages: list[str], themes: list[str], parent=None):
        super().__init__(parent)
        self.languages = languages
        self.themes = themes

        layout = QtWidgets.QVBoxLayout(self)

        # ── Profiles section ──────────────────────────────────────
        layout.addWidget(MDivider(self.tr("Profiles")))

        profile_row = QtWidgets.QHBoxLayout()
        self.profile_combo = MComboBox().small()
        self.profile_combo.setMinimumWidth(180)
        self.profile_combo.setToolTip(self.tr("Select a profile to load"))
        profile_row.addWidget(self.profile_combo, 1)

        self.btn_save_profile = MPushButton(self.tr("Save")).small()
        self.btn_save_profile.setToolTip(self.tr("Save current settings as a new profile"))
        self.btn_delete_profile = MPushButton(self.tr("Delete")).small()
        self.btn_delete_profile.setToolTip(self.tr("Delete the selected profile"))
        self.btn_rename_profile = MPushButton(self.tr("Rename")).small()
        self.btn_rename_profile.setToolTip(self.tr("Rename the selected profile"))
        profile_row.addWidget(self.btn_save_profile)
        profile_row.addWidget(self.btn_delete_profile)
        profile_row.addWidget(self.btn_rename_profile)

        layout.addLayout(profile_row)

        # Connect profile buttons
        self.btn_save_profile.clicked.connect(self._on_save_clicked)
        self.btn_delete_profile.clicked.connect(self._on_delete_clicked)
        self.btn_rename_profile.clicked.connect(self._on_rename_clicked)
        self.profile_combo.currentTextChanged.connect(self._on_profile_selected)

        # ── Language / Theme / Threads ────────────────────────────
        layout.addWidget(MDivider(self.tr("General")))

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

        # Track whether we are programmatically updating the combo
        self._switching_profile = False

    # ------------------------------------------------------------------
    # Profile combo management
    # ------------------------------------------------------------------

    def refresh_profiles(self, profiles: list[str], active: str = ""):
        """Repopulate the profile combo. If *active* is given, select it."""
        self._switching_profile = True
        self.profile_combo.clear()
        self.profile_combo.addItems(profiles)
        if active and active in profiles:
            self.profile_combo.setCurrentText(active)
        self._switching_profile = False

    # ------------------------------------------------------------------
    # Button handlers → emit signals
    # ------------------------------------------------------------------

    def _on_save_clicked(self):
        name, ok = QtWidgets.QInputDialog.getText(
            self, self.tr("Save Profile"), self.tr("Profile name:"))
        if ok and name.strip():
            self.profile_save_requested.emit(name.strip())

    def _on_delete_clicked(self):
        name = self.profile_combo.currentText()
        if not name:
            return
        reply = QtWidgets.QMessageBox.question(
            self, self.tr("Delete Profile"),
            self.tr("Delete profile \"%s\"?") % name,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.profile_delete_requested.emit(name)

    def _on_rename_clicked(self):
        old_name = self.profile_combo.currentText()
        if not old_name:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, self.tr("Rename Profile"),
            self.tr("New name for \"%s\":") % old_name,
            text=old_name)
        if ok and new_name.strip() and new_name.strip() != old_name:
            self.profile_rename_requested.emit(old_name, new_name.strip())

    def _on_profile_selected(self, name: str):
        if not self._switching_profile and name:
            self.profile_switch_requested.emit(name)