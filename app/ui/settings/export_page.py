from PySide6 import QtWidgets
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.combo_box import MComboBox
from ..dayu_widgets.spin_box import MSpinBox
from ..dayu_widgets.browser import MClickBrowserFolderPushButton
from ..dayu_widgets.radio_button import MRadioButton
from ..dayu_widgets.button_group import MRadioButtonGroup
from .utils import set_combo_box_width

class ExportPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.export_widgets: dict[str, MComboBox] = {}

        layout = QtWidgets.QVBoxLayout(self)

        batch_label = MLabel(self.tr("Automatic Mode")).h4()

        self.raw_text_checkbox = MCheckBox(self.tr("Export Raw Text"))
        self.translated_text_checkbox = MCheckBox(self.tr("Export Translated text"))
        self.inpainted_image_checkbox = MCheckBox(self.tr("Export Inpainted Image"))

        layout.addWidget(batch_label)
        layout.addWidget(self.raw_text_checkbox)
        layout.addWidget(self.translated_text_checkbox)
        layout.addWidget(self.inpainted_image_checkbox)

        # Export Location
        layout.addSpacing(20)
        location_label = MLabel(self.tr("Export Location")).h4()
        layout.addWidget(location_label)

        self.location_group = MRadioButtonGroup()
        self.location_translated_radio = MRadioButton(self.tr("Save in 'Translated' folder"))
        self.location_custom_radio = MRadioButton(self.tr("Save in custom folder"))
        
        self.location_group.set_button_list([
            {"text": self.tr("Save in 'Translated' folder"), "checked": True},
            {"text": self.tr("Save in custom folder")}
        ])
        
        # We need to access individual buttons to connect signals or set states if needed
        # But MRadioButtonGroup manages exclusivity. We can just use the group signal or check logic.
        # However, for the folder browser visibility, we need to know when custom is selected.
        
        # Re-implementing without MRadioButtonGroup for easier custom widget handling if needed, 
        # but Dayu's group is nice. Let's stick to individual radios or handle the group click.
        # Actually, let's use individual radios + QButtonGroup manually if Dayu is restrictive,
        # OR just use standard logic. 
        # Dayu's MRadioButtonGroup creates buttons internally. We might want more control.
        # Let's use separate MRadioButtons.

        self.radio_translated = MRadioButton(self.tr("Save in 'Translated' folder"))
        self.radio_translated.setChecked(True)
        self.radio_custom = MRadioButton(self.tr("Save in custom folder"))

        self.trans_group = QtWidgets.QButtonGroup(self)
        self.trans_group.addButton(self.radio_translated)
        self.trans_group.addButton(self.radio_custom)

        self.custom_folder_browser = MClickBrowserFolderPushButton(self.tr("Select Folder"))
        self.custom_folder_browser.setVisible(False)

        self.radio_custom.toggled.connect(self.custom_folder_browser.setVisible)

        layout.addWidget(self.radio_translated)
        layout.addWidget(self.radio_custom)
        layout.addWidget(self.custom_folder_browser)

        # Image Format
        layout.addSpacing(20)
        image_format_label = MLabel(self.tr("Image Format")).h4()
        layout.addWidget(image_format_label)

        format_layout = QtWidgets.QHBoxLayout()
        format_label = MLabel(self.tr("Save images as:"))
        self.image_format_combo = MComboBox().small()
        self.image_format_combo.addItems(['PNG', 'JPG', 'WEBP'])
        set_combo_box_width(self.image_format_combo, ['PNG', 'JPG', 'WEBP'])

        self.image_quality_label = MLabel(self.tr("Quality:"))
        self.image_quality_spinbox = MSpinBox().small()
        self.image_quality_spinbox.setRange(1, 100)
        self.image_quality_spinbox.setValue(100)
        self.image_quality_spinbox.setSuffix('%')

        # Hide quality controls for PNG (lossless)
        self.image_quality_label.setVisible(False)
        self.image_quality_spinbox.setVisible(False)
        self.image_format_combo.currentTextChanged.connect(self._on_format_changed)

        format_layout.addWidget(format_label)
        format_layout.addWidget(self.image_format_combo)
        format_layout.addWidget(self.image_quality_label)
        format_layout.addWidget(self.image_quality_spinbox)
        format_layout.addStretch(1)
        layout.addLayout(format_layout)

        # File format conversion
        layout.addSpacing(20)
        file_conversion_label = MLabel(self.tr("File Format Conversion")).h4()
        layout.addWidget(file_conversion_label)

        self.from_file_types = ['pdf', 'epub', 'cbr', 'cbz', 'cb7', 'cbt', 'zip', 'rar']
        available_file_types = ['pdf', 'cbz', 'cb7', 'zip']

        for file_type in self.from_file_types:
            save_layout = QtWidgets.QHBoxLayout()
            save_label = MLabel(self.tr("Save {file_type} as:").format(file_type=file_type))
            save_combo = MComboBox().small()
            save_combo.addItems(available_file_types)
            set_combo_box_width(save_combo, available_file_types)

            # Defaults
            if file_type in ['cbr', 'cbt']:
                save_combo.setCurrentText('cbz')
            elif file_type == 'rar':
                save_combo.setCurrentText('zip')
            elif file_type == 'epub':
                save_combo.setCurrentText('pdf')
            elif file_type in available_file_types:
                save_combo.setCurrentText(file_type)

            self.export_widgets[f'.{file_type.lower()}_save_as'] = save_combo

            save_layout.addWidget(save_label)
            save_layout.addWidget(save_combo)
            save_layout.addStretch(1)
            layout.addLayout(save_layout)

        layout.addStretch(1)

    def _on_format_changed(self, fmt: str):
        show_quality = fmt != 'PNG'
        self.image_quality_label.setVisible(show_quality)
        self.image_quality_spinbox.setVisible(show_quality)