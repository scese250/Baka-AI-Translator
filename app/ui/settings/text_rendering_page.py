from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QFontDatabase
from ..dayu_widgets.label import MLabel
from ..dayu_widgets.spin_box import MSpinBox
from ..dayu_widgets.browser import MClickBrowserFileToolButton
from ..dayu_widgets.check_box import MCheckBox
from ..dayu_widgets.push_button import MPushButton
from ..dayu_widgets.combo_box import MFontComboBox

class MClickableColorButton(QtWidgets.QPushButton):
    def __init__(self, color="#000000", parent=None):
        super(MClickableColorButton, self).__init__(parent)
        self.setFixedSize(30, 30)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setProperty('selected_color', color)
        self.update_style()
        self.clicked.connect(self.select_color)

    def update_style(self):
        color = self.property('selected_color')
        self.setStyleSheet(f"background-color: {color}; border: 1px solid #ccc; border-radius: 4px;")

    def set_color(self, color):
        self.setProperty('selected_color', color)
        self.update_style()

    def get_color(self):
        return self.property('selected_color')

    def select_color(self):
        color_dialog = QtWidgets.QColorDialog()
        current_color = self.get_color()
        color_dialog.setCurrentColor(QtGui.QColor(current_color))
        
        if color_dialog.exec() == QtWidgets.QDialog.Accepted:
            color = color_dialog.selectedColor()
            if color.isValid():
                self.set_color(color.name())

class ClassTextConfigWidget(QtWidgets.QWidget):
    def __init__(self, class_name, display_name, parent=None):
        super().__init__(parent)
        self.class_name = class_name
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 5, 0, 5)

        # Label
        self.label = MLabel(display_name).h4()
        self.label.setFixedWidth(120)
        
        # Text Color Checkbox and Button
        self.custom_color_checkbox = MCheckBox(self.tr("Overwrite Text Color"))
        self.text_color_btn = MClickableColorButton("#000000")
        
        # Enable/Disable color btn based on checkbox
        self.custom_color_checkbox.toggled.connect(self.text_color_btn.setEnabled)
        self.text_color_btn.setEnabled(False) # Default disabled
        
        # Outline Checkbox
        self.outline_checkbox = MCheckBox(self.tr("Outline"))
        self.outline_checkbox.setChecked(True)
        
        # Outline Color
        self.outline_color_btn = MClickableColorButton("#FFFFFF")
        
        # Outline Width
        self.outline_width_spinbox = MSpinBox() 
        self.outline_width_combo = QtWidgets.QComboBox()
        self.outline_width_combo.addItems([str(f/10) for f in range(5, 55, 5)]) # 0.5 to 5.0
        self.outline_width_combo.setCurrentText("1.0")
        self.outline_width_combo.setFixedWidth(60)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.custom_color_checkbox)
        self.layout.addWidget(self.text_color_btn)
        self.layout.addSpacing(20)
        self.layout.addWidget(self.outline_checkbox)
        self.layout.addWidget(self.outline_color_btn)
        self.layout.addWidget(self.outline_width_combo)
        self.layout.addStretch()

    def get_settings(self):
        settings = {
            'outline_enabled': self.outline_checkbox.isChecked(),
            'outline_color': self.outline_color_btn.get_color(),
            'outline_width': self.outline_width_combo.currentText()
        }
        
        if self.custom_color_checkbox.isChecked():
            settings['text_color'] = self.text_color_btn.get_color()
            
        return settings

    def set_settings(self, settings):
        if not settings: return
        
        text_color = settings.get('text_color')
        if text_color:
            self.custom_color_checkbox.setChecked(True)
            self.text_color_btn.set_color(text_color)
        else:
            self.custom_color_checkbox.setChecked(False)
            
        self.outline_checkbox.setChecked(settings.get('outline_enabled', False))
        self.outline_color_btn.set_color(settings.get('outline_color', '#FFFFFF'))
        self.outline_width_combo.setCurrentText(str(settings.get('outline_width', '1.0')))


class TextRenderingPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # Uppercase
        self.uppercase_checkbox = MCheckBox(self.tr("Render Text in UpperCase"))
        layout.addWidget(self.uppercase_checkbox)

        # Font section
        font_layout = QtWidgets.QVBoxLayout()
        min_font_layout = QtWidgets.QHBoxLayout()
        max_font_layout = QtWidgets.QHBoxLayout()
        min_font_label = MLabel(self.tr("Minimum Font Size:"))
        max_font_label = MLabel(self.tr("Maximum Font Size:"))

        self.min_font_spinbox = MSpinBox().small()
        self.min_font_spinbox.setFixedWidth(60)
        self.min_font_spinbox.setMaximum(100)
        self.min_font_spinbox.setValue(12) # Default updated to 12 based on image

        self.max_font_spinbox = MSpinBox().small()
        self.max_font_spinbox.setFixedWidth(60)
        self.max_font_spinbox.setMaximum(100)
        self.max_font_spinbox.setValue(60)

        min_font_layout.addWidget(min_font_label)
        min_font_layout.addWidget(self.min_font_spinbox)
        min_font_layout.addStretch()

        max_font_layout.addWidget(max_font_label)
        max_font_layout.addWidget(self.max_font_spinbox)
        max_font_layout.addStretch()

        font_label = MLabel(self.tr("Font:")).h4()

        font_browser_layout = QtWidgets.QHBoxLayout()
        import_font_label = MLabel(self.tr("Import Font:"))
        self.font_browser = MClickBrowserFileToolButton(multiple=True)
        self.font_browser.set_dayu_filters([".ttf", ".ttc", ".otf", ".woff", ".woff2"])
        self.font_browser.setToolTip(self.tr("Import the Font to use for Rendering Text on Images"))

        font_browser_layout.addWidget(import_font_label)
        font_browser_layout.addWidget(self.font_browser)
        font_browser_layout.addStretch()

        font_layout.addWidget(font_label)
        font_layout.addLayout(font_browser_layout)
        font_layout.addLayout(min_font_layout)
        font_layout.addLayout(max_font_layout)

        # Font selector
        import os
        font_folder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'resources', 'fonts'
        )
        self.font_selector = MFontComboBox().small()
        self.font_selector.setToolTip(self.tr("Font"))
        self.font_selector.setMaximumWidth(200)
        if os.path.isdir(font_folder_path):
            for f in os.listdir(font_folder_path):
                if f.endswith((".ttf", ".ttc", ".otf", ".woff", ".woff2")):
                    QFontDatabase.addApplicationFont(os.path.join(font_folder_path, f))
        font_layout.addWidget(self.font_selector)

        layout.addSpacing(10)
        layout.addLayout(font_layout)
        layout.addSpacing(10)

        # Class Overrides
        self.classes_config = {}
        # Define classes to configure
        # Define classes to configure
        classes = [
            ("text_bubble", self.tr("Bubble Text")),
            ("text_free", self.tr("Free Text")),
            ("text_sfx", self.tr("Sound Effect")),
            ("text_narration", self.tr("Narration")),
            ("text_inside_black_bubble", self.tr("Inside Black Bubble")),
        ]

        for cls_name, display_name in classes:
            widget = ClassTextConfigWidget(cls_name, display_name)
            layout.addWidget(widget)
            self.classes_config[cls_name] = widget
            
            # Set default for text_free to Blue to distinguish
            if cls_name == "text_free":
                 widget.text_color_btn.set_color("#0000FF")

        layout.addStretch(1)

    def get_settings(self):
        overrides = {}
        for cls_name, widget in self.classes_config.items():
            overrides[cls_name] = widget.get_settings()
            
        return {
             'min_font_size': self.min_font_spinbox.value(),
             'max_font_size': self.max_font_spinbox.value(),
             'upper_case': self.uppercase_checkbox.isChecked(),
             'color_overrides': overrides
        }

    def set_settings(self, settings):
        if not settings: return
        self.min_font_spinbox.setValue(settings.get('min_font_size', 12))
        self.max_font_spinbox.setValue(settings.get('max_font_size', 60))
        self.uppercase_checkbox.setChecked(settings.get('upper_case', False))
        
        overrides = settings.get('color_overrides', {})
        for cls_name, widget in self.classes_config.items():
            if cls_name in overrides:
                widget.set_settings(overrides[cls_name])
