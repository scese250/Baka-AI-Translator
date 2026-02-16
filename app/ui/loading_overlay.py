from PySide6 import QtCore, QtWidgets, QtGui


class LoadingOverlay(QtWidgets.QWidget):
    """Semi-transparent overlay with a loading message, shown during model pre-loading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)

        self._label = QtWidgets.QLabel(self.tr("Loading models..."), self)
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setStyleSheet(
            "color: #ffffff;"
            "font-size: 18px;"
            "font-weight: 600;"
            "background: transparent;"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.addStretch()
        layout.addWidget(self._label, 0, QtCore.Qt.AlignCenter)
        layout.addStretch()

        self.hide()

        if parent:
            parent.installEventFilter(self)

    # ------------------------------------------------------------------
    def show_overlay(self, text: str = None):
        """Resize to parent, show with semi-transparent background."""
        if text:
            self._label.setText(text)
        if self.parent():
            self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def hide_overlay(self):
        self.hide()

    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj == self.parent() and event.type() == QtCore.QEvent.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 160))
        painter.end()
        super().paintEvent(event)
