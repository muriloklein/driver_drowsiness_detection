from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel

from src.fatigue import FatigueLevel

BLINK_MS = 400
STYLE_ON = "background: #d32f2f; color: white; font-size: 18px; font-weight: bold; padding: 6px;"
STYLE_OFF = "background: #7f1d1d; color: #ddd; font-size: 18px; font-weight: bold; padding: 6px;"


class AlertBanner(QLabel):
    """Faixa vermelha que pisca enquanto o nível é ALERTA; escondida nos demais."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._on = False
        self._timer = QTimer(self)
        self._timer.setInterval(BLINK_MS)
        self._timer.timeout.connect(self._blink)
        self.hide()

    def set_state(self, level, reasons=()):
        """Chame a cada quadro com state.level e state.reasons."""
        if level == FatigueLevel.ALERTA:
            self.setText("⚠ ALERTA: " + " · ".join(reasons) if reasons else "⚠ ALERTA")
            if not self._timer.isActive():
                self._on = True
                self._apply()
                self.show()
                self._timer.start()
        elif self._timer.isActive() or self.isVisible():
            self._timer.stop()
            self.hide()

    def _blink(self):
        self._on = not self._on
        self._apply()

    def _apply(self):
        self.setStyleSheet(STYLE_ON if self._on else STYLE_OFF)
