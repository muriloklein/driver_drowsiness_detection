import sys

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()
window.webcam_btn.setChecked(True)
sys.exit(app.exec())
