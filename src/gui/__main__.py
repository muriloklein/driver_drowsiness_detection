import sys

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()
window.start(0)
sys.exit(app.exec())
