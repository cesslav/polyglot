import sys
import io
import warnings
from translate_m2m import translate
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow
from qt_design import plan


class UI_app(QMainWindow):
    def __init__(self):
        super().__init__()
        f = io.StringIO(plan)
        uic.loadUi(f, self)  # Загружаем дизайн
        self.text2.setReadOnly(True)
        self.checkBtn.clicked.connect(self.check)

    def check(self):
        from_text = ((self.text1.toPlainText()).strip('\n').split('\n'))[0]
        self.text2.setPlaceholderText(translate(from_text)[0])


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)
    ex = UI_app()
    ex.show()
    sys.exit(app.exec())