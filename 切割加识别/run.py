import sys
import main as u1
from PyQt5.QtWidgets import *
import Recognition1 as u2
import image as u3


class FirstUi(QMainWindow):
    
    def __init__(self, parent=None):
        super(FirstUi, self).__init__(parent)
        self.ui = u1.Ui_MainWindow()
        self.ui.setupUi(self)

    def slot1(self):
        win.hide()
        win2.show()

    def slot2(self):
        win.hide()
        win3.show()


class SecondUi(QWidget):

    def __init__(self, parent=None):
        super(SecondUi, self).__init__(parent)
        a = u2.MainWidget(self)
        self.btn = QPushButton("返回", self)
        self.btn.move(680, 700)
        a.resize(700, 700)

    def Quit(self):
        win2.hide()
        win.show()

class ThirdUi(QWidget):

    def __init__(self, parent=None):
        super(ThirdUi, self).__init__(parent)
        a = u3.picture(self)
        self.btn = QPushButton("返回", self)
        self.btn.move(600, 580)
        self.btn.resize(130, 30)
        a.resize(800, 700)

    def Quit(self):
        win2.hide()
        win.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FirstUi()
    win.show()
    win2 = SecondUi()
    win3 = ThirdUi()
    win2.btn.clicked.connect(win.show)
    win2.btn.clicked.connect(win2.close)
    win3.btn.clicked.connect(win.show)
    win3.btn.clicked.connect(win3.close)
    sys.exit(app.exec_())