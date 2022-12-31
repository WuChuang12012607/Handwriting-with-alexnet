import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import word_to_letters as mm



class picture(QWidget):

    def __init__(self, Parent = None):
        super(picture, self).__init__(Parent)
        self.resize(800, 700)
        self.setWindowTitle("图片模式")

        self.label = QLabel(self)
        self.label.setFixedSize(500, 500)
        self.label.move(20, 100)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                 )
        self.__btn_Save = QPushButton("预测")
        self.__btn_Save.setParent(self)

        self.__btn_Save.setFixedSize(125, 125)
        self.__btn_Save.move(600, 250)

        self.__textbox = QLineEdit(self)
        # self.__textbox.move(20, 20)
        self.__textbox.resize(130,25)
        self.__textbox.setReadOnly(True)
        self.__textbox.move(600,400)
        btn = QPushButton(self)
        btn.setText("导入图片")
        btn.move(20, 30)
        btn.clicked.connect(self.openimage)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "Image Files(*.jpg *.png)")
        jpg = QtGui.QPixmap(imgName)
        jpg2 = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg2)
        jpg.save('wsy.png')


    def on_btn_Save_Clicked(self):  # 按钮点击事件触发
        ans = mm.final_read("wsy.png")  # 调用函数进行预测
        self.__textbox.setText(ans)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())
