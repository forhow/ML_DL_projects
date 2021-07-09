import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

'''
- designer 실행
- ui 파일 생성
    > 버튼
'''

#
form_window = uic.loadUiType('./mainWidget.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.path = None
        self.setupUi(self)
        self.btn_exit.clicked.connect(QCoreApplication.instance().quit)
        self.model = load_model('../model/catNdog_binary_classification.h5')

        self.btn_select.clicked.connect(self.predict_image)

    def predict_image(self):
        # file chooser load
        # 선택한 파일의 경로를 튜플로 리턴 (경로, 형식)
        self.path = QFileDialog.getOpenFileName(
            self, "Open file",'../datasets/train',
            "Image Files(*.jpg);;All Files(*.*)",
            '../datasets/train',
        )
        print(self.path)

        if self.path[0]:
            # 파일 열기 - 필요없음
            # with open(self.path[0], 'r') as f:

            # 로드한 파일을 pixmap으로 전환
            pixmap = QPixmap(self.path[0])
            self.lbl_image.setPixmap(pixmap)

            try:
                img = Image.open(self.path[0])
                img = img.convert('RGB')
                img = img.resize((64,64))
                data = np.asarray(img)
                data = data / 255
                data = data.reshape(1,64,64,3)
            except:
                print('error')

            predict_value = self.model.predict(data)[0][0]
            print(predict_value)

            if predict_value >0.5:
                self.lbl_predict.setText('이 이미지는 ' + str((predict_value*100).round()) + '% 확률로 Dog 입니다.')
            else:
                self.lbl_predict.setText('이 이미지는 ' + str(((1-predict_value)*100).round()) + '% 확률로 Cat 입니다.')


app = QApplication(sys.argv)
mainWindow = Exam()
mainWindow.show()
sys.exit(app.exec_())

