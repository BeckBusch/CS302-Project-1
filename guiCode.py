# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'projectGUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import math


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(793, 812)
        MainWindow.setFixedSize(MainWindow.size())
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabSwitcher = QtWidgets.QTabWidget(self.centralwidget)
        self.tabSwitcher.setGeometry(QtCore.QRect(0, 0, 791, 551))
        self.tabSwitcher.setObjectName("tabSwitcher")
        self.prediction = QtWidgets.QWidget()
        self.prediction.setObjectName("prediction")

        self.clearCanvaButton = QtWidgets.QPushButton(self.prediction)
        self.clearCanvaButton.setGeometry(QtCore.QRect(20, 320, 81, 23))
        self.clearCanvaButton.setObjectName("clearCanvaButton")
        self.submitCanvasButton = QtWidgets.QPushButton(self.prediction)
        self.submitCanvasButton.setGeometry(QtCore.QRect(144, 320, 131, 23))
        self.submitCanvasButton.setObjectName("submitCanvasButton")
        self.canvasLabel = QtWidgets.QLabel(self.prediction)
        self.canvasLabel.setGeometry(QtCore.QRect(20, 60, 254, 250))
        self.canvasLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.canvasLabel.setLineWidth(2)
        self.canvasLabel.setText("")
        self.canvasLabel.setObjectName("canvasLabel")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.prediction)
        self.textBrowser_2.setGeometry(QtCore.QRect(290, 200, 191, 141))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.picture = QtWidgets.QLabel(self.prediction)
        self.picture.setGeometry(QtCore.QRect(290, 80, 41, 41))
        self.picture.setFrameShape(QtWidgets.QFrame.Box)
        self.picture.setText("")
        self.picture.setPixmap(QtGui.QPixmap("Images/sevenPicture.png"))
        self.picture.setAlignment(QtCore.Qt.AlignCenter)
        self.picture.setObjectName("picture")
        self.picture_2 = QtWidgets.QLabel(self.prediction)
        self.picture_2.setGeometry(QtCore.QRect(340, 80, 41, 41))
        self.picture_2.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_2.setText("")
        self.picture_2.setPixmap(QtGui.QPixmap("Images/bPicture.png"))
        self.picture_2.setObjectName("picture_2")
        self.picture_3 = QtWidgets.QLabel(self.prediction)
        self.picture_3.setGeometry(QtCore.QRect(390, 80, 41, 41))
        self.picture_3.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_3.setText("")
        self.picture_3.setPixmap(QtGui.QPixmap("Images/qPicture.png"))
        self.picture_3.setObjectName("picture_3")
        self.picture_4 = QtWidgets.QLabel(self.prediction)
        self.picture_4.setGeometry(QtCore.QRect(440, 80, 41, 41))
        self.picture_4.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_4.setText("")
        self.picture_4.setPixmap(QtGui.QPixmap("Images/mPicture.png"))
        self.picture_4.setObjectName("picture_4")
        self.picture_5 = QtWidgets.QLabel(self.prediction)
        self.picture_5.setGeometry(QtCore.QRect(290, 130, 41, 41))
        self.picture_5.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_5.setText("")
        self.picture_5.setPixmap(QtGui.QPixmap("Images/sPicture.png"))
        self.picture_5.setObjectName("picture_5")
        self.picture_6 = QtWidgets.QLabel(self.prediction)
        self.picture_6.setGeometry(QtCore.QRect(340, 130, 41, 41))
        self.picture_6.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_6.setText("")
        self.picture_6.setPixmap(QtGui.QPixmap("Images/sixPicture.png"))
        self.picture_6.setObjectName("picture_6")
        self.picture_7 = QtWidgets.QLabel(self.prediction)
        self.picture_7.setGeometry(QtCore.QRect(390, 130, 41, 41))
        self.picture_7.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_7.setText("")
        self.picture_7.setPixmap(QtGui.QPixmap("Images/fourPicture.png"))
        self.picture_7.setObjectName("picture_7")
        self.picture_8 = QtWidgets.QLabel(self.prediction)
        self.picture_8.setGeometry(QtCore.QRect(440, 130, 41, 41))
        self.picture_8.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_8.setText("")
        self.picture_8.setPixmap(QtGui.QPixmap("Images/tPicture.png"))
        self.picture_8.setObjectName("picture_8")
        self.label_6 = QtWidgets.QLabel(self.prediction)
        self.label_6.setGeometry(QtCore.QRect(290, 60, 61, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.prediction)
        self.label_7.setGeometry(QtCore.QRect(290, 180, 61, 16))
        self.label_7.setObjectName("label_7")


        self.tabSwitcher.addTab(self.prediction, "")
        self.dataSet = QtWidgets.QWidget()
        self.dataSet.setObjectName("dataSet")
        self.downloadButton = QtWidgets.QPushButton(self.dataSet)
        self.downloadButton.setGeometry(QtCore.QRect(30, 150, 101, 23))
        self.downloadButton.setObjectName("downloadButton")
        self.progressBar = QtWidgets.QProgressBar(self.dataSet)
        self.progressBar.setEnabled(False)
        self.progressBar.setGeometry(QtCore.QRect(30, 180, 331, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setObjectName("progressBar")
        self.timeRemainingLabel = QtWidgets.QLabel(self.dataSet)
        self.timeRemainingLabel.setEnabled(False)
        self.timeRemainingLabel.setGeometry(QtCore.QRect(40, 210, 81, 16))
        self.timeRemainingLabel.setObjectName("timeRemainingLabel")
        self.timeRemaininCount = QtWidgets.QLabel(self.dataSet)
        self.timeRemaininCount.setGeometry(QtCore.QRect(130, 210, 141, 16))
        self.timeRemaininCount.setObjectName("timeRemaininCount")
        self.cancelButton = QtWidgets.QPushButton(self.dataSet)
        self.cancelButton.setEnabled(False)
        self.cancelButton.setGeometry(QtCore.QRect(220, 150, 101, 23))
        self.cancelButton.setObjectName("cancelButton")

        self.tabSwitcher.addTab(self.dataSet, "")
        self.viewDataset = QtWidgets.QWidget()
        self.viewDataset.setObjectName("viewDataset")
        self.tabSwitcher.addTab(self.viewDataset, "")

        self.viewDataset.layout = QtWidgets.QVBoxLayout()

        self.inputLine = QtWidgets.QLineEdit()

        self.searchBar = QtWidgets.QHBoxLayout()
        self.searchBar.addStretch(1)
        self.searchBar.addWidget(self.inputLine)

        self.viewDataset.layout.addLayout(self.searchBar)
        self.scrollArea = QtWidgets.QScrollArea(widgetResizable = True)

        # self.vertScrollBar.setValue(100)

        # self.scrollArea.setMouseTracking(True)
        # self.scrollArea.horizontalScrollBar().setEnabled(False)


        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tableWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.tableWidget.setRowCount(9)
        self.tableWidget.setColumnCount(14)

        self.tableWidget.verticalHeader().setDefaultSectionSize(50)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(50)


        # self.pic = QtGui.QPixmap("C:\\Users\\Samuel Mason\\Downloads\\28pix.png")
        # self.pic = QtGui.QPixmap("C:\\Users\\Samuel Mason\\Downloads\\28pix2.png")
        # self.testWidget = QtWidgets.QPushButton()

###############
        # self.pictureFolder = "C:\\Users\\Samuel Mason\\Downloads\\Leaf\\"
        # self.pictureList = []

        # for images in os.listdir(self.pictureFolder):
        #     self.pictureList.append(images)

        # for i in range(2):
        #     print(self.pictureList[i])

        # for i in range(7):
        #     for j in range(15):

        #         self.this_image = self.pictureList[i + j]
        #         self.pic = QtGui.QPixmap(self.pictureFolder + self.this_image)
        #         self.label = QtWidgets.QLabel("test")
        #         self.label.setPixmap(self.pic)
        #         self.tableWidget.setCellWidget(j, i, self.label)
                          

        self.scrollArea.setWidget(self.tableWidget)

        # self.scrollZone.setWidget(self.scrollArea)

        self.viewDataset.setLayout(self.viewDataset.layout)

        self.viewDataset.layout.addWidget(self.scrollArea)

        self.currentCount = QtWidgets.QLabel()
        self.viewDataset.layout.addWidget(self.currentCount)

        self.itemCount = QtWidgets.QLabel()
        self.viewDataset.layout.addWidget(self.itemCount)





        # image_contents_widget = QtWidgets.QWidget()
        # self.scrollArea.setWidget(image_contents_widget)
        # self.page_layout = QtWidgets.QVBoxLayout(image_contents_widget)


        # contents_width = self.scrollArea.frameGeometry().width()
        # contents_height = self.scrollArea.frameGeometry().height()

        # self.tableWidget = QtGui.QTableWidget()

        # column_count = math.floor(contents_width / 28)
        # row_count = math.floor(814255 / column_count) + 1

        # self.tableWidget.setRowCount(row_count)
        # self.tableWidget.setColumnCount(column_count)

        # for image in os.listdir(DIRECTORY):
        #     pixmap = QtGui.QPixmap(os.path.join(DIRECTORY, file location))
        #     if not pixmap.isNull():
        #         label = QtGui.QLabel(pixmap = pixmap)
        #         page_layout.addWidget(label)

        # width / image_size (floor) = images horiz
        # height / image_size (floor) = images vert


        self.training = QtWidgets.QWidget()
        self.training.setObjectName("training")
        self.modelSelector = QtWidgets.QComboBox(self.training)
        self.modelSelector.setGeometry(QtCore.QRect(130, 150, 201, 22))
        self.modelSelector.setObjectName("modelSelector")
        self.modelSelector.addItem("")
        self.modelSelector.addItem("")
        self.modelSelector.addItem("")
        self.iterationsInput = QtWidgets.QSpinBox(self.training)
        self.iterationsInput.setGeometry(QtCore.QRect(130, 180, 61, 22))
        self.iterationsInput.setObjectName("iterationsInput")
        self.TrainingSlider = QtWidgets.QSlider(self.training)
        self.TrainingSlider.setGeometry(QtCore.QRect(130, 210, 151, 22))
        self.TrainingSlider.setOrientation(QtCore.Qt.Horizontal)
        self.TrainingSlider.setInvertedAppearance(False)
        self.TrainingSlider.setInvertedControls(False)
        self.TrainingSlider.setObjectName("TrainingSlider")
        self.TrainingProgressBar = QtWidgets.QProgressBar(self.training)
        self.TrainingProgressBar.setGeometry(QtCore.QRect(30, 280, 301, 23))
        self.TrainingProgressBar.setProperty("value", 24)
        self.TrainingProgressBar.setTextVisible(True)
        self.TrainingProgressBar.setObjectName("TrainingProgressBar")
        self.DatasetSelector = QtWidgets.QComboBox(self.training)
        self.DatasetSelector.setGeometry(QtCore.QRect(130, 120, 201, 22))
        self.DatasetSelector.setObjectName("DatasetSelector")
        self.DatasetSelector.addItem("")
        self.label_2 = QtWidgets.QLabel(self.training)
        self.label_2.setGeometry(QtCore.QRect(30, 210, 91, 16))
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.training)
        self.label_3.setGeometry(QtCore.QRect(40, 120, 81, 16))
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.training)
        self.label_4.setGeometry(QtCore.QRect(40, 150, 81, 16))
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.training)
        self.label_5.setGeometry(QtCore.QRect(40, 180, 81, 16))
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.TrainingValue = QtWidgets.QLabel(self.training)
        self.TrainingValue.setGeometry(QtCore.QRect(290, 210, 71, 16))
        self.TrainingValue.setObjectName("TrainingValue")
        self.startTrainingButton = QtWidgets.QPushButton(self.training)
        self.startTrainingButton.setGeometry(QtCore.QRect(30, 250, 91, 23))
        self.startTrainingButton.setObjectName("startTrainingButton")
        self.tabSwitcher.addTab(self.training, "")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 580, 771, 192))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 560, 81, 16))
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 793, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabSwitcher.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.clearCanvaButton.setText(_translate("MainWindow", "Clear Canvas"))
        self.submitCanvasButton.setText(_translate("MainWindow", "Submit Drawing"))
        self.label_6.setText(_translate("MainWindow", "Examples:"))
        self.label_7.setText(_translate("MainWindow", "Predictions:"))
        self.tabSwitcher.setTabText(self.tabSwitcher.indexOf(self.prediction), _translate("MainWindow", "Prediction"))
        self.downloadButton.setText(_translate("MainWindow", "Begin Download"))
        self.timeRemainingLabel.setText(_translate("MainWindow", "Time Remaininig:"))
        self.timeRemaininCount.setText(_translate("MainWindow", "XX Minutes and XX Seconds"))
        self.cancelButton.setText(_translate("MainWindow", "Cancel Download"))
        self.tabSwitcher.setTabText(self.tabSwitcher.indexOf(self.dataSet), _translate("MainWindow", "DataSet"))
        self.tabSwitcher.setTabText(self.tabSwitcher.indexOf(self.viewDataset), _translate("MainWindow", "View DataSet"))
        self.currentCount.setText("Images currently on-screen: 126")
        self.modelSelector.setItemText(0, _translate("MainWindow", "LeNet"))
        self.modelSelector.setItemText(1, _translate("MainWindow", "AlexNet"))
        self.modelSelector.setItemText(2, _translate("MainWindow", "VGG11"))
        self.DatasetSelector.setItemText(0, _translate("MainWindow", "EMNIST"))
        self.label_2.setText(_translate("MainWindow", "Training Dataset:"))
        self.label_3.setText(_translate("MainWindow", "Dataset:"))
        self.label_4.setText(_translate("MainWindow", "ML Model:"))
        self.label_5.setText(_translate("MainWindow", "Iterations:"))
        self.TrainingValue.setText(_translate("MainWindow", "99% Training"))
        self.startTrainingButton.setText(_translate("MainWindow", "Start Training"))
        self.tabSwitcher.setTabText(self.tabSwitcher.indexOf(self.training), _translate("MainWindow", "Training"))
        self.label.setText(_translate("MainWindow", "Program Status"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
