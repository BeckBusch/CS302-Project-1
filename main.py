from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import time, datetime, math
import zipfile, wget
import guiCode
import threading
import torchvision
import os
import idx2numpy

urlLink = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
destination = "gzip.zip"

progressCheck = 0
timeTracker = 0
timeString = ""

rowCount = 50
prev = 0

image_array = []

def train_picture_list():
    pictureFolder = "C:\\Users\\Samuel Mason\\Downloads\\Leaf\\"
    image = 'train-images-idx3-ubyte'
    image_array = idx2numpy.convert_from_file(pictureFolder + image)

    return image_array

def bar_custom(current, total, width=80):
    global progressCheck, timeTracker, timeString
    timeElasped = 0
    
    progress = math.ceil((current/total) * 100)


    if (progress != progressCheck):
        print("calc")
        progressCheck = progress
        timeElasped = time.time() - timeTracker
        timeElasped = timeElasped * (100-progress)
        timeElasped = math.ceil(timeElasped)
        timeTracker = time.time()
        timeString = str(datetime.timedelta(seconds=timeElasped))



    ui.progressBar.setProperty("value", progress)
    ui.timeRemaininCount.setText(timeString)

def update_pos():

    global prev
    pos = ui.tableWidget.verticalScrollBar().value()
    print(pos)
    for i in range(14):
        if pos > prev:
            for j in range(abs(pos - prev)):
                ui.tableWidget.removeCellWidget(pos - j - 1, i)
        else:
            for j in range(abs(pos - prev)):
                ui.tableWidget.removeCellWidget(prev + 15 - j - 1, i)

    for i in range(14):
        for j in range(15):

            height, width = 28, 28#image_array[i + j].shape
            bytesPerLine = 1 * width
            q_image = QtGui.QImage(image_array[pos + i * j + i + j].data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
            qpix = QtGui.QPixmap(q_image)
            qpix = qpix.scaled(45, 45, Qt.KeepAspectRatio, Qt.FastTransformation)
            label = QtWidgets.QLabel("")
            label.setPixmap(qpix)
            ui.tableWidget.setCellWidget(pos + j, i, label)

    prev = pos

def emnistDownload():
    global progressCheck

    #ui.cancelButton.setEnabled(True)

    progressCheck = 0
    #print("check")
    wget.download(urlLink, bar=bar_custom)
    
    ui.timeRemainingLabel.setText("Unzipping:")
    zf = zipfile.ZipFile(destination)
    uncompress_size = sum((file.file_size for file in zf.infolist()))
    extracted_size = 0
    for file in zf.infolist():
        extracted_size += file.file_size
        progress = math.ceil((extracted_size/uncompress_size) * 100)
        ui.progressBar.setProperty("value", progress)
        zf.extract(file)

def guiEdits():
    #ui.cancelButton.clicked.connect(test)
    ui.downloadButton.clicked.connect(emnistDownload)
    ui.tableWidget.verticalScrollBar().valueChanged.connect(update_pos)

def test():
    print("no")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = guiCode.Ui_MainWindow()
    ui.setupUi(MainWindow)
    image_array = train_picture_list()
    guiEdits()
    MainWindow.show()
    sys.exit(app.exec_())

