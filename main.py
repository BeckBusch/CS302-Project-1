from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
import time, datetime, math
import zipfile, wget
import guiCode
import threading
import torchvision
from torchvision import transforms
import os
import idx2numpy
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

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
    image = 'emnist-byclass-test-images-idx3-ubyte'
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
        for j in range(9):

            transform=torchvision.transforms.Compose([
                lambda x: transforms.functional.rotate(x, -90),
                lambda x: transforms.functional.hflip(x),
                ])
            
            PIL_image = Image.fromarray(image_array[pos + j + 9 * i].astype('uint8'), 'L')
            PIL_image = transform(PIL_image)
            qim = ImageQt(PIL_image)

            qpix = QtGui.QPixmap.fromImage(qim)
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

