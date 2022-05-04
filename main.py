from PyQt5 import QtGui, QtWidgets
import time, datetime, math
import zipfile, wget
import guiCode
import torchvision
from PyQt5.QtCore import Qt
import torch
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import idx2numpy
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
import neuralnet
import os
import gzip
import shutil

progressCheck = 0
timeTracker = 0
timeString = ""

last_x = None
last_y = None
paintCanvas = None

prev = 0

image_array = [] 
dataset_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emnist-byclass-train-images-idx3-ubyte')
full_dataset = open(dataset_location)

#emnist download link
urlLink = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
#destination for emnist zip file in root
destination = "gzip.zip"
# location of the prefered traininig file once uinzipped
trainLocation = "gzip\\emnist-byclass-test-images-idx3-ubyte.gz"
#place to store the unzipped training file for use by other parts of the code
trainDestination = "emnist-byclass-test-images-idx3-ubyte"

def picture_list_test():
    global image_array
    image = 'emnist-byclass-test-images-idx3-ubyte'
    picture_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), image)
    image_array = idx2numpy.convert_from_file(picture_location)
    ui.tableWidget.setRowCount(math.floor(len(image_array) / 14) + 1)
    ui.itemCount.setText(f"Total number of images: {len(image_array)}")
    update_pos()

def picture_list_train():
    global image_array
    image = 'emnist-byclass-train-images-idx3-ubyte'
    picture_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), image)
    image_array = idx2numpy.convert_from_file(picture_location)
    ui.tableWidget.setRowCount(math.floor(len(image_array) / 14) + 1)
    ui.itemCount.setText(f"Total number of images: {len(image_array)}")
    update_pos()

def emnistDownload():
    global progressCheck, full_dataset

    #ui.cancelButton.setEnabled(True)
    progressCheck = 0
    #wget.download(urlLink, bar=bar_custom)

    ui.timeRemainingLabel.setText("Unzipping:")
    zf = zipfile.ZipFile(destination)
    uncompress_size = sum((file.file_size for file in zf.infolist()))
    extracted_size = 0
    for file in zf.infolist():
        extracted_size += file.file_size
        progress = math.ceil((extracted_size/uncompress_size) * 100)
        ui.progressBar.setProperty("value", progress)
        zf.extract(file)

    ui.timeRemainingLabel.setText("Unzipping Data:")
    with gzip.open(trainLocation, 'rb') as f_in:
        with open(trainDestination, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    full_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "\\emnist-byclass-train-images-idx3-ubyte")

def bar_custom(current, total, width=80):
    global progressCheck, timeTracker, timeString
    timeElapsed = 0

    progress = math.ceil((current/total) * 100)

    if (progress != progressCheck):
        progressCheck = progress
        timeElapsed = time.time() - timeTracker
        timeElapsed = timeElapsed * (100-progress)
        timeElapsed = math.ceil(timeElapsed)
        timeTracker = time.time()
        timeString = str(datetime.timedelta(seconds=timeElapsed))

    ui.progressBar.setProperty("value", progress)
    ui.timeRemaininCount.setText(timeString)

def cancelDownload():
    pass

def model_selector(selected_model):
    global transform 
    transform = neuralnet.model_selector(selected_model)
    import_state = neuralnet.import_model("no")

def train_button():
    if 1 == 1:
        training_loader, train_size, _ = neuralnet.loaders(full_dataset, transform)# if import_state == True:
        train(1, training_loader, train_size)
    else:
        pass
 
def train(epoch, training_loader, train_size):
    neuralnet.train(epoch, training_loader, train_size)

def test(testing_loader):
    neuralnet.test(testing_loader)

def clearCanvas():
    paintCanvas.fill(QtGui.QColor("white"))
    ui.canvasLabel.setPixmap(paintCanvas)

def painting(e):
    global last_x, last_y

    if last_x is None:
        last_x = e.x()
        last_y = e.y()
        return

    painter = QtGui.QPainter(paintCanvas)
    p = painter.pen()
    p.setWidth(12)
    p.setColor(QtGui.QColor("black"))
    painter.setPen(p)
    #painter.drawLine(last_x+580, last_y+220, e.x()+580, e.y()+220)
    painter.drawEllipse(e.x(), e.y(), 12, 12)
    # painter.drawLine(last_x, last_y, e.x(), e.y())
    painter.end()
    ui.canvasLabel.setPixmap(paintCanvas)

    # Update the origin for next time.
    last_x = e.x()
    last_y = e.y()

def mouseRel(e):
    global last_x, last_y
    last_x = None
    last_y = None

def saveImage():
    print("test")
    channels_count = 4
    image = paintCanvas.toImage()
    image.save(r'tmp.png')

def update_pos():
    global prev
    pos = ui.tableWidget.verticalScrollBar().value()
    for i in range(14):
        if pos > prev:
            for j in range(abs(pos - prev)):
                ui.tableWidget.removeCellWidget(pos - j - 1, i)
        else:
            for j in range(abs(pos - prev)):
                ui.tableWidget.removeCellWidget(prev + 15 - j - 1, i)

    for rows in range(9):
        for cols in range(14):

            if (14 * (pos + rows) + cols >= len(image_array)):
                ui.currentCount.setText(f"Images currently on screen: {len(image_array) - math.floor(len(image_array) / 14) * 14 + 112}")
                break;
            ui.currentCount.setText(f"Images currently on screen: 126")

            transform=torchvision.transforms.Compose([
                lambda x: transforms.functional.rotate(x, -90),
                lambda x: transforms.functional.hflip(x),
                ])
            
            PIL_image = Image.fromarray(image_array[14 * (pos + rows) + cols].astype('uint8'), 'L')
            PIL_image = transform(PIL_image)
            qim = ImageQt(PIL_image)

            qpix = QtGui.QPixmap.fromImage(qim)
            qpix = qpix.scaled(45, 45, Qt.KeepAspectRatio, Qt.FastTransformation)
            label = QtWidgets.QLabel("")
            label.setPixmap(qpix)
            ui.tableWidget.setCellWidget(pos + rows, cols, label)

    prev = pos

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = guiCode.Ui_MainWindow()
    ui.setupUi(MainWindow)
    picture_list_test()

    ui.downloadButton.clicked.connect(emnistDownload)
    ui.clearCanvaButton.clicked.connect(clearCanvas)
    ui.submitCanvasButton.clicked.connect(saveImage)
    ui.tableWidget.verticalScrollBar().valueChanged.connect(update_pos)
    ui.modelSelector.currentTextChanged.connect(model_selector)
    ui.startTrainingButton.clicked.connect(train_button)
    ui.testingImages.clicked.connect(picture_list_test)
    ui.trainingImages.clicked.connect(picture_list_train)
    #emnistDownload()
    #ui.canvasLabel = QtWidgets.QLabel(ui.prediction)
    #   ui.canvasLabel.setGeometry(QtCore.QRect(20, 60, 250, 250))
    paintCanvas = QtGui.QPixmap(250, 250)
    paintCanvas.fill(QtGui.QColor("white"))
    ui.canvasLabel.setPixmap(paintCanvas)
    ui.canvasLabel.mouseMoveEvent = painting
    ui.canvasLabel.mouseReleaseEvent = mouseRel
    
    MainWindow.show()
    MainWindow.update()
    sys.exit(app.exec_())