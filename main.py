from PyQt5 import QtGui, QtWidgets
import time, datetime, math
import zipfile, wget
import guiCode
import torchvision
from PyQt5.QtCore import Qt
from torchvision import transforms
import idx2numpy
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

urlLink = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
destination = "gzip.zip"

progressCheck = 0
timeTracker = 0
timeString = ""

last_x = None
last_y = None
paintCanvas = None

prev = 0

image_array = []        

def train_picture_list():
    pictureFolder = "C:\\Users\\Samuel Mason\\Downloads\\Leaf\\"
    image = 'emnist-byclass-test-images-idx3-ubyte'
    image_array = idx2numpy.convert_from_file(image)
    ui.tableWidget.setRowCount(math.floor(len(image_array) / 14) + 1)
    ui.itemCount.setText(f"Total number of images: {len(image_array)}")
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

def emnistDownload():
    global progressCheck

    #ui.cancelButton.setEnabled(True)
    progressCheck = 0
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

def cancelDownload():
    pass

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

def test(e):
    pass
    #print("pogdfgf")
    #print(e)
    #ui.downloadButton.setText("tstst")

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

            if (14 * (pos + rows) + cols >= math.floor(len(image_array))):
                ui.currentCount.setText(f"Images currently on screen: {112 + cols}")
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
    image_array = train_picture_list()
    update_pos()

    ui.downloadButton.clicked.connect(emnistDownload)
    ui.clearCanvaButton.clicked.connect(clearCanvas)
    ui.submitCanvasButton.clicked.connect(saveImage)
    ui.tableWidget.verticalScrollBar().valueChanged.connect(update_pos)
   

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