from PyQt5 import QtCore, QtGui, QtWidgets
import time, datetime, math
import zipfile, wget
import guiCode
import threading
import torchvision
import numpy as np

urlLink = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
destination = "gzip.zip"

progressCheck = 0
timeTracker = 0
timeString = ""

last_x = None
last_y = None
paintCanvas = None

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
    p.setWidth(16)
    p.setColor(QtGui.QColor("black"))
    painter.setPen(p)
    #painter.drawLine(last_x+580, last_y+220, e.x()+580, e.y()+220)
    painter.drawLine(last_x, last_y, e.x(), e.y())
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

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = guiCode.Ui_MainWindow()
    ui.setupUi(MainWindow)

    ui.downloadButton.clicked.connect(emnistDownload)
    ui.clearCanvaButton.clicked.connect(clearCanvas)
    ui.submitCanvasButton.clicked.connect(saveImage)

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

