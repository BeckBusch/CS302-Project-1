from PyQt5 import QtCore, QtGui, QtWidgets
import time, datetime, math
import zipfile, wget
import guiCode
import threading
import torchvision

urlLink = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
destination = "gzip.zip"

progressCheck = 0
timeTracker = 0
timeString = ""

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


def clearCanvas():
    ui.canvas.fill(QtGui.QColor("white"))

def guiEdits():
    #ui.cancelButton.clicked.connect(test)
    ui.downloadButton.clicked.connect(emnistDownload)
    ui.clearCanvaButton.clicked.connect(clearCanvas)

    ui.canvasLabel = QtWidgets.QLabel(ui.prediction)
    
    #ui.canvasLabel.setMouseTracking(True)
    ui.canvasLabel.setGeometry(QtCore.QRect(20, 60, 250, 250))

    ui.canvas = QtGui.QPixmap(250, 250)
    ui.canvas.fill(QtGui.QColor("blue"))
    ui.canvasLabel.setPixmap(ui.canvas)

    ui.canvasLabel.mouseMoveEvent = test


def test(e):
    print("pogdfgf")
    print(e)
    ui.downloadButton.setText("tstst")
    print(QtGui.QCursor.pos())

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = guiCode.Ui_MainWindow()
    ui.setupUi(MainWindow)
    guiEdits()
    MainWindow.show()
    sys.exit(app.exec_())

