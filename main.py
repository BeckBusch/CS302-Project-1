import torchvision
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import math
import zipfile
import wget
import guiCode

urlLink = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
destination = "gzip.zip"

def bar_custom(current, total, width=80):
    progress = math.ceil((current/total) * 100)
    ui.progressBar.setProperty("value", progress)

def emnistDownload():
    print("check")
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
    ui.downloadButton.clicked.connect(emnistDownload)

def test():
    print("pogdfgf")
    ui.downloadButton.setText("tstst")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = guiCode.Ui_MainWindow()
    ui.setupUi(MainWindow)
    guiEdits()
    MainWindow.show()
    sys.exit(app.exec_())

