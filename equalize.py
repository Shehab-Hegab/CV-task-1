# this file contains equalize property
import sys
from PyQt5 import QtWidgets
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui, QtCore
from PyQt5.uic import loadUi
import numpy as np
import cv2 as cv
from Filtering import Ui_MainWindow
from PIL import Image
from matplotlib import pyplot as plt





class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_equalize_1.clicked.connect(self.load_image_for_equalize)
        self.pushButton_equalize_2.clicked.connect(self.equalize_image)
        
    def load_image_for_equalize(self):
        # Open file dialog to select an image
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")

        # Check if a file is selected
        if file_name:
            # Load the image
            image = cv2.imread(file_name)

            # Convert BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert image to QImage
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)

            # Set the pixmap to the QLabel
            self.label_equalize_input_3.setPixmap(pixmap)
            self.label_equalize_input_3.setScaledContents(True)

    def equalize_image(self):
        # Retrieve the original image from the label
        pixmap = self.label_equalize_input_3.pixmap()
        image = pixmap.toImage()

        # Convert the image to OpenCV format
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        img = np.array(ptr).reshape(height, width, 4)  # 4 channels for RGBA

        # Convert RGBA to RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Convert image to grayscale
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(gray_image)

        # Convert grayscale to RGB
        equalized_image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

        # Convert image to QImage
        height, width, channel = equalized_image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(equalized_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Set the pixmap to the QLabel
        self.label_equalize_output_3.setPixmap(pixmap)
        self.label_equalize_output_3.setScaledContents(True)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())