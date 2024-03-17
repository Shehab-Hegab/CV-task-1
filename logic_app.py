# this file contains the functions for threshholding and for Hybrid Images
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui,QtCore
from PyQt5.uic import loadUi
import numpy as np
import cv2
import cv2 as cv
from Filtering import Ui_MainWindow

class MainWindow_2(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow_2, self).__init__()
        self.setupUi(self)
        # Connect the load image button to thresholding Fuction
        self.pushButton_Normalize_load_4.clicked.connect(self.load_image_for_thresholding)
        self.pushButton_Normalize_4.clicked.connect(lambda: self.set_thresholding_method('global'))
        self.pushButton_Normalize_3.clicked.connect(lambda: self.set_thresholding_method('local'))
        # Connect the load image buttons to their respective functions
        self.pushButton_Hyprid_load_3.clicked.connect(lambda: self.load_image_for_hyprid(1))
        self.pushButton_Hyprid_load_4.clicked.connect(lambda: self.load_image_for_hyprid(2))
        # Connect the generate hyprid image button to the function
        self.pushButton_Hyprid_load_5.clicked.connect(self.generate_hyprid_and_display)
        self.loaded_images = [None, None]
    def load_image_for_thresholding(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_path)

            # Set the scaling mode to keep the aspect ratio and fit within the label
            scaled_pixmap = pixmap.scaled(self.label_Normalize_input_4.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            # Set the alignment to center within the label
            self.label_Normalize_input_4.setAlignment(QtCore.Qt.AlignCenter)

            # Clear any existing content from the label
            self.label_Normalize_input_4.clear()

            # Display the scaled pixmap on the output label
            self.label_Normalize_input_4.setPixmap(scaled_pixmap)

            # Store the loaded image for Thresholding
            self.loaded_image = pixmap.toImage()  # Convert QPixmap to QImage

    def local_thresholding(self):
        # Check if an image has been loaded
        if not hasattr(self, 'loaded_image'):
            QMessageBox.warning(self, "Warning", "Please load an image before thresholding.")
            return

        # Convert QImage data to numpy array
        image_format = QImage.Format_RGB888 if self.loaded_image.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        image = self.loaded_image.convertToFormat(image_format)
        if image_format == QImage.Format_RGB888:
            channels = 3
        else:
            channels = 4
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_array = np.array(ptr).reshape(image.height(), image.width(), channels)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Apply local thresholding
        block_size = 11  # Example value
        constant = 2     # Example value
        thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

        # Display the thresholded image
        self.display_thresholded_image(thresholded_image)
        
    def global_thresholding(self):
        # Check if an image has been loaded
        if not hasattr(self, 'loaded_image'):
            QMessageBox.warning(self, "Warning", "Please load an image before thresholding.")
            return

        # Convert QImage data to numpy array
        image_format = QImage.Format_RGB888 if self.loaded_image.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        image = self.loaded_image.convertToFormat(image_format)
        if image_format == QImage.Format_RGB888:
            channels = 3
        else:
            channels = 4
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_array = np.array(ptr).reshape(image.height(), image.width(), channels)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Apply global thresholding using Otsu's method
        _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Display the thresholded image
        self.display_thresholded_image(thresholded_image)

    def display_thresholded_image(self, thresholded_image):
        # Convert the thresholded image to 8-bit grayscale if necessary
        if len(thresholded_image.shape) == 3:
            thresholded_image = cv2.cvtColor(thresholded_image, cv2.COLOR_BGR2GRAY)

        # Convert thresholded image to QImage
        height, width = thresholded_image.shape
        bytes_per_line = width
        q_image = QImage(thresholded_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # Convert QImage to QPixmap for display
        thresholded_pixmap = QPixmap.fromImage(q_image)

        # Display the thresholded image on the output label
        self.label_Normalize_output_5.setPixmap(thresholded_pixmap.scaled(self.label_Normalize_output_5.size()))

    def set_thresholding_method(self, method):
        # Set the thresholding method
        self.thresholding_method = method
        
        # Call the thresholding function based on the selected method
        if method == 'global':
            self.global_thresholding()
        elif method == 'local':
            self.local_thresholding()
        else:
            print("Invalid thresholding method specified.")

    def threshold_and_display(self):
        # Check if an image has been loaded
        if not hasattr(self, 'loaded_image'):
            QMessageBox.warning(self, "Warning", "Please load an image before thresholding.")
            return

        # Convert loaded image to numpy array
        image_format = QImage.Format_RGB888 if self.loaded_image.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        image = self.loaded_image.convertToFormat(image_format)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_array = np.array(ptr).reshape(image.height(), image.width(), 3)  # Assuming RGB image

        # Perform thresholding based on the selected method
        try:
            if self.thresholding_method == 'global':
                thresholded_image = self.global_thresholding(image)
            elif self.thresholding_method == 'local':
                block_size = 11  # Example value
                constant = 2     # Example value
                thresholded_image = self.local_thresholding(image, block_size, constant)
            else:
                raise ValueError("Invalid thresholding method specified.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during thresholding: {str(e)}")
            return

        if thresholded_image is not None:
            # Convert thresholded image to QPixmap for display
            thresholded_qimage = QImage(thresholded_image.data, thresholded_image.shape[1], thresholded_image.shape[0], thresholded_image.strides[0], QImage.Format_RGB888)
            thresholded_pixmap = QPixmap.fromImage(thresholded_qimage)

            # Display the thresholded image on the output label
            self.label_Normalize_output_5.setPixmap(thresholded_pixmap.scaled(self.label_Normalize_output_5.size()))

    def load_image_for_hyprid(self, image_index):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_path)

            # Display the loaded image on the appropriate label
            if image_index == 1:
                self.label_Hyprid_input_4.setPixmap(pixmap.scaled(self.label_Hyprid_input_4.size()))
            elif image_index == 2:
                self.label_Hyprid_input_3.setPixmap(pixmap.scaled(self.label_Hyprid_input_3.size()))

            # Store the loaded image for hyprid image generation
            self.loaded_images[image_index - 1] = pixmap.toImage() 

    def generate_hyprid_and_display(self):
        # Check if both images have been loaded
        if None in self.loaded_images:
            QMessageBox.warning(self, "Warning", "Please load both images before generating a hyprid image.")
            return

        try:
            # Generate hyrid image
            low_pass, high_pass, hyprid_image = self.generate_hyprid_image(self.loaded_images[0], self.loaded_images[1])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during hyprid image generation: {str(e)}")
            return

        if hyprid_image is not None:
            try:
                # Convert Hyprid image to QPixmap for display
                hyprid_qimage = QImage(hyprid_image.data, hyprid_image.shape[1], hyprid_image.shape[0], hyprid_image.strides[0], QImage.Format_RGB888)
                hyprid_pixmap = QPixmap.fromImage(hyprid_qimage)
                
                # Debug print to check the pixmap size
                print("Hyprid Pixmap Size:", hyprid_pixmap.size())
                
                # Display the hyprid image on the output label
                self.label_Hyprid_output_2.setPixmap(hyprid_pixmap.scaled(self.label_Hyprid_output_2.size()))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred during displaying hyprid image: {str(e)}")
                print("Error:", e)

    def generate_hyprid_image(self, image1, image2):
        # Convert QImage data to numpy array for both images
        image1_format = QImage.Format_RGB888 if image1.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        image1 = image1.convertToFormat(image1_format)
        if image1_format == QImage.Format_RGB888:
            channels1 = 3
        else:
            channels1 = 4
        ptr1 = image1.bits()
        ptr1.setsize(image1.byteCount())
        image1_array = np.array(ptr1).reshape(image1.height(), image1.width(), channels1)

        image2_format = QImage.Format_RGB888 if image2.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        image2 = image2.convertToFormat(image2_format)
        if image2_format == QImage.Format_RGB888:
            channels2 = 3
        else:
            channels2 = 4
        ptr2 = image2.bits()
        ptr2.setsize(image2.byteCount())
        image2_array = np.array(ptr2).reshape(image2.height(), image2.width(), channels2)

        # Apply Gaussian blur to the first image
        low_pass = cv2.GaussianBlur(image1_array, (15, 15), 0)

        # Apply Laplacian filter to the second image
        high_pass = cv2.subtract(image2_array, cv2.GaussianBlur(image2_array, (15, 15), 0))

        # Combine the images to create the hyprid image
        hyprid_image = cv2.add(low_pass, high_pass)

        return low_pass, high_pass, hyprid_image
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow_2()
    main_window.show()
    sys.exit(app.exec_())