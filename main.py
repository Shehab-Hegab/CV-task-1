import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
import numpy as np
import cv2 as cv

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("Filtering.ui", self)  # Load your UI file

        # Connect the load image button to the function
        self.pushButton_Normalize_load_2.clicked.connect(self.load_image_for_normalization)

        # Connect the normalize button to the function
        self.pushButton_Normalize_2.clicked.connect(self.normalize_image_and_display)
        
        self.loaded_image = None  # Initialize loaded_image attribute

    def load_image_for_normalization(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_path)

            # Display the loaded image on the input label
            self.label_Normalize_input_2.setPixmap(pixmap.scaled(self.label_Normalize_input_2.size()))

            # Store the loaded image for later normalization
            self.loaded_image = pixmap.toImage()

    def normalize_image_and_display(self):
        # Check if an image has been loaded
        if self.loaded_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image before normalizing.")
            return

        try:
            # Perform normalization
            normalized_image = self.normalize_image(self.loaded_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during normalization: {str(e)}")
            return

        if normalized_image is not None:
            # Convert normalized image to QPixmap for display
            normalized_qimage = QImage(normalized_image.data, normalized_image.shape[1], normalized_image.shape[0], normalized_image.strides[0], QImage.Format_RGB888)
            normalized_pixmap = QPixmap.fromImage(normalized_qimage)

            # Display the normalized image on the output label
            self.label_Normalize_output_2.setPixmap(normalized_pixmap.scaled(self.label_Normalize_output_2.size()))

    def normalize_image(self, image):
        # Ensure the image has 3 or 4 channels (RGB or RGBA)
        image_format = QImage.Format_RGB888 if image.format() == QImage.Format_RGB32 else QImage.Format_RGBA8888
        image = image.convertToFormat(image_format)
        if image_format == QImage.Format_RGB888:
            channels = 3
        else:
            channels = 4

        # Convert QImage data to numpy array
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_array = np.array(ptr).reshape(image.height(), image.width(), channels)

        # Normalize using OpenCV's min-max normalization
        normalized_image = cv.normalize(image_array, None, 0, 255, cv.NORM_MINMAX)

        # Return the normalized image as a NumPy array
        return normalized_image

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
