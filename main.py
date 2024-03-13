import sys

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

    # Additive Noise:

    def uniform_noise(img, var):
        uniform_noise = np.random.randint(0, var, img.shape)
        new_image = img + uniform_noise
        new_image = np.clip(new_image, 0, 255)
        return new_image

    def gaussian_noise(img, var):
        mean = 0
        gaussian_noise = np.random.normal(mean, var, img.shape)
        gaussian_noise.round()
        new_image = img + gaussian_noise
        new_image = np.clip(new_image, 0, 255)
        return new_image

    def salt_pepper_noise(img, density=None):

        # Check if the image is RGB and convert it to gray scale
        if len(img.shape) == 3:
            gray_img = np.mean(img, axis=2)
        else:
            gray_img = img

        rows, cols = gray_img.shape

        if density == None:
            density = np.random.uniform(0, 1)
        pixels_number = rows * cols
        noise_pixels = int(density * pixels_number)

        salt_noise = np.random.randint(0, noise_pixels)
        pepper_noise = noise_pixels - salt_noise

        for i in range(salt_noise):
            row = np.random.randint(0, rows - 1)
            col = np.random.randint(0, cols - 1)
            gray_img[row][col] = 255
        for i in range(pepper_noise):
            row = np.random.randint(0, rows - 1)
            col = np.random.randint(0, cols - 1)
            gray_img[row][col] = 0

        return gray_img

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # Connect the load image button to the function
        self.pushButton_Normalize_load_2.clicked.connect(self.load_image_for_normalization)
        self.pushButton_filters_load.clicked.connect(self.load_image_for_filtering)
        # Connect the normalize button to the function
        self.pushButton_Normalize_2.clicked.connect(self.normalize_image_and_display)
        self.pushButton_histograms_load_2.clicked.connect(self.load_image_for_histogram)
        self.comboBox.currentIndexChanged.connect(self.update_parameters)
        self.pushButton_Normalize_load_3.clicked.connect(self.load_image_for_input_Noise)
        self.comboBox_2.currentIndexChanged.connect(self.apply_LP_filters)
        self.comboBox_3.currentIndexChanged.connect(self.apply_edge_detection)

        self.filter_parameters = {"Gaussian": {"KernelSize": 3, "Std": 1},
                                  "Uniform": {"KernelSize": 3},
                                  "Salt": {},
                                  "Pepper-Noise": {},
                                  "LP-Filter": {"KernelSize": 3, "Radius": 40},
                                  "HP-Filter": {"KernelSize": 3, "Radius": 40}
                                  }

    def update_parameters(self):
        # Get the selected filter from the combobox
        selected_filter = self.comboBox.currentText()

        # Check if the selected filter is in the filter parameters dictionary
        if selected_filter in self.filter_parameters:
            # Apply the selected filter with its parameters to the image
            parameters = self.filter_parameters[selected_filter]
            self.apply_filter(selected_filter, parameters)
        else:
            print("Filter not found in filter parameters dictionary!")

    def load_image_for_filtering(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)

        if file_name:
            self.load_image(file_name, self.label_filters_input)

    def load_image(self, file_path, target_label):
        image = cv.imread(file_path)
        # this new var is for when applying low pass filter
        global array_image
        array_image = image
        # print("The Array is: ", array_image) #printing the array
        image_qt = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(image_qt)
        target_label.setPixmap(pixmap.scaled(target_label.size()))

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

    def load_image_for_histogram(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            self.load_image(file_name, self.label_histograms_input_2)
            # Load the image using OpenCV
            image = cv.imread(file_name)
            # print(image)
            self.show_histogram(image, self.label_histograms_hinput_2)

    def normalize_image_and_display(self):
        # Check if an image has been loaded
        if not hasattr(self, 'loaded_image'):
            QMessageBox.warning(self, "Warning", "Please load an image before normalizing.")
            return

        # Perform normalization
        try:
            normalized_image = self.normalize_image(self.loaded_image)
        except Exception as e:  # Catch any unexpected errors during normalization
            QMessageBox.critical(self, "Error", f"An error occurred during normalization: {str(e)}")
            return

        if normalized_image is not None:
            # Convert normalized image to QPixmap for display
            normalized_qimage = QImage(normalized_image.data, normalized_image.shape[1], normalized_image.shape[0],
                                       normalized_image.strides[0], QImage.Format_RGB888)
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

    def apply_filter(self, filter_name, parameters):

        if filter_name == "LP-Filter":

            # Convert the input image to grayscale
            gray_image = cv.cvtColor(array_image, cv.COLOR_BGR2GRAY)

            # Perform FFT
            fft_image = np.fft.fft2(gray_image)
            fft_shifted = np.fft.fftshift(fft_image)

            # Create a circular low pass filter
            rows, cols = gray_image.shape
            center_row, center_col = rows // 2, cols // 2
            x, y = np.ogrid[:rows, :cols]
            radius = parameters.get("Radius", 50)
            mask = np.zeros((rows, cols), dtype=bool)
            mask[(x - center_row) ** 2 + (y - center_col) ** 2 <= radius ** 2] = True

            # Apply the filter
            fft_shifted_filtered = fft_shifted * mask

            # Perform inverse FFT
            ifft_shifted_filtered = np.fft.ifftshift(fft_shifted_filtered)
            ifft_filtered = np.fft.ifft2(ifft_shifted_filtered)

            # Convert back to uint8
            filtered_image = np.abs(ifft_filtered).astype(np.uint8)

            # Convert grayscale image to RGB format for display
            filtered_image_rgb = cv.cvtColor(filtered_image, cv.COLOR_GRAY2RGB)

            # Convert image to QImage
            h, w, c = filtered_image_rgb.shape
            qimage = QImage(filtered_image_rgb.data, w, h, w * c, QImage.Format_RGB888)

            # Convert QImage to QPixmap and display on the output label
            pixmap = QPixmap.fromImage(qimage)
            self.label_filters_output.setPixmap(pixmap.scaled(self.label_filters_output.size()))

        elif filter_name == "HP-Filter":
            # Convert the input image to grayscale
            gray_image = cv.cvtColor(array_image, cv.COLOR_BGR2GRAY)

            # Perform FFT
            fft_image = np.fft.fft2(gray_image)
            fft_shifted = np.fft.fftshift(fft_image)

            # Create a circular low pass filter
            rows, cols = gray_image.shape
            center_row, center_col = rows // 2, cols // 2
            x, y = np.ogrid[:rows, :cols]
            radius = parameters.get("Radius", 50)
            mask = np.ones((rows, cols), dtype=bool)  # opposite to LP-filter we initialize all elements to true
            mask[(x - center_row) ** 2 + (y - center_col) ** 2 <= radius ** 2] = False

            # Apply the filter
            fft_shifted_filtered = fft_shifted * mask

            # Perform inverse FFT
            ifft_shifted_filtered = np.fft.ifftshift(fft_shifted_filtered)
            ifft_filtered = np.fft.ifft2(ifft_shifted_filtered)

            # Convert back to uint8
            filtered_image = np.abs(ifft_filtered).astype(np.uint8)

            # Convert grayscale image to RGB format for display
            filtered_image_rgb = cv.cvtColor(filtered_image, cv.COLOR_GRAY2RGB)

            # Convert image to QImage
            h, w, c = filtered_image_rgb.shape
            qimage = QImage(filtered_image_rgb.data, w, h, w * c, QImage.Format_RGB888)

            # Convert QImage to QPixmap and display on the output label
            pixmap = QPixmap.fromImage(qimage)
            self.label_filters_output.setPixmap(pixmap.scaled(self.label_filters_output.size()))



        else:
            pass


    #  here is the method for 3 color channels
    # def show_histogram(self, image, label):
    #     # Calculate histogram for each color channel
    #     hist_red = cv.calcHist([image], [0], None, [256], [0, 256])
    #     hist_green = cv.calcHist([image], [1], None, [256], [0, 256])
    #     hist_blue = cv.calcHist([image], [2], None, [256], [0, 256])

    #     # Plot histograms using Matplotlib
    #     plt.figure()
    #     plt.plot(hist_red, color='red')
    #     plt.plot(hist_green, color='green')
    #     plt.plot(hist_blue, color='blue')
    #     plt.xlabel('Intensity')
    #     plt.ylabel('Frequency')
    #     plt.title('Histogram')
    #     plt.grid(True)

    #     # Convert the plot to a QImage
    #     plt.savefig('histogram.png')
    #     hist_image = Image.open('histogram.png')
    #     hist_image = hist_image.convert('RGB')
    #     hist_qimage = QImage(hist_image.tobytes(), hist_image.width, hist_image.height, QImage.Format_RGB888)

    #     # Convert QImage to QPixmap and display on the output label
    #     pixmap = QPixmap.fromImage(hist_qimage)
    #     label.setPixmap(pixmap.scaled(label.size()))

    def show_histogram(self, image, label):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        rows, cols = image.shape
        hist = np.zeros(256)
        for row in range(rows):
            for col in range(cols):
                intensity = int(image[row][col])
                hist[intensity] += 1
        plt.figure(figsize=(15, 7))
        plt.bar(range(256), hist, color='blue')
        plt.xticks(np.arange(0, 256, 10))
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.grid(True)
        plt.savefig('histogram.png')

        try:
            # Open the saved histogram image
            hist_image = Image.open('histogram.png')
            hist_image = hist_image.convert('L')  # Convert to grayscale
            hist_qimage = QImage(hist_image.tobytes(), hist_image.width, hist_image.height, QImage.Format_Grayscale8)

            # Convert QImage to QPixmap and display on the output label
            pixmap = QPixmap.fromImage(hist_qimage)
            label.setPixmap(pixmap.scaled(label.size()))

        except Exception as e:
            print("Error loading histogram image:", e)




    def load_image_for_input_Noise(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            # Load the original image
            original_image = cv2.imread(file_name)
            # Display the original image in label_Normalize_input_3
            self.display_image(original_image, self.label_Normalize_input_3)
            # Add noise to the image
            noisy_image = self.add_noise(original_image)
            # Display the noisy image in label_Normalize_output_4
            self.display_image(noisy_image, self.label_Normalize_output_4)
            # Store the noisy image for filtering
            self.noisy_image = noisy_image
            # Store the original image for edge detection
            self.original_image = original_image

    def apply_LP_filters(self):
        selected_filter = self.comboBox_2.currentText()
        if hasattr(self, 'noisy_image'):
            filtered_image = self.filter_image(self.noisy_image, selected_filter)
            if filtered_image is not None:
                self.display_image(filtered_image, self.label_Normalize_output_3)
            else:
                print("Error: Invalid filter type selected")

    def apply_edge_detection(self):
        selected_edge_detection = self.comboBox_3.currentText()
        if hasattr(self, 'original_image'):
            edge_detected_image = self.detect_edges(self.original_image, selected_edge_detection)
            if edge_detected_image is not None:
                self.display_image(edge_detected_image, self.label_Normalize_output_6)
            else:
                print("Error: Invalid edge detection type selected")

    def detect_edges(self, image, edge_detection_type):
        if edge_detection_type == "Sobel":
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_edges = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            return sobel_edges.astype('uint8')
        elif edge_detection_type == "Roberts":
            roberts_cross_v = np.array([[1, 0], [0, -1]])
            roberts_cross_h = np.array([[0, 1], [-1, 0]])
            roberts_x = cv2.filter2D(image, -1, roberts_cross_v)
            roberts_y = cv2.filter2D(image, -1, roberts_cross_h)
            roberts_edges = np.abs(roberts_x) + np.abs(roberts_y)
            return roberts_edges.astype('uint8')
        elif edge_detection_type == "Prewitt":
            prewitt_x = cv2.filter2D(image, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
            prewitt_y = cv2.filter2D(image, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
            prewitt_edges = np.abs(prewitt_x) + np.abs(prewitt_y)
            return prewitt_edges.astype('uint8')
        elif edge_detection_type == "Canny":
            canny_edges = cv2.Canny(image, 100, 200)
            return canny_edges
        else:
            print("Error: Invalid edge detection type selected")
            return None

    def filter_image(self, image, filter_type):
        if filter_type == "Gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif filter_type == "Average":
            return cv2.blur(image, (5, 5))
        elif filter_type == "Median":
            return cv2.medianBlur(image, 5)
        else:
            print("Error: Invalid filter type selected")
            return None

    def add_noise(self, image):
        mean = 0
        stddev = 50  # Increase the standard deviation for more noise
        h, w, c = image.shape
        noise = np.random.normal(mean, stddev, (h, w, c)).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return noisy_image

    def display_image(self, image, label):
        if len(image.shape) == 3:  # RGB image
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        else:  # Binary image
            height, width = image.shape
            qImg = QImage(image.data, width, height, width, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # Maintain aspect ratio of the pixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())