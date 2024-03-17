import sys

import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QComboBox,QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui, QtCore
from PyQt5.uic import loadUi
import numpy as np
import cv2 as cv
from Filtering import Ui_MainWindow
from PIL import Image
from matplotlib import pyplot as plt
import pyqtgraph as pg
# Noise functions
def uniform_noise(img, var=100):
    uniform_noise = np.random.randint(0, var, img.shape)
    new_image = np.clip(img + uniform_noise, 0, 255).astype(np.uint8)
    save_noisy_image(new_image)
    return new_image

def gaussian_noise(img, var=50):
    mean = 0
    gaussian_noise = np.random.normal(mean, var, img.shape)
    gaussian_noise.round()
    new_image = np.clip(img + gaussian_noise, 0, 255).astype(np.uint8)
    save_noisy_image(new_image)
    return new_image

def salt_pepper_noise(img, density=None):
    if len(img.shape) == 3:
        gray_img = np.mean(img, axis=2)
    else:
        gray_img = img

    rows, cols = gray_img.shape

    if density is None:
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

    noisy_img = gray_img.astype(np.uint8)
    save_noisy_image(noisy_img)
    return noisy_img

def save_noisy_image(img):
    if len(img.shape) == 3:
        plt.imsave("noisy.jpg", img)
    else:
        plt.imsave("noisy.jpg", img, cmap='gray')
        
class MainWindow(QMainWindow, Ui_MainWindow):
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
        self.pushButton_Normalize_load_3.clicked.connect(self.apply_noise)
        self.comboBox_2.currentIndexChanged.connect(self.apply_LP_filters)
        self.comboBox_3.currentIndexChanged.connect(self.apply_edge_detection)
        self.comboBox_4.currentIndexChanged.connect(self.noise_combobox)
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
            self.show_histogram(image, self.red_histogram,self.green_histogram,self.blue_histogram,self.red_cdf,self.green_cdf,self.blue_cdf)

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
    def plot_histogram(self, hist, title, color):
        
        hist_values = hist.flatten().tolist()  # Flatten and convert histogram values to list
        bins = np.arange(len(hist_values)).tolist() # Generate bins based on the length of hist_values
        
        plt = pg.PlotWidget(title=title)
        plt.plot(bins, hist_values, fillLevel=0, brush=color, stepMode=False)
        # plt.setLabel('left', 'Frequency')
        # plt.setLabel('bottom', 'Intensity')
        plt.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, 
            QtWidgets.QSizePolicy.Expanding
        )
        return plt

    def show_histogram(self, image, label_red, label_green, label_blue, label_red_cdf, label_green_cdf, label_blue_cdf):
        # Calculate histogram for each color channel
        hist_red = cv.calcHist([image], [0], None, [256], [0, 256])
        hist_green = cv.calcHist([image], [1], None, [256], [0, 256])
        hist_blue = cv.calcHist([image], [2], None, [256], [0, 256])

        # Clear labels
        label_red.clear()
        label_green.clear()
        label_blue.clear()
        label_red_cdf.clear()
        label_green_cdf.clear()
        label_blue_cdf.clear()

        # Plot histograms
        plot_red = self.plot_histogram(hist_red, 'Red Histogram', (255, 0, 0, 150))
        plot_green = self.plot_histogram(hist_green, 'Green Histogram', (0, 255, 0, 150))
        plot_blue = self.plot_histogram(hist_blue, 'Blue Histogram', (0, 0, 255, 150))

        # Plot cumulative histograms
        cumulative_hist_red = np.cumsum(hist_red.flatten())
        cumulative_hist_green = np.cumsum(hist_green.flatten())
        cumulative_hist_blue = np.cumsum(hist_blue.flatten())
        plot_red_cdf = self.plot_histogram(cumulative_hist_red, 'Red Channel CDF', (255, 0, 0, 150))
        plot_green_cdf = self.plot_histogram(cumulative_hist_green, 'Green Channel CDF', (0, 255, 0, 150))
        plot_blue_cdf = self.plot_histogram(cumulative_hist_blue, 'Blue Channel CDF', (0, 0, 255, 150))

        # Add plots to labels
        layout_red = QtWidgets.QVBoxLayout()
        layout_red.addWidget(plot_red)
        label_red.setLayout(layout_red)

        layout_green = QtWidgets.QVBoxLayout()
        layout_green.addWidget(plot_green)
        label_green.setLayout(layout_green)

        layout_blue = QtWidgets.QVBoxLayout()
        layout_blue.addWidget(plot_blue)
        label_blue.setLayout(layout_blue)

        layout_red_cdf = QtWidgets.QVBoxLayout()
        layout_red_cdf.addWidget(plot_red_cdf)
        label_red_cdf.setLayout(layout_red_cdf)

        layout_green_cdf = QtWidgets.QVBoxLayout()
        layout_green_cdf.addWidget(plot_green_cdf)
        label_green_cdf.setLayout(layout_green_cdf)

        layout_blue_cdf = QtWidgets.QVBoxLayout()
        layout_blue_cdf.addWidget(plot_blue_cdf)
        label_blue_cdf.setLayout(layout_blue_cdf)

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

    def noise_combobox(self):
        selected_noise= self.comboBox_4.currentText()
        if hasattr(self, 'to_noise_img'):
            img = self.to_noise_img
            if selected_noise == "Gaussian":
                noisy_image=gaussian_noise(img)
            elif selected_noise == "Uniform":
                noisy_image= uniform_noise(img)
            elif selected_noise == "Salt- Pepper":
                noisy_image=salt_pepper_noise(img)
            if noisy_image is not None:
                noisy_image_display=cv2.imread("noisy.jpg")    
                self.display_image(noisy_image_display, self.label_Normalize_output_4)
                self.noisy_image = noisy_image_display

    def apply_noise(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            # Load the original image
            original_image = cv2.imread(file_name)

            img_test = np.array(Image.open(file_name))
            # Display the original image in label_Normalize_input_3
            self.display_image(original_image, self.label_Normalize_input_3)
            selected_noise= self.comboBox_4.currentText()
            if selected_noise == "Gaussian":
                noisy_image=gaussian_noise(img_test)
            elif selected_noise == "Uniform":
                noisy_image= uniform_noise(img_test)
            elif selected_noise == "Salt- Pepper":
                noisy_image=salt_pepper_noise(img_test)
            if noisy_image is not None:
                noisy_image_test=cv2.imread("noisy.jpg")    
                self.display_image(noisy_image_test, self.label_Normalize_output_4)
                self.noisy_image = noisy_image_test
                self.original_image = original_image
                self.to_noise_img = img_test

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