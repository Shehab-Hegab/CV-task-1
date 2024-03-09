import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui,QtCore
from PyQt5.uic import loadUi
import numpy as np
import cv2 as cv
from Filtering import Ui_MainWindow
from PIL import Image
import matplotlib.pyplot as plt

class MainWindow(QMainWindow,Ui_MainWindow):
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
        
        self.filter_parameters = {"Gaussian": {"KernelSize": 3, "Std": 1},
                                  "Uniform": {"KernelSize": 3},
                                  "Salt": {},
                                  "Pepper-Noise": {},
                                  "LP-Filter": {"KernelSize": 3,"Radius":40},
                                  "HP-Filter": {"KernelSize": 3,"Radius":40}
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
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        
        if file_name:
            self.load_image(file_name,self.label_filters_input)
        
    def load_image(self, file_path,target_label):
        image = cv.imread(file_path)
        # this new var is for when applying low pass filter
        global array_image
        array_image =image
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
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            self.load_image(file_name, self.label_histograms_input_2)
            # Load the image using OpenCV
            image = cv.imread(file_name)
            # print(image)
            self.show_histogram(image,self.label_histograms_hinput_2)


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

    def apply_filter(self,filter_name,parameters):

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
            
        elif  filter_name == "HP-Filter":
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
            mask = np.ones((rows, cols), dtype=bool) # opposite to LP-filter we initialize all elements to true
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

    # def show_histogram(self,image,label):
    #     imgForHistogram=self.rgb2gray(image/255.0)
    #     imgForHistogram = (imgForHistogram-np.min(imgForHistogram))/(np.max(imgForHistogram)-np.min(imgForHistogram))
    #     imgForHistogram = (imgForHistogram * 255).astype(np.uint8)
    #     qimg = QtGui.QImage(imgForHistogram.data, imgForHistogram.shape[1], imgForHistogram.shape[0],imgForHistogram.strides[0], QtGui.QImage.Format_Grayscale8)
    #     pixmap = QtGui.QPixmap.fromImage(qimg)
    #     label.setPixmap(pixmap.scaled(label.size()))    
        
    
    # def rgb2gray(self,rgb):
    #     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


    # def show_histogram(self, image, label):
    #     # Calculate histogram
    #     histogram = cv.calcHist([image], [0], None, [256], [0, 256])

    #     # Normalize histogram
    #     histogram /= np.sum(histogram)

    #     # Plot histogram using Matplotlib
    #     plt.figure()
    #     plt.plot(histogram)
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
        # Flatten the image array to a 1D array
        img_flat = image.ravel()

        # Calculate histogram using numpy
        histogram, bins = np.histogram(img_flat, bins=256, range=(0, 256), density=True)

        # Plot histogram using Matplotlib
        plt.figure()
        plt.plot(histogram, color='blue')  # Plot histogram in black (grayscale)
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.grid(True)

        # Convert the plot to a QImage
        plt.savefig('histogram.png')
        hist_image = Image.open('histogram.png')
        hist_image = hist_image.convert('RGB')
        hist_qimage = QImage(hist_image.tobytes(), hist_image.width, hist_image.height, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display on the output label
        pixmap = QPixmap.fromImage(hist_qimage)
        label.setPixmap(pixmap.scaled(label.size()))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
