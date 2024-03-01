import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtGui import QPainter, QColor, QPen,QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget,QMainWindow,QPushButton,QFrame
from PyQt5.QtCore import Qt
from functools import partial
from mainwindow_edit import Ui_MainWindow
from scipy import signal
import numpy as np
from PIL import Image
from scipy import fftpack
import os
import matplotlib.pyplot as plt
import myCanny
import processing
import NewHoughLine
import NewHoughCircle
import imageio
import newSnake
import math


imgForFilteration = None
imgForCircle = None
imgForLines = None
imgForHistogram = None
imgForCorners = None
pixmap = None
lb = None
flag = False
myImage = None
snakeContour = None
init = None
SnakeFlag = False
Cx =  0
Cy =  0
D  =  0

class ImageSnake (QtWidgets.QLabel):
    def __init__(self,parent=None):
        super(ImageSnake,self).__init__(parent=parent)
        self.initUI()
        self.setMouseTracking(True)
    
    def initUI(self):
        global pixmap
        #-----------------------------------------------------------
        self.setGeometry(QtCore.QRect(240, 20, 631, 371))
        self.raise_()
        self.setStyleSheet("background-color: white; inset grey; min-height: 200px;")
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(3)
        self.Drawing = False
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
            
    def mousePressEvent (self, event):
        global Cx,Cy,D
        self.Drawing = True
        self.x1 = event.x()
        self.y1 = event.y()     
    def mouseMoveEvent(self, event):  
        global Cx,Cy,D 
        if self.Drawing:
            D  = round(math.sqrt((event.x() - self.x1)**2 + (event.y() - self.y1)**2))  
            Cx = round((event.x() + self.x1) / 2)
            Cy = round((event.y() + self.y1) / 2)
            self.update()

    def mouseReleaseEvent (self,event):
        global Cx,Cy,D
        self.Drawing=False
        D  = round(math.sqrt((event.x() - self.x1)**2 + (event.y() - self.y1)**2))  
        Cx = round((event.x() + self.x1) / 2)
        Cy = round((event.y() + self.y1) / 2)
        #self.pos = event.pos()
        self.update()
        #print ("release")
        
    def paintEvent(self, event):
        global pixmap,flag,SnakeFlag,Cx,Cy,D
        super().paintEvent(event)
           
        if self.Drawing == True:
            q = QPainter(self)
            q.setPen(QPen(Qt.blue,8,Qt.SolidLine))
            if flag == True: q.drawPixmap(1,1,pixmap)   
            s = np.linspace(0, 2 * np.pi, 400)
            x = Cx +  (D//2) * np.cos(s)
            y = Cy +  (D//2) * np.sin(s)
            init = np.array([x, y]).T
            for i in range(len(init)):
                q.drawPoint(init[i, 0], init[i, 1])
            
            if SnakeFlag == True:
                q.setPen(QPen(Qt.red,4,Qt.SolidLine))
                
                for i in range(len(snakeContour)):
                    q.drawPoint(snakeContour[i, 0], snakeContour[i, 1])

        else:
            q = QPainter(self)
            q.setPen(QPen(Qt.blue,8,Qt.SolidLine))
            if flag == True: q.drawPixmap(1,1,pixmap) 
            s = np.linspace(0, 2 * np.pi, 400)
            x = Cx +  (D//2) * np.cos(s)
            y = Cy +  (D//2) * np.sin(s)
            init = np.array([x, y]).T
            for i in range(len(init)):
                q.drawPoint(init[i, 0], init[i, 1])
            
            if SnakeFlag == True:
                q.setPen(QPen(Qt.red,4,Qt.SolidLine))
                for i in range(len(snakeContour)):
                    q.drawPoint(snakeContour[i, 0], snakeContour[i, 1])

        
    def alert(self,snakeContour):
     self.update()




class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.lb = ImageSnake(self.tab_SIFT)
        self.pushButton_filters_load.clicked.connect(partial(self.loadImage, self.label_filters_input, self.label, self.label_2, 1))
        self.comboBox.currentIndexChanged.connect(self.filteration)
        self.pushButton_lines_load.clicked.connect(partial(self.loadImage,self.label_lines_input,self.label_4,self.label_5,2))
        self.pushButton_circles_load.clicked.connect(partial(self.loadImage,self.label_circles_input,self.label_6,self.label_7,3))
        self.pushButton_histograms_load.clicked.connect(partial(self.loadImage,self.label_histograms_input,self.label_11,self.label_10,4))
        self.pushButton_histograms_load_target.clicked.connect(self.histogramMatching)
        self.pushButton_corners_load.clicked.connect(partial(self.loadImage,self.label_corners_input,self.label_8,self.label_9,5))
        self.SnakeLoad.clicked.connect(self.Browse)
        self.StartSnake.clicked.connect(self.snake)

        self.radioButton.clicked.connect(self.histogramEqualization)
        self.radioButton_2.clicked.connect(self.EnableButton)
    
    def loadImage(self,inputLabel,nameLabel,sizeLabel,flag):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', 'Image files (*.jpeg *.jpg *.png)', options=QtWidgets.QFileDialog.DontUseNativeDialog)
        image = QtGui.QPixmap(filename)
        myImage = Image.open(filename)
        inputLabel.setPixmap(image)
        inputLabel.setScaledContents(True)
        sizeLabel.setText("Size: {} x {}".format(image.width(), image.height()))
        nameLabel.setText("Name:  {}".format(QtCore.QUrl.fromLocalFile(filename).fileName()))
        global imgForFilteration,imgForLines,imgForCircle,imgForHistogram,imgForCorners
        if (flag == 1):
            imgForFilteration = myImage
            self.comboBox.setDisabled(False)
            self.GroupBox2Cofig(True, True, True,"","")
            self.label_filters_output.hide()
            self.comboBox.setCurrentIndex(0)
        elif(flag == 2):
            imgForLines     = myImage
            self.HoughLine (np.array(imgForLines))
        elif(flag == 3):
            imgForCircle    = myImage
            self.HoughCircle (np.array(imgForCircle))
        

        elif(flag == 4):
            imgForHistogram = self.rgb2gray(np.array(myImage)/255.0)
            imgForHistogram = (imgForHistogram-np.min(imgForHistogram))/(np.max(imgForHistogram)-np.min(imgForHistogram))
            imgForHistogram = (imgForHistogram * 255).astype(np.uint8)
            self.showImg(imgForHistogram,QtGui.QImage.Format_Grayscale8,inputLabel)
            self.radioButton.setDisabled(False)
            self.radioButton_2.setDisabled(False)
            self.pushButton_histograms_load_target.setDisabled(True)
            self.graph2Img(imgForHistogram,self.label_histograms_hinput)
        elif (flag == 5):
            imgForCorners = myImage
            self.cornerDetection()
    def EnableButton(self):
        self.pushButton_histograms_load_target.setDisabled(False)
    def filteration(self):
        global imgForFilteration
        if self.comboBox.currentIndex()   == 1:
            self.label_filters_output.hide()
            self.GroupBox2Cofig(False, True, False,"size not w*w","")
            self.filterBtn.clicked.connect(self.smoothingFilter)

        elif self.comboBox.currentIndex() == 2:
            self.GroupBox2Cofig(True, True, True,"","")
            self.medianFilter()

        elif self.comboBox.currentIndex() == 3:
            self.GroupBox2Cofig(True, True, True,"","")
            self.sharpeningFilter()

        elif self.comboBox.currentIndex() == 4:
            self.label_filters_output.hide()
            self.GroupBox2Cofig(False,False, False,"size","sigma")
            self.filterBtn.clicked.connect(self.gaussianFilter)

        elif self.comboBox.currentIndex() == 5:
            self.GroupBox2Cofig(True, True, True,"","")
            self.frequencyDomain()
        elif self.comboBox.currentIndex() == 6:
            self.label_filters_output.hide()
            self.GroupBox2Cofig(False, False, False,"w","h")
            self.filterBtn.clicked.connect(self.lowPassFilter)
        elif self.comboBox.currentIndex() == 7:
            self.label_filters_output.hide()
            self.GroupBox2Cofig(False, False, False,"w","h")
            self.filterBtn.clicked.connect(self.highPassFilter)
        elif self.comboBox.currentIndex() == 8:
            self.GroupBox2Cofig(True, True, True,"","")
            self.prewitt()
        elif self.comboBox.currentIndex() == 9:
            self.GroupBox2Cofig(True, True, True,"","")
            self.sobel()
        elif self.comboBox.currentIndex() == 10:
            self.GroupBox2Cofig(True, True, True,"","")
            self.lplacianFilter()
        elif self.comboBox.currentIndex() == 11:
            self.GroupBox2Cofig(False, False, False,"size","sigma")
            self.filterBtn.clicked.connect(self.LOG)
        elif self.comboBox.currentIndex() == 12:
            self.GroupBox2Cofig(False, False, False,"siz1,siz2","sgm1,sgm2")
            self.filterBtn.clicked.connect(self.DOG)
    def histogramMatching(self):
        global imgForHistogram
        originalImg = imgForHistogram/255.0
        originalImg = originalImg.ravel()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '', 'Image files (*.jpeg *.jpg *.png)', options=QtWidgets.QFileDialog.DontUseNativeDialog)
        targetImg = Image.open(filename)
        targetImg = self.rgb2gray(np.array(targetImg)/255.0)
        s_values, bin_idx, s_counts = np.unique(originalImg, return_inverse=True,return_counts=True)
        t_values, t_counts = np.unique(targetImg, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        matchImg = interp_t_values[bin_idx].reshape(imgForHistogram.shape)
        matchImg = (matchImg - np.min(matchImg)) / (np.max(matchImg) - np.min(matchImg))
        matchImg = (matchImg * 255).astype(np.uint8)
        self.showImg(matchImg, QtGui.QImage.Format_Grayscale8, self.label_histograms_output)
        self.graph2Img(matchImg, self.label_histograms_houtput)

    def histogramEqualization(self):
        self.pushButton_histograms_load_target.setDisabled(True)
        global imgForHistogram
        img_eq = imgForHistogram/255.0
        img_eq = img_eq.flatten()
        hist, bins = np.histogram(img_eq, 256, density=True)
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        img_eq = np.interp(img_eq, bins[:-1], cdf)
        img_eq = img_eq.reshape(imgForHistogram.shape)
        img_eq = (img_eq - np.min(img_eq)) / (np.max(img_eq) - np.min(img_eq))
        img_eq = (img_eq * 255).astype(np.uint8)
        self.showImg(img_eq, QtGui.QImage.Format_Grayscale8, self.label_histograms_output)
        self.graph2Img(img_eq, self.label_histograms_houtput)

    def GroupBox2Cofig(self,KernelBool,StdBool,filterbtnBool,KPH,SPH):
        self.KernelSize.setPlaceholderText(KPH)
        self.Std.setPlaceholderText(SPH)
        self.KernelSize.setDisabled(KernelBool)
        self.Std.setDisabled(StdBool)
        self.filterBtn.setDisabled(filterbtnBool)
        self.KernelSize.setText('')
        self.Std.setText('')


    def smoothingFilter(self):
        global imgForFilteration
        winSize = self.KernelSize.toPlainText()
        try:
            winSize = int(winSize)
            smoothingImage =np.array(imgForFilteration)
            smoothingImage =smoothingImage /255.0
            kernel = np.ones((winSize,winSize),dtype='float32')/(winSize*winSize)

            # convolve 2d the kernel with each channel
            r = signal.convolve2d(smoothingImage[...,0],kernel,'same')
            g = signal.convolve2d(smoothingImage[...,1],kernel,'same')
            b = signal.convolve2d(smoothingImage[...,2],kernel,'same')
            # stack the channels back into a 8-bit colour depth image and plot it
            smoothingImage = np.dstack([r, g, b])
            smoothingImage = (smoothingImage-np.min(smoothingImage))/(np.max(smoothingImage)-np.min(smoothingImage))
            smoothingImage = (smoothingImage * 255).astype(np.uint8)
            self.showImg(smoothingImage,QtGui.QImage.Format_RGB888,self.label_filters_output)
        except ValueError:
            print("")
    def medianFilter(self):
        global imgForFilteration
        img = imgForFilteration
        width, height = img.size
        members= [(0,0)]*25
        medianImg = Image.new("RGB", (width, height), "white")
        for i in range (2,width-2):
            for j in range(2,height-2):

                members[0] = img.getpixel((i - 2, j - 2))
                members[1] = img.getpixel((i - 2, j-1))
                members[2] = img.getpixel((i - 2, j ))
                members[3] = img.getpixel((i - 2, j + 1))
                members[4] = img.getpixel((i - 2, j + 2))

                members[5] = img.getpixel((i - 1, j-2))
                members[6] = img.getpixel((i - 1, j-1))
                members[7] = img.getpixel((i-1  , j ))
                members[8] = img.getpixel((i - 1, j + 1))
                members[9] = img.getpixel((i - 1, j+2))

                members[10] = img.getpixel((i , j - 2))
                members[11] = img.getpixel((i, j - 1))
                members[12] = img.getpixel((i, j))
                members[13] = img.getpixel((i, j + 1))
                members[14] = img.getpixel((i , j + 2))

                members[15] = img.getpixel((i + 1, j-2))
                members[16] = img.getpixel((i + 1, j -1))
                members[17] = img.getpixel((i+1, j ))
                members[18] = img.getpixel((i + 1, j + 1))
                members[19] = img.getpixel((i + 1, j+2))

                members[20] = img.getpixel((i + 2, j - 2))
                members[21] = img.getpixel((i+2, j  - 1))
                members[22] = img.getpixel((i + 2, j ))
                members[23] = img.getpixel((i + 2, j+1))
                members[24] = img.getpixel((i + 2, j + 2))

                members.sort()
                medianImg.putpixel((i-1, j-1), (members[12]))
        medianImg = np.array(medianImg)
        r = medianImg[:, :, 0]
        g = medianImg[:, :, 1]
        b = medianImg[:, :, 2]
        medianImg = np.dstack([r, g, b])
        medianImg = medianImg.astype(np.uint8)
        self.showImg(medianImg,QtGui.QImage.Format_RGB888,self.label_filters_output)
    def sharpeningFilter(self):
        global imgForFilteration
        sharpeningImage = np.array(imgForFilteration)/255.0
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpeningImage = self.rgb2gray(sharpeningImage)
        sharpeningImage = signal.convolve2d(sharpeningImage,kernel,'same')
        sharpeningImage = (sharpeningImage - np.min(sharpeningImage)) / (np.max(sharpeningImage) - np.min(sharpeningImage))
        sharpeningImage = (sharpeningImage * 255).astype(np.uint8)
        self.showImg(sharpeningImage, QtGui.QImage.Format_Grayscale8,self.label_filters_output)
    def gaussianFilter(self):
        global imgForFilteration
        gaussianImage = np.array(imgForFilteration)
        gaussianImage = gaussianImage / 255.0
        kernelen = self.KernelSize.toPlainText()
        std = self.Std.toPlainText()
        try:
            kernelen = int(kernelen)
            std = float(std)
            kernel = self.gaussian_kernel(kernelen,std)

            r = signal.convolve2d(gaussianImage[..., 0], kernel, 'same')
            g = signal.convolve2d(gaussianImage[..., 1], kernel, 'same')
            b = signal.convolve2d(gaussianImage[..., 2], kernel, 'same')

            gaussianImage = np.dstack([r, g, b])
            gaussianImage = (gaussianImage - np.min(gaussianImage)) / (np.max(gaussianImage) - np.min(gaussianImage))
            gaussianImage = (gaussianImage * 255).astype(np.uint8)
            self.showImg(gaussianImage, QtGui.QImage.Format_RGB888,self.label_filters_output)
        except ValueError:
            print("")
    def frequencyDomain(self):
        global imgForFilteration
        spDomain = imgForFilteration
        spDomain = np.array(spDomain)/255.0
        spDomain = self.rgb2gray(spDomain)

        f = fftpack.fft2(spDomain)
        fshift = fftpack.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
        magnitude_spectrum = (magnitude_spectrum * 255).astype(np.uint8)
        self.showImg(magnitude_spectrum, QtGui.QImage.Format_Grayscale8,self.label_filters_output)
    def lowPassFilter(self):
        global imgForFilteration
        width = self.KernelSize.toPlainText()
        height = self.Std.toPlainText()
        try:
            width = float(width)
            height = float(height)
            if width > 0.5 or height > 0.5:
                print("w and h must be < 0.5")
                return
            LPFImage = imgForFilteration
            LPFImage = np.array(LPFImage) / 255.0
            LPFImage = self.rgb2gray(LPFImage)
            f = fftpack.fft2(LPFImage)
            fshift = fftpack.fftshift(f)
            LPF = self.generateFilter(fshift, width, height, "LPF")
            LPFImage = np.abs(fftpack.ifft2(LPF * fshift))
            LPFImage = (LPFImage - np.min(LPFImage)) / (np.max(LPFImage) - np.min(LPFImage))
            LPFImage = (LPFImage * 255).astype(np.uint8)
            self.showImg(LPFImage, QtGui.QImage.Format_Grayscale8,self.label_filters_output)

        except ValueError:
            print("")
    def highPassFilter(self):
        global imgForFilteration
        width = self.KernelSize.toPlainText()
        height = self.Std.toPlainText()
        try:
            width = float(width)
            height = float(height)
            if width > 0.5 or height > 0.5:
                print("w and h must be < 0.5")
                return
            HPFImage = imgForFilteration
            HPFImage = np.array(HPFImage) / 255.0
            HPFImage = self.rgb2gray(HPFImage)
            f = fftpack.fft2(HPFImage)
            fshift = fftpack.fftshift(f)
            HPF = self.generateFilter(fshift, width, height, "HPF")
            HPFImage = np.abs(fftpack.ifft2(HPF * fshift))
            HPFImage = (HPFImage - np.min(HPFImage)) / (np.max(HPFImage) - np.min(HPFImage))
            HPFImage = (HPFImage * 255).astype(np.uint8)
            self.showImg(HPFImage, QtGui.QImage.Format_Grayscale8,self.label_filters_output)

        except ValueError:
            print("")
    def prewitt(self):
        global imgForFilteration
        img = np.array(imgForFilteration)/255.0
        kernel_h = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_v = kernel_h.transpose()
        img = self.rgb2gray(img)
        img_h = signal.convolve2d(img,kernel_h,'same')
        img_v = signal.convolve2d(img, kernel_v, 'same')
        prewittImg = np.sqrt(img_h*img_h + img_v*img_v)
        prewittImg = (prewittImg - np.min(prewittImg)) / (np.max(prewittImg) - np.min(prewittImg))
        prewittImg = (prewittImg * 255).astype(np.uint8)
        self.showImg(prewittImg, QtGui.QImage.Format_Grayscale8,self.label_filters_output)
    def sobel(self):
        global imgForFilteration
        img = np.array(imgForFilteration)/255.0
        kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_v = kernel_h.transpose()
        img = self.rgb2gray(img)
        img_h = signal.convolve2d(img,kernel_h,'same')
        img_v = signal.convolve2d(img, kernel_v, 'same')
        sobelImg = np.sqrt(img_h*img_h + img_v*img_v)
        sobelImg = (sobelImg - np.min(sobelImg)) / (np.max(sobelImg) - np.min(sobelImg))
        sobelImg= (sobelImg * 255).astype(np.uint8)
        self.showImg(sobelImg, QtGui.QImage.Format_Grayscale8,self.label_filters_output)
    def lplacianFilter(self):
        global imgForFilteration
        img = np.array(imgForFilteration)/255.0
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        img = self.rgb2gray(img)
        laplacianImg = signal.convolve2d(img,kernel,'same')
        laplacianImg = (laplacianImg - np.min(laplacianImg)) / (np.max(laplacianImg) - np.min(laplacianImg))
        laplacianImg= (laplacianImg * 255).astype(np.uint8)
        self.showImg(laplacianImg, QtGui.QImage.Format_Grayscale8,self.label_filters_output)
    def LOG(self):
        global imgForFilteration
        img = np.array(imgForFilteration) / 255.0
        img = self.rgb2gray(img)
        laplacianKernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernelen = self.KernelSize.toPlainText()
        std = self.Std.toPlainText()
        try:
            kernelen = int(kernelen)
            std = float(std)
            gaussianKernel = self.gaussian_kernel(kernelen,std)
            gaussianImg = signal.convolve2d(img, gaussianKernel,'same')
            LOGImg = signal.convolve2d(gaussianImg,laplacianKernel,'same')

            LOGImg = (LOGImg - np.min(LOGImg)) / (np.max(LOGImg) - np.min(LOGImg))
            LOGImg = (LOGImg * 255).astype(np.uint8)
            self.showImg(LOGImg, QtGui.QImage.Format_Grayscale8,self.label_filters_output)
        except ValueError:
            print("")
    def DOG(self):
        global imgForFilteration
        img = np.array(imgForFilteration) / 255.0
        img = self.rgb2gray(img)
        kernelen = self.KernelSize.toPlainText()
        std = self.Std.toPlainText()

        kernelen =  [x.strip() for x in kernelen.split(',')]
        std = [x.strip() for x in std.split(',')]
        if len(kernelen)==1 or len(std) == 1:
            print ('Error')
        else:
            kernelen1 = int(kernelen[0])
            kernelen2 = int(kernelen[1])
            std1 = float(std[0])
            std2 = float(std[1])
            kernel1 = self.gaussian_kernel(kernelen1,std1)
            kernel2 = self.gaussian_kernel(kernelen2,std2)

            im1 = signal.convolve2d(img, kernel1, 'same')
            im2 = signal.convolve2d(img, kernel2, 'same')
            DOGImg = im1-im2
            DOGImg = (DOGImg - np.min(DOGImg)) / (np.max(DOGImg) - np.min(DOGImg))
            DOGImg = (DOGImg * 255).astype(np.uint8)
            #DOGImg = self.gray2binary(DOGImg)
            self.showImg(DOGImg, QtGui.QImage.Format_Grayscale8, self.label_filters_output)
    def cornerDetection(self):
        global imgForCorners
        color_img = np.array(imgForCorners)
        sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_v = sobel_h.transpose()
        img = np.array(imgForCorners)
        imggray = processing.rgb2gray(img)
        kernel = self.gaussian_kernel(7, 1.5)
        img1 =  signal.convolve2d(imggray, kernel, 'same')
        Ix = signal.convolve2d(img1, sobel_h, 'same')
        Iy = signal.convolve2d(img1, sobel_v, 'same')
        Ixx = Ix ** 2
        Ixy = Iy * Ix
        Iyy = Iy ** 2

        # offset is the step for each  winddow
        offset = 1
        height = imggray.shape[0]
        width = imggray.shape[1]

        for y in range(offset, height - offset):
            for x in range(offset, width - offset):
                Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
                Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
                Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

                # Find determinant and trace, use to get corner response
                det = (Sxx * Syy) - (Sxy ** 2)
                trace = Sxx + Syy
                k = 0.04
                r = det - k * (trace ** 2)

                if r > 2.5:
                    color_img[y, x] = [255, 0, 0, 255]

        #color_img = color_img.astype(np.int8)
        self.showImg(color_img, QtGui.QImage.Format_RGBA8888, self.label_corners_corners_output)
  
    def gaussian_kernel(self,kernlen, std):
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d
    def showImg(self,img,imgType,imglabel):
        imglabel.show()
        qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], imgType)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        imglabel.setPixmap(pixmap)
        imglabel.setScaledContents(True)
    def rgb2gray(self,rgb_image):
        return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
    def gray2binary(self,grayImg):
        for i in range(1,grayImg.shape[0]):
            for j in range(1,grayImg.shape[1]):
                if  (grayImg[i][j] >= 127):
                    grayImg[i][j] = 0
                else:
                    grayImg[i][j] = 255
        return grayImg
    def graph2Img(self,img,imglabel):
        imglabel.setScaledContents(True)
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.hist(img.ravel())
        plt.title('Histogram')
        fig.savefig('Histogram.png')
        image = QtGui.QPixmap('Histogram.png')
        imglabel.setPixmap(image)
        os.remove('Histogram.png')
    def generateFilter(self,image, w, h, filtType):
        m = np.size(image, 0)
        n = np.size(image, 1)
        LPF = np.zeros((m, n))
        HPF = np.ones((m, n))
        xi = np.round((0.5 - w / 2) * m)
        xf = np.round((0.5 + w / 2) * m)
        yi = np.round((0.5 - h / 2) * n)
        yf = np.round((0.5 + h / 2) * n)
        LPF[int(xi):int(xf), int(yi):int(yf)] = 1
        HPF[int(xi):int(xf), int(yi):int(yf)] = 0
        if filtType == "LPF":
            return LPF
        elif filtType == "HPF":
            return HPF
        else:
            print("Only Ideal LPF and HPF are supported")

    def HoughLine(self, image_rgb):
        
        if image_rgb.ndim == 3:
            image_gray = processing.rgb2gray(image_rgb)         #Converting rgb to gray
        else:
            image_gray = image_rgb
        edged = myCanny.findEdges(image_gray)
        input_image,H = NewHoughLine.hough_line(image_rgb,edged) 
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.label_lines_input_2.setPixmap(pixmap_image)
        self.label_lines_input_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_lines_input_2.setScaledContents(True)
        self.label_lines_input_2.setMinimumSize(1,1)
        self.label_lines_input_2.show()
        self.drawHoughSpace(self.label_lines_hough,H)

    def HoughCircle(self, image_rgb):
        
        if image_rgb.ndim == 3:
            image_gray = processing.rgb2gray(image_rgb)         #Converting rgb to gray
        else:
            image_gray = image_rgb
        edged = myCanny.findEdges(image_gray)
        input_image = NewHoughCircle.Hough_circle(image_rgb,edged) 
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        self.label_circles_output.setPixmap(pixmap_image)
        self.label_circles_output.setAlignment(QtCore.Qt.AlignCenter)
        self.label_circles_output.setScaledContents(True)
        self.label_circles_output.setMinimumSize(1,1)
        self.label_circles_output.show()

    def drawHoughSpace(self, label,H):
        plt.imshow(H)
        plt.savefig('acc.png')
        imgpath = 'acc.png'
        input_image = imageio.imread(imgpath)  
        height, width, channels = input_image.shape
        bytesPerLine = channels * width
        qImg = QtGui.QImage(input_image.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        pixmap01 = QtGui.QPixmap.fromImage(qImg)
        pixmap_image = QtGui.QPixmap(pixmap01)
        label.setPixmap(pixmap_image)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setScaledContents(True)
        label.setMinimumSize(1,1)
        label.show()
    
    def Browse(self):
        global pixmap,flag,myImage
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file','d:\\', "Image files (*.jpg *.png)")
        self.filePath = fname[0]
        pixmap = QPixmap(self.filePath)
        myImage = Image.open(self.filePath)
        flag = True

    def snake(self):
        global myImage,pixmap,init,snakeContour,SnakeFlag,Cx,Cy,D
        s = np.linspace(0, 2 * np.pi, 400)
        x = Cx +  (D//2) * np.cos(s)
        y = Cy +  (D//2) * np.sin(s)
        init = np.array([x, y]).T
        snakeContour = newSnake.kassSnake(np.array(myImage), init, wEdge=1.0, alpha=0.5, beta=10, gamma=0.001, maxIterations=500, maxPixelMove=1.0, convergence=0.1)
        SnakeFlag = True
        self.lb.alert(snakeContour)



    
       

if __name__ == "__main__":
   
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    window = Main()
    window.show()
    sys.exit(app.exec_())



