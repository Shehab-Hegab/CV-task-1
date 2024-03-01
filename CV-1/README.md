# The Problems Faced
1. Dealing with QPixmap, and converting  an image matrix to pixmap<br>
[Reference1](https://stackoverflow.com/questions/34232632/convert-python-opencv-image-numpy-array-to-pyqt-qpixmap-image)<br>
[Reference2](https://www.riverbankcomputing.com/static/Docs/PyQt4/qpixmap.html)

2. A problem related to a sharpening filter, There was a black layer shown after applying the filter<br>
<span style = "color:green">Solved :</span> by converting the image to grayscale(By the help of Eng.Asem)

3. A difficulty in implementing image meanshift segmentation, so we implemented a 2D meanshift by locating points.

4. A problem in drawing a circle in label and get the coordinates of the current position of the cursor<br>
<span style = "color:green">Solved :</span> by searching on each problem separately
[Reference1](https://stackoverflow.com/questions/43454882/paint-over-qlabel-with-pyqt) <br>
[Reference2](https://sites.google.com/site/rexstribeofimageprocessing/chan-vese-active-contours/wubiaotitiezi)

5. In Hough Transform (Line, Circle), if there are many shapes in picture makes process become very slow.

6. Draw (line or circle) in the picture after detection of them <br>
<span style = "color:green">Solved By:</span> Using [Image Draw Module - Pillow](https://pillow.readthedocs.io/en/3.1.x/reference/ImageDraw.html)

7. In Hough Transform Circle, circles after detection show shifted in image<br>
<span style = "color:green">Solved By:</span> Using another iteration method to compute accumulator (with help of Christeen Ramsis)



# Results

### The difference between Spatial Domain filters and frequency domain filter is that :
In Spatial Domain Filters, we make a convolution between an image and a kernel <br>
But in Frequency Domain Filters, we multiply an image with a kernel

## <u>OutPut Images</u>
## **Note!!** If the images are not shown clearly, there is a folder containing them  attached with the task file
* ## Smoothing Filter
    * 1x1 Window Size ![Image](Output_Images/1_1SmoothingFilter.png)
    * 3x3 Window Size ![Image](Output_Images/3_3SmoothingFilter.png)
    * 5x5 Window Size ![Image](Output_Images/5_5SmoothingFilter.png)
    * 7x7 Window Size ![Image](Output_Images/7_7SmoothingFilter.png)
* ## Median Filter
    * 5x5 Median Filter ![Image](Output_Images/5_5MedianFilter.png)
* ## Gaussian Filter
    * 7x7 Segma = 0.5 ![Image](Output_Images/Gauusian(7,0.5).png)
    * 7x7 Segma = 1 ![Image](Output_Images/Gauusian(7,1).png)
    * 7x7 Segma = 1.5 ![Image](Output_Images/Gauusian(7,1.5).png)
    * 7x7 Segma = 2 ![Image](Output_Images/Gauusian(7,2).png)
    * 7x7 Segma = 2.5 ![Image](Output_Images/Gauusian(7,2.5).png)
    * 7x7 Segma = 3 ![Image](Output_Images/Gauusian(7,3).png)
* ## Sharpening Filter
    *Image 1 ![Image](Output_Images/sharp.png)
    *Image 2 ![Image](Output_Images/sharp1.png)
* ## Frequency Domain Image
     ![Image](Output_Images/FD.png)
* ## Low Pass Filter
    * w = h = 0.125![Image](Output_Images/LowPassFilter(0.125).png)
    * w = h = 0.25 ![Image](Output_Images/LowPassFilter(0.25).png)
* ## High Pass Filter
    * w = h = 0.025![Image](Output_Images/HighPassFilter(0.025).png)
    * w = h = 0.125![Image](Output_Images/HighPassFilter(0.125).png)
* ## Edge Detectors
    * prewitt Filter![Image](Output_Images/prewitt.png)
    * Sobel Filter![Image](Output_Images/sobel.png)
    * Laplacian Filter![Image](Output_Images/Laplacian.png)
    * LOG Filter with gaussian(size=3,sigma=1.5)![Image](Output_Images/LOG.png)
    * DOG Filter with gaussian1(size=5,sigma=1),gaussian2(size=5,sigma=2)![Image](Output_Images/DOG.png)
* ## Histogram
    * Histogram Equalizatin![Image](Output_Images/histogramequalization.png)
    * Histogram Matching![Image](Output_Images/histogrammatching.png)
* # Hough  Transform
  * ## Line Detection ![Image](Output_Images/HoughLineTransform.png)
  
  * ## Circle Detection ![Image](Output_Images/HoughCircleTransform.png)

* ## Image Segmentation
    * Threshold Segmentation
    ![Image](Output_Images/index.jpeg)

    * Region Growing Segmentation
    ![Image](Output_Images/region.png)

    * Mean Shift Segmentation
    ![Image](Output_Images/atia.png)



* ## Snake
    ![Image](Output_Images/snake.png)

* ## Corner Detection
    ![Image](Output_Images/cornerdetection.png)

 

