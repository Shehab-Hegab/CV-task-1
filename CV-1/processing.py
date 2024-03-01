from scipy.signal import convolve2d
import numpy as np                 #Basic calculations library




def rgb2gray(image_rgb):
    r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
    image_gray = np.round(0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
    return image_gray


def blurImage(image_gray):
    kernel = np.ones((3,3),np.float32)/9                     #Blurring kernel
    res=convolve2d(image_gray,kernel,mode='same')
    return np.round(res).astype(np.uint8)
