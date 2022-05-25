import cv2 as cv 
import numpy as np 



def ProcessImage(SourceImage):
    # convert to gray
    GrayScaleImage = convertToGray(SourceImage)

    # increase contrast
    ContrastImage = increaseContrast(GrayScaleImage)

    height, width = ContrastImage.shape

    #Smoth
    imgBlur =  np.zeros((height, width, 1), np.uint8)
    imgBlur = cv.GaussianBlur(ContrastImage,(5,5),0)

    #Thresholding
    imgThresh = cv.adaptiveThreshold(imgBlur,255.0,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,25,9)
    cv.imshow('thresh',imgThresh)
    return GrayScaleImage,imgThresh

def convertToGray(SourceImage):
    height, width, numChannels = SourceImage.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv.cvtColor(SourceImage, cv.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv.split(imgHSV)
    
    #Hue, Saturation, Value
    return imgValue


def increaseContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv.morphologyEx(imgGrayscale, cv.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    imgBlackHat = cv.morphologyEx(imgGrayscale, cv.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #plus top hat
    imgGrayscalePlusTopHat = cv.add(imgGrayscale, imgTopHat) 
    #minus black hat
    Result = cv.subtract(imgGrayscalePlusTopHat, imgBlackHat)


    return Result


