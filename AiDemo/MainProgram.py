import numpy as np 
import cv2 as cv 
import preprocess
import math
import sys
# import pickle
# import tensorflow as tf
from keras.models import load_model
# import cv2

ALPHA_DICT = {'0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, 'A' : 10, 'B' : 11, 'C' : 12, 'D' : 13,
              'E': 14, 'F': 15, 'G': 16, 'H': 17, 'K': 18, 'L': 19, 'M': 20, 'N': 21, 'P': 22, 'R': 23, 'S' : 24, 'T' : 25, 'U' : 26,
              'V' : 27, 'X' : 28, 'Y' : 29, 'Z' : 30}
RESIZED_IMAGE_WIDTH = 12
RESIZED_IMAGE_HEIGHT = 28
imgSrc = cv.imread("./AiDemo/l.jpg")

# imgSrc = cv.resize(imgSrc,dsize = (1920,1080))

model = load_model('my_model.h5')  

gray, thresh = preprocess.ProcessImage(imgSrc)

#Canny 
canny_img = cv.Canny(thresh,250,255)
cv.imshow('canny', canny_img)
#Increase dilate

kernel = np.ones((3,3), np.uint8)
dilate_img = cv.dilate(canny_img, kernel ,iterations = 1)

#Draw Contour

contours, hierarchy = cv.findContours(dilate_img,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv.contourArea, reverse= True)[:10]





def findRectangle(contours, sourceImage):
    ScreenCnt = []
    for c in contours:
        SpaceAround = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c,0.06 * SpaceAround, True )
        [x, y, w, h] = cv.boundingRect(approx.copy())
        t = h/w
        if t > 1 or t < 0.5:
            continue 
        if len(approx) == 4:
            print(t)
            ScreenCnt.append(approx)
            # cv.putText(sourceImage,str(len(approx)), (x,y),cv.FONT_HERSHEY_DUPLEX,2,(0,255,0))
    return ScreenCnt if ScreenCnt is not None else None


RectContours = findRectangle(contours, imgSrc)



if RectContours is None:
    detected = 0
    print ("No plate detected")
else:
    detected = 1

n = 0
if detected == 1:
    new_image = 0
    for screenCnt in RectContours:

        cv.drawContours(imgSrc, [screenCnt], -1, (0, 255, 0), 3)
        (x1,y1) = screenCnt[0,0]
        (x2,y2) = screenCnt[1,0]
        (x3,y3) = screenCnt[2,0]
        (x4,y4) = screenCnt[3,0]
        array = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
        sorted_array = array.sort(reverse=True, key=lambda x:x[1])
        

        (x1,y1) = array[0]
        (x2,y2) = array[1]


        doi = abs(y1 - y2)
        ke = abs (x1 - x2)
        # print(str(doi) +' '+ str(ke))
        print('-----')


        angle = math.atan(doi/ke) * (180.0 / math.pi)


        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv.drawContours(mask, [screenCnt], 0, 255, -1, )
        #cv2.imshow("new_image",new_image) 
            # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = imgSrc[topx:bottomx, topy:bottomy]
        imgThresh = thresh[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx)/2, (bottomy - topy)/2

        if x1 < x2:
            rotationMatrix = cv.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv.getRotationMatrix2D(ptPlateCenter, angle, 1.0)



        roi = cv.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx ))
        imgThresh = cv.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx ))
        roi = cv.resize(roi,(0,0),fx = 3, fy = 3)
        imgThresh = cv.resize(imgThresh,(0,0),fx = 3, fy = 3)
        ####################################

        #################### Tiền xử lý ảnh đề phân đoạn kí tự ####################
        kerel3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
        thre_mor = cv.morphologyEx(imgThresh,cv.MORPH_DILATE,kerel3)
        cont,hier = cv.findContours(thre_mor,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 

        # cv.drawContours(roi, cont, -1, (100, 255, 255), 2) #Vẽ contour các kí tự trong biển số

        ##################### Lọc vùng kí tự #################
        char_x_ind = {}
        char_x = []
        chars = []
        height, width,_ = roi.shape
        roiarea = height * width

        for ind,cnt in enumerate(cont) :
            (x,y,w,h) = cv.boundingRect(cont[ind])
            ratiochar = w/h
            char_area = w*h

            if (0.01*roiarea < char_area < 0.09*roiarea) and ( 0.25 < ratiochar < 0.7):
                if x in char_x: #Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)    
                char_x_ind[x] = ind

        ############ Cắt và nhận diện kí tự ##########################

        char_x = sorted(char_x)
        img = []
        plate_num = []
        strFinalString = ""
        first_line = ""
        second_line = ""
        for i in char_x:
            (x,y,w,h) = cv.boundingRect(cont[char_x_ind[i]])
            cv.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)

            imgROI = thre_mor[y:y+h,x:x+w]     # cắt kí tự ra khỏi hình
            imgROIResized = cv.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize lại hình ảnh
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # đưa hình ảnh về mảng 1 chiều
            imgROIResized = np.expand_dims(imgROIResized, axis=-1)
            plate_num.append(imgROIResized)
        #     chars.append(imgROIResized)

        plate_num = np.array(plate_num, dtype = 'float32')
        if plate_num.size != 0:
            rls = model.predict(plate_num)
            rls=np.argmax(rls,axis=1)
            final_rls = [None]*rls.size
            for i in range(0, rls.size):
                for key, value in ALPHA_DICT.items():
                    if ALPHA_DICT[key] == rls[i]:
                        final_rls[i] = key
            for i in range(0, rls.size):
                if i%2 != 0:
                    first_line += final_rls[i]
                    if i == 3:
                        first_line += '-'
                else:
                    second_line += final_rls[i]
                    if i == 4:
                        second_line += '.'
        strFinalString = first_line + ' ' +second_line
        print(strFinalString)
        cv.putText(imgSrc, strFinalString ,(topy ,topx),cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        roi = cv.resize(roi, None, fx=0.6, fy=0.6)
        cv.imshow(str(n),cv.cvtColor(roi,cv.COLOR_BGR2RGB))
        n = n + 1

        
def checkContours(source, current, x, y ):
    for contour in source:
        if contour is not current:
            if cv.pointPolygonTest(contour, (x,y), False) >= 0.0:
                source.remove(contour)
                return True
    return False

imgSrc = cv.resize(imgSrc, None, fx=0.8, fy=0.8) 
cv.imshow('License plate', imgSrc)
cv.waitKey(0)