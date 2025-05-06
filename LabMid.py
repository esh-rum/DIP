import cv2 as cv
import numpy as np
import os

train_img = []
for root, _, files in os.walk("E:\\6th Semester\\DIP\\Lab\\Open Lab with Instructions\\dataset\\images"):
        for file in files:
            if file.endswith(".png"):
                filePath = os.path.join(root, file)
                img1 = cv.imread(filePath, 0)
                #cv.imshow('Image ', img1)
                #cv.waitKey(0)
                train_img.append(img1)  
                
