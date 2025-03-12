import numpy as np
import cv2 as cv

#Task 1
""" img = cv.imread("E:/6th Semester/DIP/Lab/lab2.2.png", 0)
cv.imshow('Image', img)
cv.waitKey(0)

row, col = img.shape

totalVal = 0 
totalVal = np.int64(totalVal)
totalNum = 0
for i in range(row):
    for j in range(col):
        totalVal = totalVal + img[i][j]
        totalNum += 1
        
thresholdMean = totalVal/totalNum
binaryImg = img.copy()
negativeImg = img.copy()

for i in range(row):
    for j in range(col):
        if img[i][j] >= thresholdMean:
            binaryImg[i][j] = 255
        else:
            binaryImg[i][j] = 0

cv.imshow('Binary Image', binaryImg)
cv.waitKey(0)

for i in range(row):
    for j in range(col):
        s = (256 - 1) - img[i][j]
        negativeImg[i][j] = s 
        
cv.imshow('Negative Image', binaryImg)
cv.waitKey(0) """

# TASK 2
""" img = cv.imread("E:/6th Semester/DIP/Lab/lab2.2.png", 0)
cv.imshow('Gradient Image', img)
cv.waitKey(0)

otherImg = cv.imread("E:/6th Semester/DIP/Lab/lab1.png", 0)
cv.imshow('Other Image', otherImg)
cv.waitKey(0)

row1, col1 = img.shape
row2, col2 = otherImg.shape

imgMean = np.mean(img)
otherMean = np.mean(otherImg)

newImgA = newImgB = newImgC = np.zeros((row1, col1), dtype=np.uint8)
newOtherImgA = newOtherImgB = newOtherImgC = np.zeros((row2, col2), dtype=np.uint8)

for i in range(row2):
    for j in range(col2):
        if otherImg[i][j] <= otherMean:
            newOtherImgA[i][j] = 0
        else:
            newOtherImgA[i][j] = 255

cv.imshow('Other Image Part A', newOtherImgA)
cv.waitKey(0)

for i in range(row2):
    for j in range(col2):
        if otherImg[i][j] <= otherMean:
            newOtherImgB[i][j] = 255
        else:
            newOtherImgB[i][j] = 0

cv.imshow('Other Image Part B', newOtherImgB)
cv.waitKey(0)

a = otherMean + 20
b = otherMean - 20
for i in range(row2):
    for j in range(col2):
        if otherImg[i][j] >= b and otherImg[i][j] <= a:
            newOtherImgC[i][j] = 0
        else:
            newOtherImgC[i][j] = 255

cv.imshow('Other Image Part C', newOtherImgC)
cv.waitKey(0)


for i in range(row1):
    for j in range(col1):
        if img[i][j] <= imgMean:
            newImgA[i][j] = 0
        else:
            newImgA[i][j] = 255

cv.imshow('Image Part A', newImgA)
cv.waitKey(0)

for i in range(row1):
    for j in range(col1):
        if img[i][j] <= imgMean:
            newImgB[i][j] = 255
        else:
            newImgB[i][j] = 0

cv.imshow('Image Part B', newImgB)
cv.waitKey(0)

a = imgMean + 20
b = imgMean - 20
for i in range(row1):
    for j in range(col1):
        if img[i][j] >= b and img[i][j] <= a:
            newImgC[i][j] = 0
        else:
            newImgC[i][j] = 255

cv.imshow('Image Part C', newImgC)
cv.waitKey(0) """

# TASK 3
""" img1 = cv.imread("E:/6th Semester/DIP/Lab/lab3.1.png", 0)
cv.imshow('Image 1', img1)
cv.waitKey(0)

img2 = cv.imread("E:/6th Semester/DIP/Lab/lab3.2.png", 0)
cv.imshow('Image 2', img2)
cv.waitKey(0)

img3 = cv.imread("E:/6th Semester/DIP/Lab/lab3.3.png", 0)
cv.imshow('Image 3', img3)
cv.waitKey(0)

row1, col1 = img1.shape
row2, col2 = img2.shape
row3, col3 = img3.shape

newImg1 = np.zeros((row1, col1), dtype=np.uint8)
newImg2 = np.zeros((row2, col2), dtype=np.uint8)
newImg3 = np.zeros((row3, col3), dtype=np.uint8)

gamma = 0.2
for i in range(row1):
    for j in range(col1):
        s = 255 * pow((img1[i][j]/255), gamma)
        newImg1[i][j] = s
        
cv.imshow(f'Image 1 with gamma{gamma}', newImg1)
cv.waitKey(0)

for i in range(row2):
    for j in range(col2):
        s = 255 * pow((img2[i][j]/255), gamma)
        newImg2[i][j] = s
        
cv.imshow(f'Image 2 with gamma{gamma}', newImg2)
cv.waitKey(0)

for i in range(row3):
    for j in range(col3):
        s = 255 * pow((img3[i][j]/255), gamma)
        newImg3[i][j] = s
        
cv.imshow(f'Image 3 with gamma{gamma}', newImg3)
cv.waitKey(0)  


gamma = 0.5
for i in range(row1):
    for j in range(col1):
        s = 255 * pow((img1[i][j]/255), gamma)
        newImg1[i][j] = s
        
cv.imshow(f'Image 1 with gamma{gamma}', newImg1)
cv.waitKey(0)

for i in range(row2):
    for j in range(col2):
        s = 255 * pow((img2[i][j]/255), gamma)
        newImg2[i][j] = s
        
cv.imshow(f'Image 2 with gamma{gamma}', newImg2)
cv.waitKey(0)

for i in range(row3):
    for j in range(col3):
        s = 255 * pow((img3[i][j]/255), gamma)
        newImg3[i][j] = s
        
cv.imshow(f'Image 3 with gamma{gamma}', newImg3)
cv.waitKey(0)  


gamma = 1.2
for i in range(row1):
    for j in range(col1):
        s = 255 * pow((img1[i][j]/255), gamma)
        newImg1[i][j] = s
        
cv.imshow(f'Image 1 with gamma{gamma}', newImg1)
cv.waitKey(0)

for i in range(row2):
    for j in range(col2):
        s = 255 * pow((img2[i][j]/255), gamma)
        newImg2[i][j] = s
        
cv.imshow(f'Image 2 with gamma{gamma}', newImg2)
cv.waitKey(0)

for i in range(row3):
    for j in range(col3):
        s = 255 * pow((img3[i][j]/255), gamma)
        newImg3[i][j] = s
        
cv.imshow(f'Image 3 with gamma{gamma}', newImg3)
cv.waitKey(0)      


gamma = 1.8
for i in range(row1):
    for j in range(col1):
        s = 255 * pow((img1[i][j]/255), gamma)
        newImg1[i][j] = s
        
cv.imshow(f'Image 1 with gamma{gamma}', newImg1)
cv.waitKey(0)

for i in range(row2):
    for j in range(col2):
        s = 255 * pow((img2[i][j]/255), gamma)
        newImg2[i][j] = s
        
cv.imshow(f'Image 2 with gamma{gamma}', newImg2)
cv.waitKey(0)

for i in range(row3):
    for j in range(col3):
        s = 255 * pow((img3[i][j]/255), gamma)
        newImg3[i][j] = s
        
cv.imshow(f'Image 3 with gamma{gamma}', newImg3)
cv.waitKey(0) 

#Log Transformation
c = 255 / np.log10(1 + np.max(img1))
for i in range(row1):
    for j in range(col1):
        s = c * np.log10(1 + img1[i][j])
        newImg1[i][j] = s
        
cv.imshow('Log Transformed Image 1', newImg1)
cv.waitKey(0)

c = 255 / np.log10(1 + np.max(img2))
for i in range(row2):
    for j in range(col2):
        s = c * np.log10(1 + img2[i][j])
        newImg2[i][j] = s
        
cv.imshow('Log Transformed Image 2', newImg2)
cv.waitKey(0)

c = 255 / np.log10(1 + np.max(img3))
for i in range(row3):
    for j in range(col3):
        s = c * np.log10(1 + img3[i][j])
        newImg3[i][j] = s
        
cv.imshow('Log Transformed Image 3', newImg3)
cv.waitKey(0)

cv.destroyAllWindows() """

# TASK 4
""" img = cv.imread("E:/6th Semester/DIP/Lab/lab1.png", 0)
cv.imshow('Image', img)
cv.waitKey(0)

row, col = img.shape

newImg = np.zeros((row, col), dtype=np.uint8)

for i in range(row):
    for j in range(col):
        if img[i][j] <= 200 and img[i][j] >= 100:
            newImg[i][j] = 210
        else:
            newImg[i][j] = img[i][j]
        
cv.imshow('Gray Level Sliced Image', newImg)
cv.waitKey(0)
cv.destroyAllWindows()

img = cv.imread("E:/6th Semester/DIP/Lab/lab2.2.png", 0)
cv.imshow('Image', img)
cv.waitKey(0)

row, col = img.shape

newImg = np.zeros((row, col), dtype=np.uint8)

for i in range(row):
    for j in range(col):
        if img[i][j] <= 200 and img[i][j] >= 100:
            newImg[i][j] = 210
        else:
            newImg[i][j] = img[i][j]
        
cv.imshow('Gray Level Sliced Image', newImg)
cv.waitKey(0)
cv.destroyAllWindows() """


# HOME TASK
img = cv.imread("E:/6th Semester/DIP/Lab/scanned.jpg", 0)
cv.imshow('Original Document', img)
cv.waitKey(0)

row, col = img.shape

imgMean = np.mean(img)

binary_img = np.zeros((row, col), dtype=np.uint8)

for i in range(row):
    for j in range(col):
        if img[i][j] >= imgMean:
            binary_img[i][j] = 255
        else:
            binary_img[i][j] = 0

cv.imshow('Binary Image', binary_img)
cv.waitKey(0)


gamma = 0.2 
enhanced_img = np.power(img / 255.0, gamma) * 255
enhanced_img = enhanced_img.astype(np.uint8)
cv.imshow(f'Enhanced Image for Gamma {gamma})', enhanced_img)
cv.waitKey(0)


x, y, w, h = 100, 100, 200, 200  
cropped_img = enhanced_img[y:y+h, x:x+w]
cv.imshow('Cropped Image', cropped_img)
cv.waitKey(0)


def connecting(newimage):
    rows, cols = newimage.shape
    new = np.zeros((rows, cols), dtype=np.uint8) # label mat
    newDict = {}
    #threshold = 254
    val = 10
    count = 0

    for i in range(rows):
        for j in range(cols):
            temp = newimage[i][j]
            
            left = new[i][j - 1] if j > 0 else 0
            top = new[i - 1][j] if i > 0 else 0
            
            '''if temp <= threshold:
                new[i][j] = 0
                count = 0'''
                
            #elif temp > threshold:  
            if temp == 255:
                new[i][j] = 255
                count = 0
            elif temp == 0:
                if left == 255 and top == 255:  
                    if count == 0:
                        val = (val + 1) % 250
                    newDict[val] = val 
                    new[i][j] = val
                    count += 1
                    #print(newDict)   
                elif left < 255 and top == 255:
                    new[i][j] = left
                    count += 1
                    #cv.imshow('new', new)
                    #cv.waitKey(0)
                elif left == 255 and top < 255:
                    new[i][j] = top
                    count += 1
                elif left < 255 and top < 255:
                    smaller = min(left, top)
                    newDict[max(left, top)] = smaller
                    count += 1 
                    #print("DIF")
                else: 
                    count += 1
                    new[i][j] = left
                    count += 1
                    #print("EQ")

    #cv.imshow('new', new)
    #cv.waitKey(0)                
    #print(newDict)
    #print("Unique labels before correction:", np.unique(new))
    

    for i in range(rows):
        for j in range(cols):
            if new[i][j] < 255: 
                temp = newDict.get(new[i][j], 255)
                new[i][j] = temp
                    
    cv.imshow('Connected Component Image', new)
    cv.waitKey(0)
    #print("Unique labels after correction:", np.unique(new))
    

connecting(binary_img)
