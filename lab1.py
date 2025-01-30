import numpy as np
import cv2 as cv

# TASK 1
def minmax(arr):
    minVal = min(arr)
    maxVal = max(arr)
    scaledArr = [0 for _ in range(len(arr))]
    for i in range(len(arr)):
        scaledArr[i] = (arr[i] - minVal) / (maxVal - minVal)
        
    return scaledArr

# TASK 2
def padMatrix(mat, padMat):
    mat = mat * 255    # 255 -> white, 0 -> black
    cv.imshow('Original Image', mat)
    cv.waitKey(0)
    
    padMat[5:505, 5:505] = mat
    
    return padMat

# TASK 3
def verticalLines(mat, sizeVertLines):
    cv.imshow('Original Image', mat)
    cv.waitKey(0)
    new = mat.copy()
    j = 0
    while j <= np.shape(mat)[1]:
        new[:, j:(j+sizeVertLines)] = 0     # selects all rows and columns from j to j+sizeVertLines
        j += 2 * sizeVertLines + 1
        
    cv.imshow('Vertical Lines Image', new)
    cv.waitKey(0)
    
def border(mat, borderSize):
    cv.imshow('Original Image', mat)
    cv.waitKey(0)
    new = mat.copy()
    new[0:borderSize, :] = 0
    new[-borderSize:, :] = 0
    new[:, 0:borderSize] = 0
    new[:, -borderSize:] = 0
    
    cv.imshow('Border Image', new)
    cv.waitKey(0)

def boxes(mat, lineWidth):
    cv.imshow('Original Image', mat)
    cv.waitKey(0)
    new = mat.copy()
    j = 0
    while j <= np.shape(mat)[1]:
        new[:, j:(j+lineWidth)] = 0     # selects all rows and columns from j to j+lizeWidth
        j += 2 * lineWidth + 1
        
    k = 0
    while k <= np.shape(mat)[0]:
        new[k:(k+lineWidth), :] = 0     # selects all columns and rows from k to k+lineWidth
        k += 2 * lineWidth + 1
        
    cv.imshow('Vertical Lines Image', new)
    cv.waitKey(0)
    

def main1():
    #TASK 1
    arr = [1, 2, 3, 4, 5]
    print(minmax(arr))
    
def main2():
    #TASK 2
    mat = np.ones((500, 500), dtype = np.uint8)      # 500x500 matrix containing zeros -> a pic matrix
    padMat = np.zeros((510, 510), dtype = np.uint8)
    
    paddedMat = padMatrix(mat, padMat)
    cv.imshow('Padded Image', paddedMat)
    cv.waitKey(0)
    
    print(f'Original Image Shape: {mat.shape} and Padded Image Shape: {paddedMat.shape}')
    
def main3():    
    #TASK 3
    val = int(input('To start enter 1, to end enter 0: '))
    
    while val:
        rows = int(input('Enter number of rows for image: '))
        cols = int(input('Enter number of columns for image: '))
        mat2 = mat = np.ones((rows, cols), dtype = np.uint8) * 255
        
        sizeVertLines = int(input('Enter width of vertical lines: '))
        verticalLines(mat2, sizeVertLines)
        
        borderSize = int(input('Enter width of border: '))
        border(mat2, borderSize)
        
        lineWidth = int(input('Enter width of lines for boxes: '))
        boxes(mat2, lineWidth)
        
        val = int(input('To continue enter 1, to end enter 0: '))
    
def main4():
    #TASK 4
    img = cv.imread('E:/6th Semester/DIP/Lab/lab1.png', 0)
    cv.imshow('Image', img)
    cv.waitKey(0)
    print(img.shape)
    
    # Resizing the image
    rows, cols = img.shape
    newRows = 512
    newCols = 512
    
    resizedImg = np.zeros((newRows, newCols), dtype=np.uint8)
    scaleRows = rows / newRows
    scaleCols = cols / newCols
    
    for i in range(newRows):
        for j in range(newCols):
            resizedImg[i, j] = img[(int(i * scaleRows)), (int(j * scaleCols))]
    
    cv.imshow('Resized Image', resizedImg)
    cv.waitKey(0)
    print(resizedImg.shape)
    
    # Downsample the image
    newRows2 = 128
    newCols2 = 128
    
    downImg = np.zeros((newRows2, newCols2), dtype=np.uint8)
    scaleRows2 = rows / newRows2
    scaleCols2 = cols / newCols2
    
    for i in range(newRows2):
        for j in range(newCols2):
            downImg[i, j] = img[(int(i * scaleRows2)), (int(j * scaleCols2))]
    
    cv.imshow('Downsampled Image', downImg)
    cv.waitKey(0)
    print(downImg.shape)
    
def main5():
    # TASK 5
    val = int(input('To start enter 1, to end enter 0: '))
    
    while val:
        rows = int(input('Enter number of rows for image: '))
        cols = int(input('Enter number of columns for image: '))
        mat2 = np.ones((rows, cols, 3), dtype = np.uint8) * 255
        
        blue = [255, 0, 0]
        green = [0, 255, 0]
        red = [0, 0, 255]
        
        boxSize1 = int(rows / 8)
        boxSize2 = int(cols / 8)
        
        mat2[0:boxSize1, 0:boxSize2] = 0
        mat2[0:boxSize1, -boxSize2:] = blue
        mat2[-boxSize1:, 0:boxSize2] = green   
        mat2[-boxSize1:, -boxSize2:] = red
        
        cv.imshow('Image', mat2)
        cv.waitKey(0)
                
        val = int(input('To continue enter 1, to end enter 0: '))
    
    
#main1()
#main2() 
main3()
#main4()
#main5()

