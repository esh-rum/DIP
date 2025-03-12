import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TASK - 1
def apply_padding(image, kernel_size, pad_value):
    padding_size = kernel_size // 2
    return cv.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=pad_value)

def apply_median_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.median(region)
    
    return filtered_image

def apply_min_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.min(region)
    
    return filtered_image

def apply_max_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.max(region)
    
    return filtered_image

def main():
    image_path = "E:/6th Semester/DIP/Lab/Lab 5/Lab 5/Fig01.tif"
    image_path2 = "E:/6th Semester/DIP/Lab/Lab 5/Lab 5/Fig02.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    # MEDIAN
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 15, 31]:
        filtered_image = apply_median_filter(original_image, kernel_size)
        cv.imshow(f"Median Filter {kernel_size}x{kernel_size}", filtered_image) 
        cv.waitKey(0)       
    
    
    original_image2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
    
    if original_image2 is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image2", original_image2)
    cv.waitKey(0)
    
    for kernel_size in [3, 15, 31]:
        filtered_image2 = apply_median_filter(original_image2, kernel_size)
        cv.imshow(f"Median Filter {kernel_size}x{kernel_size}", filtered_image2)
        cv.waitKey(0)
    
    # MIN 
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 15, 31]:
        filtered_image = apply_min_filter(original_image, kernel_size)
        cv.imshow(f"Min Filter {kernel_size}x{kernel_size}", filtered_image)  
        cv.waitKey(0)      
    
    
    original_image2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
    
    if original_image2 is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image2", original_image2)
    cv.waitKey(0)
    
    for kernel_size in [3, 15, 31]:
        filtered_image2 = apply_min_filter(original_image2, kernel_size)
        cv.imshow(f"Min Filter {kernel_size}x{kernel_size}", filtered_image2)
        cv.waitKey(0)
    
    # MAX
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 15, 31]:
        filtered_image = apply_max_filter(original_image, kernel_size)
        cv.imshow(f"Max Filter {kernel_size}x{kernel_size}", filtered_image)    
        cv.waitKey(0)    
    
    
    original_image2 = cv.imread(image_path2, cv.IMREAD_GRAYSCALE)
    
    if original_image2 is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image2", original_image2)
    cv.waitKey(0)
    
    for kernel_size in [3, 15, 31]:
        filtered_image2 = apply_max_filter(original_image2, kernel_size)
        cv.imshow(f"Median Filter {kernel_size}x{kernel_size}", filtered_image2)
        cv.waitKey(0)
    
    cv.destroyAllWindows()


main()
    
    
# TASK - 2
def apply_padding(image, kernel_size, pad_value):
    padding_size = kernel_size // 2
    return cv.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=pad_value)

def apply_hori_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.float64)
    padded_image = apply_padding(image, kernel_size, 0)
    hori = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(region * hori)
    
    return filtered_image

def apply_veri_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.float64)
    padded_image = apply_padding(image, kernel_size, 0)
    veri = np.array([[-1,0,-1],[-2,0,2],[-1,0,1]])
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(region * veri)
    
    return filtered_image

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val) 
    return normalized


def main2():
    image_path = "E:/6th Semester/DIP/Lab/Lab 5/Lab 5/Fig03.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    # Horizontal Sobel 
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    hori_image = apply_hori_filter(original_image, 3) 
    cv.imshow(f"Sobel Horizontal Image", hori_image)  
    cv.waitKey(0)      
    
    # Vertical Sobel 
    verti_image = apply_veri_filter(original_image, 3) 
    cv.imshow(f"Sobel Vertital Image", verti_image)  
    cv.waitKey(0)   
    
    row, col = original_image.shape
    final = np.zeros((row, col), dtype=np.uint8)
    
    final = np.sqrt(hori_image ** 2 + verti_image ** 2)
    final = normalize_image(final)
    
    cv.imshow("Mag Image", final)  
    cv.waitKey(0)   
    
    phase = np.arctan2(verti_image, hori_image)  # Compute angles in radians
    phase = normalize_image(phase)  

    cv.imshow("Phase Image", phase)
    cv.waitKey(0)   
    
    cv.destroyAllWindows()

#main2()

# TASK -3 
def apply_padding(image, kernel_size, pad_value):
    padding_size = kernel_size // 2
    return cv.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=pad_value)

def apply_lap_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    padded_image = apply_padding(image, kernel_size, 0)
    lap = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(region * lap)
    
    return filtered_image

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized = (image - min_val) / (max_val - min_val) 
    return normalized


def main3():
    image_path = "E:/6th Semester/DIP/Lab/Lab 5/Lab 5/Fig03.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    lap_image = apply_lap_filter(original_image, 3) 
    cv.imshow(f"Laplacian Image", lap_image)  
    cv.waitKey(0)      
    
    sharpened_image = (original_image + lap_image)

    cv.imshow("Sharpened Image", sharpened_image)
    cv.waitKey(0)  
    
    cv.destroyAllWindows()
    
#main3()