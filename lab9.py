import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# TASK - 1
def apply_padding(image, kernel_size, pad_value):
    padding_size = kernel_size // 2
    return cv.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=pad_value)

def apply_erosion_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    mask = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            if np.all(region[mask == 1] == 255):
                filtered_image[i, j] = 255
    
    return filtered_image

def main():
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\fp.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 5, 10]:
        filtered_image = apply_erosion_filter(original_image, kernel_size)
        cv.imshow(f"Erosion Filter {kernel_size}x{kernel_size} on FP Image", filtered_image) 
        cv.waitKey(0)
        
    
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\Fig01.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 5, 10]:
        filtered_image = apply_erosion_filter(original_image, kernel_size)
        cv.imshow(f"Erosion Filter {kernel_size}x{kernel_size} on Fig01", filtered_image) 
        cv.waitKey(0)
    
main()


# TASK - 2
def apply_dilation_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    mask = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            if np.any(region[mask == 1] == 255):
                filtered_image[i, j] = 255
    
    return filtered_image

def main2():
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\broken_text.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 5, 10]:
        filtered_image = apply_dilation_filter(original_image, kernel_size)
        cv.imshow(f"Dilation Filter {kernel_size}x{kernel_size} on Text Image", filtered_image) 
        cv.waitKey(0)
        
    
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\Fig01.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3, 5, 10]:
        filtered_image = apply_dilation_filter(original_image, kernel_size)
        cv.imshow(f"Dilation Filter {kernel_size}x{kernel_size} on Fig01", filtered_image) 
        cv.waitKey(0)
        
#main2()

# TASK - 3
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

def main3():
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\Fig01.tif"
    original_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3]:
        filtered_image = apply_min_filter(original_image, kernel_size)
        cv.imshow(f"Min Filter {kernel_size}x{kernel_size} for erosion", filtered_image) 
        cv.waitKey(0)
  
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [3]:
        filtered_image = apply_max_filter(original_image, kernel_size)
        cv.imshow(f"Max Filter {kernel_size}x{kernel_size} for Dilation", filtered_image) 
        cv.waitKey(0)
        
#main3()

# TASK - 4
def apply_erosion_filter2(image, kernel_size, se):
    height, width = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    mask = se
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            if np.all(region[mask == 1] == 255):
                filtered_image[i, j] = 255
    
    return filtered_image

def apply_dilation_filter2(image, kernel_size, se):
    height, width = image.shape
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    mask = se
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            if np.any(region[mask == 1] == 255):
                filtered_image[i, j] = 255
    
    return filtered_image

def create_diamond_mask(kernel_size):
    """Create a diamond-shaped mask of a given kernel size."""
    mask = np.zeros((kernel_size, kernel_size), dtype=np.bool)
    mid = kernel_size // 2
    # Fill the mask with a diamond shape
    for i in range(kernel_size):
        for j in range(kernel_size):
            if abs(i - mid) + abs(j - mid) <= mid:
                mask[i, j] = 1
                
    return mask

def main4():
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\Objects.png"
    original_image = cv.imread(image_path, 0)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [70]: 
        se = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
        filtered_image = apply_erosion_filter2(original_image, kernel_size, se)
        #cv.imshow(f"Rectangle", filtered_image) 
        #cv.waitKey(0)
        
    for kernel_size in [70]:
        se = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
        filtered_image2 = apply_dilation_filter2(filtered_image, kernel_size, se)
        cv.imshow(f"Rectangle", filtered_image2) 
        cv.waitKey(0)
        
    
    for kernel_size in [78]: 
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filtered_image = apply_erosion_filter2(original_image, kernel_size, se)
        #cv.imshow(f"Circle", filtered_image) 
        #cv.waitKey(0)
        
    for kernel_size in [78]:
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        filtered_image2 = apply_dilation_filter2(filtered_image, kernel_size, se)
        cv.imshow(f"Circle", filtered_image2) 
        cv.waitKey(0)
        
    size = 100
    diamond_mask = create_diamond_mask(size)
    filtered_image = apply_erosion_filter2(original_image, size, diamond_mask)
    filtered_image2 = apply_dilation_filter2(filtered_image, size, diamond_mask)
    cv.imshow(f"Diamond", filtered_image2)
    cv.waitKey(0)

        
#main4()


# TASK - 5
def main5():
    image_path = "E:\\6th Semester\\DIP\\Lab\\Lab 9\\Lab 9\\rice.png"
    original_image = cv.imread(image_path, 0)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    for kernel_size in [15]:
        filtered_image = apply_min_filter(original_image, kernel_size)
        #cv.imshow(f"Min Filter {kernel_size}x{kernel_size} for erosion", filtered_image) 
        #cv.waitKey(0)
    
    for kernel_size in [15]:
        filtered_image2 = apply_max_filter(filtered_image, kernel_size)
        #cv.imshow(f"Max Filter {kernel_size}x{kernel_size} for Dilation", filtered_image2) 
        #cv.waitKey(0)
        
    final = original_image - filtered_image2
    cv.imshow(f"Top Hat Transformation", final) 
    cv.waitKey(0)
    
        
#main5()
