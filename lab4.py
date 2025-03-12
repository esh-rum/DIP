import cv2
import numpy as np
import matplotlib.pyplot as plt

# TASK - 1
""" def apply_contrast_stretch(image, height, width):
    lower_bound = np.percentile(image, 5)
    upper_bound = np.percentile(image, 95)
    
    for row in range(height):
        for col in range(width):
            pixel = image[row, col]
            if pixel < lower_bound:
                image[row, col] = 0
            elif pixel > upper_bound:
                image[row, col] = 255
            else:
                image[row, col] = 255 * (pixel - lower_bound) / (upper_bound - lower_bound)
    
    return image

def main():
    image_path = "E:/6th Semester/DIP/Lab/Lab 4/Lab 4/low_con.jpg"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    height, width = original_image.shape
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)
    
    enhanced_image = apply_contrast_stretch(original_image, height, width)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() """
    
    
# TASK - 2
""" def histogram_equalization(image, height, width):
    total_pixels = height * width
    hist_values = np.zeros(256, dtype=np.uint32)
    
    for row in range(height):
        for col in range(width):
            pixel = image[row, col]
            hist_values[pixel] += 1
    
    plt.plot(hist_values)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    probability_density = hist_values / total_pixels
    transformation_function = np.zeros(256, dtype=np.float32)
    cumulative_sum = 0.0
    
    for intensity in range(256):
        cumulative_sum += probability_density[intensity]
        transformation_function[intensity] = round(cumulative_sum * 255)
    
    for row in range(height):
        for col in range(width):
            pixel = image[row, col]
            image[row, col] = transformation_function[pixel]
    
    plt.plot(transformation_function)
    plt.xlabel("Input Pixel Intensity")
    plt.ylabel("Output Pixel Intensity")
    plt.show()
    
    return image

def main():
    image_path = "E:/6th Semester/DIP/Lab/Lab 4/Lab 4/fig05.tif"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    height, width = original_image.shape
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)
    
    equalized_image = histogram_equalization(original_image, height, width)
    cv2.imshow("Equalized Image", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() """
    

# TASK - 3
def create_filter(kernel_size, divisor):
    return np.ones((kernel_size, kernel_size), dtype=np.float32) / divisor

def apply_padding(image, kernel_size, pad_value):
    padding_size = kernel_size // 2
    return cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=pad_value)

def apply_filter(image, kernel, kernel_size):
    height, width = image.shape
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    pad_size = kernel_size // 2
    padded_image = apply_padding(image, kernel_size, 0)
    
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.sum(region * kernel)
    
    return filtered_image

def main():
    image_path = "E:/6th Semester/DIP/Lab/Lab 4/Lab 4/fig05.tif"
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv2.imshow("Original Image", original_image)
    cv2.waitKey(0)
    
    kernel = create_filter(3, 9)
    filtered_image = apply_filter(original_image, kernel, 3)
    
    cv2.imshow("Filtered Image", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
