import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to compute DFT manually
def manual_dft(image):
    z=0
    rows, cols = image.shape
    dft = np.zeros((rows, cols), dtype=complex)
    for u in range(rows):
        for v in range(cols):
            sum_val = 0
            for x in range(rows):
                for y in range(cols):
                    sum_val += image[x, y] * np.exp(-2j * np.pi * ((u * x / rows) + (v * y / cols)))
                    print(z)
                    z+=1
            dft[u, v] = sum_val
    return dft

# Apply the transformation F(x, y) = F(x, y) * (-1)^(x + y)
def apply_transformation(dft):
    rows, cols = dft.shape
    transformed_dft = np.zeros_like(dft, dtype=complex)
    for x in range(rows):
        for y in range(cols):
            transformed_dft[x, y] = dft[x, y] * ((-1) ** (x + y))
    return transformed_dft



def main():
    # Load the image
    img = cv2.imread("E:\\6th Semester\\DIP\\Lab\\LAB11 (1)\\LAB11\\Fig01 (1).tif", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    # Compute DFT manually
    manual_dft_result = manual_dft(image)
    transformed_manual_dft = apply_transformation(manual_dft_result)
    
    # Compute DFT using built-in functions
    dft_builtin = np.fft.fft2(image)
    dft_builtin_shifted = np.fft.fftshift(dft_builtin)

    # Display results
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    # Manual DFT
    plt.subplot(2, 3, 2)
    plt.title("Manual DFT (Magnitude)")
    plt.imshow(np.log(1 + np.abs(manual_dft_result)), cmap='gray')

    # Transformed Manual DFT
    plt.subplot(2, 3, 3)
    plt.title("Transformed Manual DFT")
    plt.imshow(np.log(1 + np.abs(transformed_manual_dft)), cmap='gray')

    # Built-in DFT
    plt.subplot(2, 3, 4)
    plt.title("Built-in DFT (Magnitude)")
    plt.imshow(np.log(1 + np.abs(dft_builtin)), cmap='gray')

    # Built-in DFT (Shifted)
    plt.subplot(2, 3, 5)
    plt.title("Built-in DFT (Shifted)")
    plt.imshow(np.log(1 + np.abs(dft_builtin_shifted)), cmap='gray')

    plt.tight_layout()
    plt.show()
    
#main()

# TASK - 02
# Function to create a distance map
def create_distance_map(rows, cols):
    center_x, center_y = rows // 2, cols // 2
    distance_map = np.zeros((rows, cols))
    for x in range(rows):
        for y in range(cols):
            distance_map[x, y] = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return distance_map

# Function to create an ideal low-pass filter
def ideal_low_pass_filter(distance_map, cutoff):
    return (distance_map <= cutoff).astype(float)

# Function to create an ideal high-pass filter
def ideal_high_pass_filter(distance_map, cutoff):
    return (distance_map > cutoff).astype(float)

# Main function
def main2():
    # Load the image
    image = cv2.imread("E:\\6th Semester\\DIP\\Lab\\LAB11 (1)\\LAB11\\Fig01 (1).tif", cv2.IMREAD_GRAYSCALE)

    # Compute DFT using built-in functions
    dft_builtin = np.fft.fft2(image)
    dft_builtin_shifted = np.fft.fftshift(dft_builtin)

    # Create distance map
    rows, cols = image.shape
    distance_map = create_distance_map(rows, cols)

    # Create filters
    cutoff = 30  # Adjust cutoff frequency as needed
    low_pass_filter = ideal_low_pass_filter(distance_map, cutoff)
    high_pass_filter = ideal_high_pass_filter(distance_map, cutoff)

    # Apply filters
    low_pass_result = dft_builtin_shifted * low_pass_filter
    high_pass_result = dft_builtin_shifted * high_pass_filter

    # Inverse DFT
    low_pass_image = np.fft.ifft2(np.fft.ifftshift(low_pass_result)).real
    high_pass_image = np.fft.ifft2(np.fft.ifftshift(high_pass_result)).real

    # Display results
    plt.figure(figsize=(12, 8))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    # Low-Pass Filter
    plt.subplot(2, 3, 2)
    plt.title("Low-Pass Filter")
    plt.imshow(low_pass_filter, cmap='gray')

    # High-Pass Filter
    plt.subplot(2, 3, 3)
    plt.title("High-Pass Filter")
    plt.imshow(high_pass_filter, cmap='gray')

    # Low-Pass Filtered Image
    plt.subplot(2, 3, 4)
    plt.title("Low-Pass Filtered Image")
    plt.imshow(low_pass_image, cmap='gray')

    # High-Pass Filtered Image
    plt.subplot(2, 3, 5)
    plt.title("High-Pass Filtered Image")
    plt.imshow(high_pass_image, cmap='gray')

    plt.tight_layout()
    plt.show()

#main2()

# TASK - 03
# Function to create a Gaussian filter in the spatial domain
def create_gaussian_filter(size, sigma):
    half_size = size // 2
    gaussian_filter = np.zeros((size, size))
    for i in range(-half_size, half_size + 1):
        for j in range(-half_size, half_size + 1):
            gaussian_filter[i + half_size, j + half_size] = np.exp(-(i**2 + j**2) / (2 * sigma**2))
    return gaussian_filter / np.sum(gaussian_filter)  # Normalize the filter

# Function to pad an image or filter to a specified size
def pad_to_size(array, target_size):
    padded_array = np.zeros(target_size, dtype=array.dtype)
    rows, cols = array.shape
    padded_array[:rows, :cols] = array
    return padded_array

# Main function
def main3():
    # Load the image
    image = cv2.imread("E:\\6th Semester\\DIP\\Lab\\LAB11 (1)\\LAB11\\Fig01 (1).tif", cv2.IMREAD_GRAYSCALE)

    # Create a Gaussian filter
    filter_size = 9
    sigma = 5
    gaussian_filter = create_gaussian_filter(filter_size, sigma)

    # Calculate the new size for padding
    new_size = (image.shape[0] + filter_size - 1, image.shape[1] + filter_size - 1)

    # Pad the image and filter to the new size
    padded_image = pad_to_size(image, new_size)
    padded_filter = pad_to_size(gaussian_filter, new_size)

    # Compute DFT of the padded image and filter
    dft_image = np.fft.fft2(padded_image)
    dft_filter = np.fft.fft2(padded_filter)

    # Dot multiply in the frequency domain
    filtered_dft = dft_image * dft_filter

    # Compute the inverse DFT to return to the spatial domain
    filtered_image = np.fft.ifft2(filtered_dft).real

    # Display results
    plt.figure(figsize=(12, 8))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    # Gaussian Filter (Spatial Domain)
    plt.subplot(2, 3, 2)
    plt.title("Gaussian Filter (Spatial Domain)")
    plt.imshow(gaussian_filter, cmap='gray')

    # Gaussian Filter (Frequency Domain)
    plt.subplot(2, 3, 3)
    plt.title("Gaussian Filter (Frequency Domain)")
    plt.imshow(np.log(1 + np.abs(np.fft.fftshift(dft_filter))), cmap='gray')

    # Padded Image
    plt.subplot(2, 3, 4)
    plt.title("Padded Image")
    plt.imshow(padded_image, cmap='gray')

    # Filtered Image (Spatial Domain)
    plt.subplot(2, 3, 5)
    plt.title("Filtered Image (Spatial Domain)")
    plt.imshow(filtered_image, cmap='gray')

    plt.tight_layout()
    plt.show()

main3()