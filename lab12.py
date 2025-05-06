import cv2 as cv
import numpy as np
import skimage
from skimage.feature import graycomatrix
import matplotlib.pyplot as plt
import math
import skimage.feature

# TASK - 01
def calculate_histogram_properties(image):
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()

    hist_prob = hist / np.sum(hist)

    mean = np.sum(np.arange(256) * hist_prob)
    variance = np.sum((np.arange(256) - mean) ** 2 * hist_prob)
    skewness = np.sum((np.arange(256) - mean) ** 3 * hist_prob) / (variance ** 1.5)
    uniformity = np.sum(hist_prob ** 2)

    # Calculate entropy
    entropy = -np.sum([p * math.log2(p) for p in hist_prob if p > 0])

    # Display results
    print(f"Skewness: {skewness}")
    print(f"Uniformity: {uniformity}")
    print(f"Entropy: {entropy}")


    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()
    
def main():
    original_image = cv.imread(r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image3.png",  cv.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    calculate_histogram_properties(original_image)
    
    original_image = cv.imread(r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image2.png",  cv.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    calculate_histogram_properties(original_image)
    
    original_image = cv.imread(r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image1.png",  cv.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    calculate_histogram_properties(original_image)
    
#main()


# TASK - 02
def GLCM_parameters(img):
    # Calculate GLCM (distance=1, angle=0°)
    glcm = skimage.feature.graycomatrix(
        image=img,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=False,  # Count (i,j) and (j,i) separately
        normed=True  # Normalize to probabilities
    )


    max_probability= np.max(glcm)
    contrast= skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
    energy= skimage.feature.graycoprops(glcm, 'energy')[0, 0]
    entropy= -np.sum([p * math.log2(p) for p in glcm.flatten() if p > 0])

    # Display results
    print(f"Max_prob: {max_probability}")
    print(f"Contrast: {contrast}")
    print(f"Energy: {energy}")
    print(f"Entropy: {entropy}")


    return 0

def main2():
    original_image = cv.imread(r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image3.png",  cv.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    GLCM_parameters(original_image)
    
    original_image = cv.imread(r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image2.png",  cv.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    GLCM_parameters(original_image)
    
    original_image = cv.imread(r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image1.png",  cv.IMREAD_GRAYSCALE)
    if original_image is None:
        print("Error: Unable to load image.")
        return
    
    cv.imshow("Original Image", original_image)
    cv.waitKey(0)
    
    GLCM_parameters(original_image)
    
#main2()

# TASK - 03
def spectral_analysis(image):
    # Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    # Get the dimensions of the image
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Create a grid of coordinates
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - center_col)**2 + (y - center_row)**2)
    theta = np.arctan2(y - center_row, x - center_col)

    # Radial profile S(r)
    r_bins = np.arange(0, np.max(r), 1)
    radial_profile = np.zeros_like(r_bins, dtype=np.float64)
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i + 1])
        radial_profile[i] = np.mean(magnitude_spectrum[mask])

    # Angular profile S(theta)
    theta_bins = np.linspace(-np.pi, np.pi, 360)
    angular_profile = np.zeros_like(theta_bins, dtype=np.float64)
    for i in range(len(theta_bins) - 1):
        mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1])
        angular_profile[i] = np.mean(magnitude_spectrum[mask])

    # Plot the radial profile S(r)
    plt.figure()
    plt.plot(r_bins[:-1], radial_profile[:-1])
    plt.title("Radial Profile S(r)")
    plt.xlabel("Radius (r)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

    # Plot the angular profile S(theta)
    plt.figure()
    plt.plot(theta_bins[:-1], angular_profile[:-1])
    plt.title("Angular Profile S(theta)")
    plt.xlabel("Angle (theta)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

def main3():
    # Load and process images
    image_paths = [
        r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image5.png",
        r"E:\6th Semester\DIP\Lab\Lab 12\Lab 12\image4.png"
    ]

    for path in image_paths:
        original_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if original_image is None:
            print(f"Error: Unable to load image {path}.")
            continue

        cv.imshow("Original Image", original_image)
        cv.waitKey(0)

        spectral_analysis(original_image)

main3()