import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def apply_padding(image, kernel_size, pad_value):
    padding_size = kernel_size // 2
    return cv.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv.BORDER_CONSTANT, value=pad_value)


def normal_lbp(img):
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    
    lbpFinal = np.zeros_like(img, dtype=np.uint8)
    
    kernel_size = 3
    rep = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    
    padded_img = apply_padding(img, kernel_size, 0)
    height, width = img.shape
    
    for i in range(height):
        for j in range(width):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            center_pixel = region[1, 1]
            
            binary_values = (region >= center_pixel).astype(np.uint8)
            
            ind = 0
            for k in range(3):
                for l in range(3):
                    if k == 1 and l == 1:
                        continue
                    rep[k, l] = binary_values[k, l] * weights[ind]
                    ind += 1
                    
            lbpFinal[i, j] = np.sum(rep)
    
    return lbpFinal

def rotated_lbp(img):
    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    
    lbpFinal = np.zeros_like(img, dtype=np.uint8)
    
    kernel_size = 3
    rep = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    
    padded_img = apply_padding(img, kernel_size, 0)
    height, width = img.shape
    
    for i in range(height):
        for j in range(width):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            center_pixel = region[1, 1]
            
            binary_values = (region >= center_pixel).astype(np.uint8)
            
            neighbor_pixels = np.zeros(8, dtype=np.uint8)
            ind1 = 0
            for m in range(3):
                for n in range(3):
                    if m == 1 and n == 1:
                        continue 
                    neighbor_pixels[ind1] = binary_values[m, n]
                    ind1 += 1
                    
            dif = np.abs(neighbor_pixels - center_pixel)
            D = np.argmax(dif)
            rotated = np.roll(binary_values, -D)
            
            ind = 0
            for k in range(3):
                for l in range(3):
                    if k == 1 and l == 1:
                        continue
                    rep[k, l] = rotated[k, l] * weights[ind]
                    ind += 1
                    
            lbpFinal[i, j] = np.sum(rep)
    
    return lbpFinal

def main():
    img = cv.imread("E:\\6th Semester\\DIP\\Lab\\Lab 10\\Lab 10\\image3.png", cv.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image")
        return
    
    cv.imshow("Original Image", img)
    cv.waitKey(0)
    
    lbp = normal_lbp(img)
    cv.imshow("LBP Image", lbp)
    cv.waitKey(0)
    
    plt.hist(lbp.ravel(), bins=256, range=(0, 256))
    plt.title("Histogram of Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()    
    
    lbpr = rotated_lbp(img)
    cv.imshow("LBP R Image", lbp)
    cv.waitKey(0)
    
    plt.hist(lbpr.ravel(), bins=256, range=(0, 256))
    plt.title("Histogram of LBP R Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()    
    
main()


""" import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2

# Step 1: Sobel gradient function
def apply_sobel_filter(image, kernel):
    kernel_size = kernel.shape[0]
    num_pad_pixels = kernel_size // 2
    padded_image = np.pad(image, pad_width=num_pad_pixels, mode='constant', constant_values=0)
    output_image = np.zeros_like(image, dtype=np.float32)

    for k in range(num_pad_pixels, image.shape[0] + num_pad_pixels):
        for l in range(num_pad_pixels, image.shape[1] + num_pad_pixels):
            sub_image = padded_image[k - num_pad_pixels:k + num_pad_pixels + 1, l - num_pad_pixels:l + num_pad_pixels + 1]
            output_image[k - num_pad_pixels, l - num_pad_pixels] = np.sum(sub_image * kernel)

    return output_image

# Load grayscale image and resize
picam2 = Picamera2()
picam2.start()
frame = picam2.capture_array()
cv2.imshow("live", frame)
cv2.waitKey(0)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
image = cv2.resize(image, (128, 64))

# Step 2: Compute gradients
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

grad_x = apply_sobel_filter(image, sobel_x)
grad_y = apply_sobel_filter(image, sobel_y)

magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
angle = (np.arctan2(grad_y, grad_x) * 180 / np.pi) % 180

# Parameters
cell_size = 8
block_size = 2
bin_count = 9
bin_width = 180 // bin_count

height, width = image.shape
cells_y = height // cell_size
cells_x = width // cell_size

# Step 3: Compute histogram for each cell
cell_hist = np.zeros((cells_y, cells_x, bin_count), dtype=np.float32)

for i in range(cells_y):
    for j in range(cells_x):
        for y in range(cell_size):
            for x in range(cell_size):
                row = i * cell_size + y
                col = j * cell_size + x
                mag = magnitude[row, col]
                ang = angle[row, col]
                bin_idx = int(ang // bin_width) % bin_count
                cell_hist[i, j, bin_idx] += mag

# Step 4: Compute HoG descriptor by blocks and concatenate
block_stride = 1
hog_descriptor = []

blocks_y = cells_y - block_size + 1
blocks_x = cells_x - block_size + 1

for i in range(blocks_y):
    for j in range(blocks_x):
        block = cell_hist[i:i+block_size, j:j+block_size, :].flatten()

        # Manual L2 normalization
        epsilon = 1e-6
        l2_sum = 0
        for val in block:
            l2_sum += val * val
        l2_norm = (l2_sum) ** 0.5 + epsilon

        # Normalize manually
        for idx in range(len(block)):
            block[idx] = block[idx] / l2_norm

        # Manual Clipping
        for idx in range(len(block)):
            if block[idx] < 0:
                block[idx] = 0
            elif block[idx] > 0.2:
                block[idx] = 0.2

        # Renormalize manually after clipping
        l2_sum_clip = 0
        for val in block:
            l2_sum_clip += val * val
        l2_norm_clip = (l2_sum_clip) ** 0.5 + epsilon

        for idx in range(len(block)):
            block[idx] = block[idx] / l2_norm_clip

        hog_descriptor.extend(block)

hog_descriptor = np.array(hog_descriptor)

# Step 5: Display HoG Descriptor
print("\nHoG Descriptor (first 100 values) ")
print(hog_descriptor[:100])
print(f"\nTotal HoG Descriptor Length: {len(hog_descriptor)}")

# Step 6: Display full histogram for the whole image
# Divide descriptor into 9-bin segments and sum each bin across all
hog_hist = np.zeros(bin_count)
for i in range(0, len(hog_descriptor), bin_count):
    hog_hist += hog_descriptor[i:i+bin_count]

# Bin centers
bin_centers = np.arange(0, 180, 20) + 10

plt.figure(figsize=(7, 4))
plt.bar(bin_centers, hog_hist, width=18, align='center', color='salmon', edgecolor='black')
plt.title("Histogram of Oriented Gradients (Entire Image)")
plt.xlabel("Orientation (degrees)")
plt.ylabel("Summed Magnitude")
plt.xticks(bin_centers)
plt.grid(True)
plt.tight_layout()
plt.show() """