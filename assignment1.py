import os
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def find(label_mat, x):
    """Finds the root label of x with path compression."""
    if x not in label_mat:
        return x
    if label_mat[x] != x:
        label_mat[x] = find(label_mat, label_mat[x])  # Path compression
    return label_mat[x]

def isThere(val, array):
    if val in array:
        return True
    return False

def cca(image):
    rows, cols = image.shape
    new_img = np.zeros((rows, cols), dtype=np.uint8) 
    label_mat = {}
    whites = np.arange(150, 255, 1) #bg
    blacks = np.arange(0, 75, 1) #nucleus
    grays = np.arange(76, 149, 1) #cytoplasm
    val = 0
    
    for i in range(rows):
        for j in range(cols):
            temp = image[i][j]
            
            left = new_img[i][j - 1] if j > 0 else 0
            top = new_img[i - 1][j] if i > 0 else 0
            top_left = new_img[i - 1][j - 1] if (i > 0 and j > 0) else 0
            top_right = new_img[i - 1][j + 1] if (i > 0 and j < cols - 1) else 0
            
            neighbors = [left, top, top_left, top_right]
            labeled_neighbors = [n for n in neighbors if n > 0]
            
            if temp == 0:
                new_img[i][j] = 0
            elif isThere(temp, whites):
                if not labeled_neighbors:
                    val += 1
                    label_mat[val] = 255
                    new_img[i][j] = 255
                    print('w')
                else:
                    print('o')
                    smallest_label = min(labeled_neighbors)
                    new_img[i][j] = smallest_label

                    for neighbor in labeled_neighbors:
                        root_a = find(label_mat, smallest_label)
                        root_b = find(label_mat, neighbor)
                        if root_a != root_b:
                            label_mat[root_b] = root_a
            elif isThere(temp, blacks):
                if not labeled_neighbors:
                    val += 1
                    label_mat[val] = 0
                    new_img[i][j] = 0
                    print('b')
                else:
                    print('p')
                    smallest_label = min(labeled_neighbors)
                    new_img[i][j] = smallest_label

                    for neighbor in labeled_neighbors:
                        root_a = find(label_mat, smallest_label)
                        root_b = find(label_mat, neighbor)
                        if root_a != root_b:
                            label_mat[root_b] = root_a
            elif isThere(temp, grays):
                if not labeled_neighbors:
                    val += 1
                    label_mat[val] = 128  
                    new_img[i][j] = 128
                    print('g')
                else:
                    print('q')
                    smallest_label = min(labeled_neighbors)
                    new_img[i][j] = smallest_label

                    for neighbor in labeled_neighbors:
                        root_a = find(label_mat, smallest_label)
                        root_b = find(label_mat, neighbor)
                        if root_a != root_b:
                            label_mat[root_b] = root_a

    for i in range(rows):
        for j in range(cols):
            if new_img[i][j] > 0:
                new_img[i][j] = find(label_mat, new_img[i][j]) 
                    
    cv.imshow('final', new_img)  
    cv.waitKey(0)
    
    unique_labels = set(label_mat.values()) - {0}  # Remove background (0)
    print(f"Number of objects detected: {len(unique_labels)}")
    print("Unique labels after correction:", unique_labels)


train_img = [] 
histograms = [] 
for root, _, files in os.walk("E:/6th Semester/DIP/Lab/Assignment 1/train-20250217T120950Z-001/train/images"):
        for file in files:
            if file.endswith(".bmp"):
                filePath = os.path.join(root, file)
                img1 = cv.imread(filePath, 0)
                #cv.imshow('Image ', img1)
                #cv.waitKey(0)
                train_img.append(img1)  
                hist = cv.calcHist([img1], [0], None, [256], [0, 256])
                hist = hist/hist.max()
                histograms.append((file, hist))
                
print(f"Loaded {len(train_img)} images successfully.")

"""plt.figure(figsize=(10, 6))

for filename, hist in histograms:
    plt.plot(hist, label=filename)

plt.title("Grayscale Histograms of Images")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Normalized Frequency")
plt.legend(loc="upper left", fontsize="small")
plt.grid(True)
plt.show()"""

img1 = cv.imread("E:/6th Semester/DIP/Lab/Assignment 1/train-20250217T120950Z-001/train/images/003.bmp", 0)
cv.imshow('Image 1', img1)
cv.waitKey(0)
#newimg1 = np.where(img1 > 128, 255, 0)
cca(img1)

# bgs original are white, i need to replace with black, nucleus is darkest and need to replace with white, cytoplasm is gray, need to replace with 255/2
# 0-75 -> nucleus
# 75-150 -> cytoplasm
# >150 -> bg  

"""import os
import cv2 as cv
import numpy as np

def negative_transform(image, max_val):
    return (max_val - 1) - image

def connected_components_8(img, target_vals):
    img_padded = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=1)
    rows, cols = img_padded.shape
    labels = np.zeros((rows, cols), dtype=np.uint8)
    label_dict = {}
    next_label = 1

    def is_target(val):
        return val in target_vals
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if is_target(img_padded[i, j]):
                neighbors = [labels[i - 1, j], labels[i, j - 1], labels[i - 1, j - 1], labels[i - 1, j + 1]]
                neighbors = [n for n in neighbors if n > 0]
                
                if not neighbors:
                    labels[i, j] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    labels[i, j] = min_label
                    for n in neighbors:
                        if n != min_label:
                            label_dict[n] = min_label
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if labels[i, j] > 0:
                current_label = labels[i, j]
                while current_label in label_dict:
                    current_label = label_dict[current_label]
                labels[i, j] = current_label
    
    labels = labels[1:-1, 1:-1]
    return labels, np.unique(labels).tolist()

def extract_high_intensity_pixels(original, thresholded):
    return np.array([original[i, j] for i in range(thresholded.shape[0]) for j in range(thresholded.shape[1]) if thresholded[i, j] >= 250])

def process_image(img):
    img_neg = negative_transform(np.copy(img), 256)
    img_mean = np.mean(img_neg)
    target_range = np.arange(int(img_mean), 255, 1)
    labeled_img, labels = connected_components_8(img_neg, target_range)
    
    histogram = np.bincount(labeled_img.ravel(), minlength=np.max(labels) + 1)
    primary_label = np.argmax(histogram[1:]) + 1
    
    img_filtered = np.where(labeled_img == primary_label, 126, 0).astype(np.uint8)
    
    thresholded_img = (img_neg > np.mean(extract_high_intensity_pixels(img_neg, img_filtered)) + 32).astype(np.uint8) * 255
    
    labeled_final, labels_final = connected_components_8(img_neg + 20, np.arange(np.mean(thresholded_img) + 32, 255, 1))
    
    final_histogram = np.bincount(labeled_final.ravel(), minlength=np.max(labels_final) + 1)
    for label in range(len(final_histogram)):
        if final_histogram[label] < 400:
            labeled_final[labeled_final == label] = 0
    
    result = np.where(img_filtered == 0, 0, np.where(labeled_final > 0, 255, 128)).astype(np.uint8)
    return result

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".bmp"):
            img = cv.imread(os.path.join(input_dir, filename), 0)
            processed_img = process_image(img)
            cv.imwrite(os.path.join(output_dir, filename), processed_img)

def compute_dice_score(gt_dir, pred_dir):
    dice_scores = {"background": [], "nucleus": [], "cytoplasm": []}
    
    for filename in os.listdir(pred_dir):
        if filename.endswith(".bmp"):
            pred_img = cv.imread(os.path.join(pred_dir, filename), 0)
            gt_img = cv.imread(os.path.join(gt_dir, filename.replace(".bmp", ".png")), 0)
            if gt_img is None or pred_img is None:
                continue
            
            for label, name in zip([0, 128, 255], ["background", "cytoplasm", "nucleus"]):
                gt_mask = (gt_img == label).astype(np.uint8)
                pred_mask = (pred_img == label).astype(np.uint8)
                intersection = np.sum(gt_mask * pred_mask)
                union = np.sum(gt_mask) + np.sum(pred_mask)
                if union > 0:
                    dice_scores[name].append((2.0 * intersection) / union)
    
    return {name: np.nanmean(scores) if scores else 0.0 for name, scores in dice_scores.items()}

# Directories setup
data_dir = "E:/6th Semester/DIP/Lab/Assignment 1/train-20250217T120950Z-001/train/images"
res_dir = "E:/6th Semester/DIP/Lab/Assignment 1/train-20250217T120950Z-001/output"
gt_dir = "E:/6th Semester/DIP/Lab/Assignment 1/train-20250217T120950Z-001/train/masks"

process_directory(data_dir, res_dir)
dice_scores = compute_dice_score(gt_dir, res_dir)
print(dice_scores) """

