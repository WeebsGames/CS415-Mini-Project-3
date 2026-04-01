import os
import glob #used to parse through the training images folder
import numpy as np
import cv2
import math


def calculate_hs_histogram(img, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_h = 179
    max_s = 255
    hs_hist = np.zeros((math.ceil((max_h + 1) / bin_size), math.ceil((max_s + 1) / bin_size)))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h / bin_size), math.floor(s / bin_size)] += 1
    if hs_hist.sum() > 0:
        hs_hist /= hs_hist.sum()
    return hs_hist


def build_training_histogram(training_dir, bin_size):
    image_paths = sorted(glob.glob(os.path.join(training_dir, "*")))
    if not image_paths:
        raise FileNotFoundError(f"No training images found in {training_dir}")

    combined_hist = None
    valid_images = 0
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        hist = calculate_hs_histogram(img, bin_size)
        if combined_hist is None:
            combined_hist = hist
        else:
            combined_hist += hist
        valid_images += 1

    if valid_images == 0:
        raise ValueError(f"No readable images found in {training_dir}")

    combined_hist /= valid_images
    return combined_hist


def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h / bin_size), math.floor(s / bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask


def collect_training_hs(training_dir):
    image_paths = sorted(glob.glob(os.path.join(training_dir, "*")))
    if not image_paths:
        raise FileNotFoundError(f"No training images found in {training_dir}")

    all_samples = []
    for path in image_paths:
        if not path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hs = hsv[:, :, :2].reshape(-1, 2).astype(np.float64)
        all_samples.append(hs)

    if not all_samples:
        raise ValueError(f"No readable training images found in {training_dir}")

    return np.vstack(all_samples)


def estimate_gaussian_model(training_dir):
    samples = collect_training_hs(training_dir)
    mean_vector = np.mean(samples, axis=0)
    covariance_matrix = np.cov(samples, rowvar=False)
    return mean_vector, covariance_matrix


def gaussian_skin_segmentation(img, mean_vector, covariance_matrix, mahalanobis_threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hs = hsv[:, :, :2].reshape(-1, 2).astype(np.float64)
    cov_reg = covariance_matrix + np.eye(2) * 1e-6
    inv_cov = np.linalg.inv(cov_reg)
    delta = hs - mean_vector
    mahalanobis_sq = np.sum(delta @ inv_cov * delta, axis=1)
    mask = (mahalanobis_sq <= mahalanobis_threshold).astype(np.uint8)
    return mask.reshape(height, width, 1), mahalanobis_sq.reshape(height, width)


# Training
training_dir = "training_images"
bin_size = 20
hs_hist = build_training_histogram(training_dir, bin_size)

# Testing
img_test = cv2.imread("testing_image.bmp")
if img_test is None:
    raise FileNotFoundError("testing_image.bmp not found or could not be read")

threshold = 0.03
mask = color_segmentation(img_test, hs_hist, bin_size, threshold)

img_seg = img_test * mask

mean_vector, covariance_matrix = estimate_gaussian_model(training_dir)
print("Gaussian mean vector (H, S):", mean_vector)
print("Gaussian covariance matrix:")
print(covariance_matrix)

mahalanobis_threshold = 16.0
gauss_mask, _ = gaussian_skin_segmentation(img_test, mean_vector, covariance_matrix, mahalanobis_threshold)
img_seg_gaussian = img_test * gauss_mask

cv2.imshow("Input", img_test)
cv2.imshow("Mask", (mask * 255).astype(np.uint8))
cv2.imshow("Segmentation", img_seg.astype(np.uint8))
cv2.imshow("Gaussian Mask", (gauss_mask * 255).astype(np.uint8))
cv2.imshow("Gaussian Segmentation", img_seg_gaussian.astype(np.uint8))
cv2.waitKey()

def harris_corner(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


harris_corner("checkerboard-1.png")
harris_corner("toy-1.png")