import sys
import numpy as np
import cv2

def printImageFeatures(image):
    # Image characteristics
    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    # print some features
    print("Image Height: %d" % height)
    print("Image Width: %d" % width)
    print("Image channels: %d" % nchannels)
    print("Number of elements : %d" % image.size)


# Read the image from argv
image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "./lena.jpg", cv2.IMREAD_GRAYSCALE );

if np.shape(image) == ():
    print("Image file could not be open!")
    exit(-1)

printImageFeatures(image)

try:
    iterations = int(sys.argv[2])
except:
    iterations = 1

cv2.imshow("Original", image)

# Average Filters (for comparison)
avg_3x3 = image.copy()
for i in range(iterations):
    avg_3x3 = cv2.blur(avg_3x3, (3,3))
cv2.imshow(f"Average Filter 3x3 - {iterations} Iter", avg_3x3)

avg_5x5 = image.copy()
for i in range(iterations):
    avg_5x5 = cv2.blur(avg_5x5, (5,5))
cv2.imshow(f"Average Filter 5x5 - {iterations} Iter", avg_5x5)

avg_7x7 = image.copy()
for i in range(iterations):
    avg_7x7 = cv2.blur(avg_7x7, (7,7))
cv2.imshow(f"Average Filter 7x7 - {iterations} Iter", avg_7x7)


# Median Filters
median_3x3 = image.copy()
for i in range(iterations):
    median_3x3 = cv2.medianBlur(median_3x3, 3)
cv2.imshow(f"Median Filter 3x3 - {iterations} Iter", median_3x3)

median_5x5 = image.copy()
for i in range(iterations):
    median_5x5 = cv2.medianBlur(median_5x5, 5)
cv2.imshow(f"Median Filter 5x5 - {iterations} Iter", median_5x5)

median_7x7 = image.copy()
for i in range(iterations):
    median_7x7 = cv2.medianBlur(median_7x7, 7)
cv2.imshow(f"Median Filter 7x7 - {iterations} Iter", median_7x7)


# Gaussian Filters
gaussian_3x3 = image.copy()
for i in range(iterations):
    gaussian_3x3 = cv2.GaussianBlur(gaussian_3x3, (3,3), 0)
cv2.imshow(f"Gaussian Filter 3x3 - {iterations} Iter", gaussian_3x3)

gaussian_5x5 = image.copy()
for i in range(iterations):
    gaussian_5x5 = cv2.GaussianBlur(gaussian_5x5, (5,5), 0)
cv2.imshow(f"Gaussian Filter 5x5 - {iterations} Iter", gaussian_5x5)

gaussian_7x7 = image.copy()
for i in range(iterations):
    gaussian_7x7 = cv2.GaussianBlur(gaussian_7x7, (7,7), 0)
cv2.imshow(f"Gaussian Filter 7x7 - {iterations} Iter", gaussian_7x7)


cv2.waitKey(0)
cv2.destroyAllWindows()