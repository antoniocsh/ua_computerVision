import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

image = cv2.imread("../images/deti.bmp", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("../images/input.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image could not be loaded!")
    exit(-1)

def contrast_stretch(img):

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img)

    print("Min intensity:", minVal)
    print("Max intensity:", maxVal)

    # apply formula
    final = ((img - minVal) / (maxVal - minVal)) * 255

    # convert to uint8 pq ya
    final = final.astype(np.uint8)

    return final


# Apply contrast stretching
result1 = contrast_stretch(image)
result2 = contrast_stretch(image2)

# Histograms
hist_original1 = cv2.calcHist([image],[0],None,[256],[0,256])
hist_result1 = cv2.calcHist([result1],[0],None,[256],[0,256])

hist_original2 = cv2.calcHist([image2],[0],None,[256],[0,256])
hist_result2 = cv2.calcHist([result2],[0],None,[256],[0,256])

# Show images
cv2.imshow("Original DETI", image)
cv2.imshow("Stretched DETI", result1)

cv2.imshow("Original input", image2)
cv2.imshow("Stretched input", result2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot histograms
plt.figure("DETI Histogram")
plt.plot(hist_original1, 'b', label="Original")
plt.plot(hist_result1, 'r', label="Stretched")
plt.legend()

plt.figure("Input Histogram")
plt.plot(hist_original2, 'b', label="Original")
plt.plot(hist_result2, 'r', label="Stretched")
plt.legend()

plt.show()