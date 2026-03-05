# Histogram Equalization Example
# Paulo Dias

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

image = cv2.imread("../images/TAC_PULMAO.bmp", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image!")
    exit(-1)

# Histogram Equalization
equalized = cv2.equalizeHist(image)

# Calc Histograms
hist_original = cv2.calcHist([image], [0], None, [256], [0,256])
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0,256])

# Show Images
cv2.imshow("Original Image", image)
cv2.imshow("Equalized Image", equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot Histograms
plt.figure("Histograms")

plt.subplot(2,1,1)
plt.title("Original Histogram")
plt.plot(hist_original)
plt.xlim([0,256])

plt.subplot(2,1,2)
plt.title("Equalized Histogram")
plt.plot(hist_equalized)
plt.xlim([0,256])

plt.show()