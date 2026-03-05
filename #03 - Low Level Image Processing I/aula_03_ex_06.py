import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# Read color image
image = cv2.imread("../images/Fruits-RGB.tif", cv2.IMREAD_COLOR)

if np.shape(image) == ():
    print("Image file could not be open!")
    exit(-1)

# Show original image
cv2.imshow("Original Image", image)

# Split channels
b, g, r = cv2.split(image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram parameters
histSize = 256
histRange = [0,256]

# Compute histograms
hist_b = cv2.calcHist([b],[0],None,[histSize],histRange)
hist_g = cv2.calcHist([g],[0],None,[histSize],histRange)
hist_r = cv2.calcHist([r],[0],None,[histSize],histRange)
hist_gray = cv2.calcHist([gray],[0],None,[histSize],histRange)

# Function to draw histogram using OpenCV
def draw_histogram(hist, window_name):

    histImageWidth = 512
    histImageHeight = 512

    histImage = np.zeros((histImageWidth,histImageHeight,1), np.uint8)

    binWidth = int(np.ceil(histImageWidth*1.0/histSize))

    cv2.normalize(hist, hist, 0, histImageHeight, cv2.NORM_MINMAX)

    for i in range(histSize):
        cv2.rectangle(histImage,
                     (i*binWidth,0),
                     ((i+1)*binWidth,int(hist[i])),
                     (125),
                     -1)

    histImage = np.flipud(histImage)

    cv2.imshow(window_name, histImage)


# Draw histograms
draw_histogram(hist_b, "Blue Histogram")
draw_histogram(hist_g, "Green Histogram")
draw_histogram(hist_r, "Red Histogram")
draw_histogram(hist_gray, "Gray Histogram")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Matplotlib visualization
plt.plot(hist_b, 'b', label="Blue")
plt.plot(hist_g, 'g', label="Green")
plt.plot(hist_r, 'r', label="Red")
plt.plot(hist_gray, 'k', label="Gray")

plt.xlim(histRange)
plt.legend()
plt.show()