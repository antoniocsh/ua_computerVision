import sys
import cv2
import numpy as np


def printImageFeatures(image):

    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    print("Image Height:", height)
    print("Image Width:", width)
    print("Image Channels:", nchannels)
    print("Number of Elements:", image.size)


# Read image
image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

if np.shape(image) == ():
    print("Image file could not be opened!")
    exit(-1)

printImageFeatures(image)

# Canny edge detection
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Convert ORIGINAL image to color so we can draw red lines
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
image_prob = image_color.copy()

# Standard Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

if lines is not None:
    for rho, theta in lines[:,0]:

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

        cv2.line(image_color, pt1, pt2, (0,0,255), 2)


# Probabilistic Hough Transform
linesP = cv2.HoughLinesP(edges,
                         1,
                         np.pi/180,
                         threshold=80,
                         minLineLength=50,
                         maxLineGap=10)

if linesP is not None:
    for x1, y1, x2, y2 in linesP[:,0]:
        cv2.line(image_prob, (x1,y1), (x2,y2), (0,0,255), 2)


# Show results
cv2.imshow("Original Image", image)
cv2.imshow("Detected Lines - Standard Hough", image_color)
cv2.imshow("Detected Lines - Probabilistic Hough", image_prob)

cv2.waitKey(0)
cv2.destroyAllWindows()