import sys
import numpy as np
import cv2

def printImageFeatures(image):

    if len(image.shape) == 2:
        height, width = image.shape
        nchannels = 1
    else:
        height, width, nchannels = image.shape

    print("Image Height:", height)
    print("Image Width:", width)
    print("Image channels:", nchannels)
    print("Number of elements:", image.size)


# Read image
image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
#image = cv2.imread( "./lena.jpg", cv2.IMREAD_GRAYSCALE );

if np.shape(image) == ():
    print("Image file could not be open!")
    exit(-1)

printImageFeatures(image)

cv2.imshow("Original", image)


# Canny Detector - different thresholds
edges1 = cv2.Canny(image, 1, 255)
cv2.imshow("Canny (1 , 255)", edges1)

edges2 = cv2.Canny(image, 220, 225)
cv2.imshow("Canny (220 , 225)", edges2)

edges3 = cv2.Canny(image, 1, 128)
cv2.imshow("Canny (1 , 128)", edges3)


cv2.waitKey(0)
cv2.destroyAllWindows()