import cv2 
import numpy as np
import sys


image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );

img_copy = image.copy()
# img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.threshold(img_copy, 120, 255, cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
img_2 = img_copy.copy()
img_copy = cv2.dilate(img_copy, kernel, iterations=1)
img_2 = cv2.dilate(img_2, kernel2, iterations=1)

cv2.imshow( "image", img_copy)
cv2.imshow( "image2", img_2)    

cv2.waitKey(0)
cv2.destroyAllWindows()