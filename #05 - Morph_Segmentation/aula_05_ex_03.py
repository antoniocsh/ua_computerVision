import cv2 
import numpy as np
import sys


image = cv2.imread("../images/mon1.bmp", cv2.IMREAD_GRAYSCALE );

img_copy = image.copy()
_, img_copy = cv2.threshold(img_copy, 90, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
img_2 = img_copy.copy()
img_copy = cv2.erode(img_copy, kernel, iterations=2)
img_2 = cv2.erode(img_2, kernel2, iterations=2)

cv2.imshow("original", image)
cv2.imshow( "erode ellipse 11x11", img_copy)
cv2.imshow( "erode square 9x9", img_2)    

cv2.waitKey(0)
cv2.destroyAllWindows()