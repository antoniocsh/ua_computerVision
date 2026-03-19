import cv2 
import numpy as np
import sys


art4_og = cv2.imread("../images/art4.bmp", cv2.IMREAD_GRAYSCALE );
art3_og = cv2.imread("../images/art3.bmp", cv2.IMREAD_GRAYSCALE ); 

art4 = art4_og.copy()
art44 = art4_og.copy()
art444 = art4_og.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (22, 22))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))

art4 = cv2.dilate(art4, kernel, iterations=1)
art4 = cv2.erode(art4, kernel, iterations=1)

art44 = cv2.dilate(art44, kernel2, iterations=1)
art44 = cv2.erode(art44, kernel2, iterations=1)

art444 = cv2.dilate(art444, kernel3, iterations=1)
art444 = cv2.erode(art444, kernel3, iterations=1)

cv2.imshow("original art4", art4_og)
cv2.imshow("close ellipse 22x22", art4)
cv2.imshow("close ellipse 11x11", art44)
cv2.imshow("close ellipse 33x33", art444)

cv2.waitKey(0)
cv2.destroyAllWindows()