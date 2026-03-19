import cv2 
import numpy as np
import sys


art2_og = cv2.imread("../images/art2.bmp", cv2.IMREAD_GRAYSCALE );
art3_og = cv2.imread("../images/art3.bmp", cv2.IMREAD_GRAYSCALE ); 

art2 = art2_og.copy()
art3 = art3_og.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

art22 = art2.copy()

art3 = cv2.erode(art3, kernel, iterations=1)
art3 = cv2.dilate(art3, kernel, iterations=1)

art2 = cv2.erode(art2, kernel2, iterations=1)
art2 = cv2.dilate(art2, kernel2, iterations=1)

art22 = cv2.erode(art22, kernel3, iterations=1)
art22 = cv2.dilate(art22, kernel3, iterations=1)

cv2.imshow("original art3", art3_og)
cv2.imshow("art3", art3)
cv2.imshow("original art2", art2_og)
cv2.imshow( "art2 9x3", art2)
cv2.imshow( "art2 3x9", art22)    

cv2.waitKey(0)
cv2.destroyAllWindows()