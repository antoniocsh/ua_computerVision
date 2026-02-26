 # Aula_01_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys


image1 = cv2.imread( "quality_100.jpg", cv2.IMREAD_UNCHANGED )

# OpenCV reads images in BGR format, not RGB.

gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
hls_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HLS)
xyz_image = cv2.cvtColor(image1, cv2.COLOR_BGR2XYZ)
hsv_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
ycrcb_image = cv2.cvtColor(image1, cv2.COLOR_BGR2YCrCb)

cv2.imshow("Original", image1)
cv2.imshow("Gray Scale", gray_image)
cv2.imshow("HLS", hls_image)
cv2.imshow("XYZ", xyz_image)
cv2.imshow("HSV", hsv_image)
cv2.imshow("YCrCb", ycrcb_image)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )