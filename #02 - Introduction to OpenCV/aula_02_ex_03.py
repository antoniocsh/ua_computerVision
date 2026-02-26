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

cv2.namedWindow("Window")

def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:   
        cv2.circle(image1, center=(x, y), radius=50, color=(0, 255, 0), thickness=-1)
        cv2.imshow("Window", image1) 

cv2.setMouseCallback("Window", mouse_handler)

cv2.imshow("Window", image1)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )