 # Aula_01_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys


image1 = cv2.imread( "../images/deti.jpg", cv2.IMREAD_UNCHANGED )
image2 = cv2.imread( "../images/deti.bmp", cv2.IMREAD_UNCHANGED )

image3 = image1 - image2
image4 = cv2.subtract(image1, image2)


# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "Display window 2", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "Display window 3", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "Display window 4", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image1)
cv2.imshow( "Display window 2", image2)
cv2.imshow( "Display window 3", image3)
cv2.imshow( "Display window 4", image4)


cv2.imwrite("saved_image.jpg", image3)


# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
