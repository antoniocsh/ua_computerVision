 # Aula_01_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys


image100 = cv2.imread( "quality_100.jpg", cv2.IMREAD_UNCHANGED )
image70 = cv2.imread( "quality_70.jpg", cv2.IMREAD_UNCHANGED )
image40 = cv2.imread( "quality_40.jpg", cv2.IMREAD_UNCHANGED )
image10 = cv2.imread( "quality_10.jpg", cv2.IMREAD_UNCHANGED )

diff70 = cv2.subtract(image100, image70)
diff40 = cv2.subtract(image100, image40)
diff10 = cv2.subtract(image100, image10)

# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Original", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "70% quality", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "40% quality", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "10% quality", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "diff 70", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "diff 40", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "diff 10", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Original", image100)
cv2.imshow( "70% quality", image70)
cv2.imshow( "40% quality", image40)
cv2.imshow( "10% quality", image10)
cv2.imshow( "diff 70", diff70)
cv2.imshow( "diff 40", diff40)
cv2.imshow( "diff 10", diff10)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
