 # Aula_01_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys

try:
	image = cv2.imread( sys.argv[1], cv2.IMREAD_UNCHANGED )
except:
	# Read the image
	image = cv2.imread( "../images/lena.jpg", cv2.IMREAD_UNCHANGED )
try: 
	gray_level = int(sys.argv[2])
except:
	gray_level = 128

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

# Image characteristics
print(image.shape)
height, width = image.shape[0], image.shape[1]

print("Image Size: (%d,%d)" % (height, width))
print("Image Type: %s" % (image.dtype))

image_2 = image.copy()
if image.ndim == 3:
	for i in range(0, height):
		for j in range(0, width):
			if image[i,j,0] < gray_level:
				image_2[i,j,0] = 0
			if image[i,j,1] < gray_level:
				image_2[i,j,1] = 0
			if image[i,j,2] < gray_level:
				image_2[i,j,2] = 0
else:
	for i in range(0, height):
		for j in range(0, width):
			if image[i,j] < gray_level:
				image_2[i,j] = 0

# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )
cv2.namedWindow( "Display window 2", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image)
cv2.imshow( "Display window 2", image_2)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
