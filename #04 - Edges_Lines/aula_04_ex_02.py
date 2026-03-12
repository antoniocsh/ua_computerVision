import sys
import numpy as np
import cv2

def printImageFeatures(image):
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)

# Read the image from argv
image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
#image = cv2.imread( "./lena.jpg", cv2.IMREAD_GRAYSCALE );

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)
try:
	iterations = int(sys.argv[2])
except:
	iterations = 3

cv2.imshow('Orginal', image)

# Average filter 3 x 3
image_3x3 = image.copy()
for i in range(0, iterations):
	image_3x3 = cv2.blur( image_3x3, (3, 3))
cv2.namedWindow( f"Average Filter 3 x 3 - {iterations} Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( f"Average Filter 3 x 3 - {iterations} Iter", image_3x3 )

image_5x5 = image.copy()
for i in range(0, iterations):
	image_5x5 = cv2.blur( image_5x5, (5, 5))
cv2.namedWindow( f"Average Filter 5 x 5 - {iterations} Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( f"Average Filter 5 x 5 - {iterations} Iter", image_5x5 )

image_7x7 = image.copy()
for i in range(0, iterations):
	image_7x7 = cv2.blur( image_7x7, (7, 7))
cv2.namedWindow( f"Average Filter 7 x 7 - {iterations} Iter", cv2.WINDOW_AUTOSIZE )
cv2.imshow( f"Average Filter 7 x 7 - {iterations} Iter", image_7x7 )

cv2.waitKey(0)


