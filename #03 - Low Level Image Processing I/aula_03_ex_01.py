# Aula_02_ex_03.py
#
# Historam visualization with openCV
#
# Paulo Dias

#import
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

SPACING = 20
# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
# image = cv2.imread( "../images/lena.jpg", cv2.IMREAD_UNCHANGED );
image = cv2.imread( "filme.jpeg", cv2.IMREAD_UNCHANGED );
height, width = image.shape[0], image.shape[1]

is_grayscale = image.ndim == 2

if is_grayscale:
	color = (255)
else:
	color = (120, 120, 120)

image2 = image.copy()

if is_grayscale:
	for i in range(0, height, SPACING):
		cv2.line(image2, (0, i), (width-1, i), color, 1)
	for j in range(0, width, SPACING):
		cv2.line(image2, (j, 0), (j, height-1), color, 1)
else:
	for i in range(0, height, SPACING):
		cv2.line(image2, (0, i), (width-1, i), color, 1)
	for j in range(0, width, SPACING):
		cv2.line(image2, (j, 0), (j, height-1), color, 1)

# cv2.imwrite('filme_grid.jpeg', image2)

cv2.imshow('Image', image)
cv2.imshow('Image with grid', image2)


cv2.waitKey(0)

