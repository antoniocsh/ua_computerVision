import sys
import numpy as np
import cv2

image = cv2.imread( "../images/lena.jpg", cv2.IMREAD_GRAYSCALE );

if  np.shape(image) == ():
    # Failed Reading
    print("Image file could not be open!")
    exit(-1)

cv2.imshow('Orginal', image)

#thresh_binary
_, imageThreshBinary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow( "Threshold Binary", imageThreshBinary )

#thresh_binary_inv
_, imageThreshBinaryInv = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow( "Threshold Binary Inverted", imageThreshBinaryInv )

#thresh_trunc
_, imageThreshTrunc = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow( "Threshold Truncated", imageThreshTrunc )

#thresh_tozero
_, imageThreshToZero = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow( "Threshold To Zero", imageThreshToZero )

#thresh_tozero_inv
_, imageThreshToZeroInv = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow( "Threshold To Zero Inverted", imageThreshToZeroInv )

cv2.waitKey(0)
