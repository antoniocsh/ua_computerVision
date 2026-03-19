import sys
import numpy as np
import cv2

# image = cv2.imread( sys.argv[1] , cv2.IMREAD_GRAYSCALE );
image = cv2.imread("image.jpeg" , cv2.IMREAD_COLOR );

rows,cols,channels  = image.shape
M = cv2.getRotationMatrix2D((0,0),25,1)
print(M)
M[0][2] = -50
M[1][2] = 100
print(M) 

cv2.imshow('Orginal', image)

dst = cv2.warpAffine(image,M,(cols,rows))
 
cv2.imshow('img',dst)
# cv2.imwrite("image_tf.jpg", dst)

cv2.waitKey(0)
