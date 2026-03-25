import numpy as np
import cv2
import math

src = cv2.imread("image.jpeg")
dst = cv2.imread("image_tf.jpg")

# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(src,None)
src = cv2.drawKeypoints(src, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp2, des2 = sift.detectAndCompute(dst,None)
dst = cv2.drawKeypoints(dst, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT Keypoints Og', src)
cv2.imshow('SIFT Keypoints Rotated', dst)
# cv2.imwrite("image_sift.jpg", src)
cv2.waitKey(0)
cv2.destroyAllWindows()