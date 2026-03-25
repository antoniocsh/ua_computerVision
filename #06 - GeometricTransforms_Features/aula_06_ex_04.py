import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

img1 = cv2.imread("image.jpeg")
img2 = cv2.imread("image_tf.jpg")

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)


# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img4 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img5 = np.vstack((img3, img4))

cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Matches', img5)
cv2.waitKey(0)
cv2.destroyAllWindows()