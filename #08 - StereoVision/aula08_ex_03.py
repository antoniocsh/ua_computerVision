import numpy as np
import cv2
import glob

# Load calibration parameters
data = np.load("stereoParams.npz")

intrinsics1 = data["intrinsics1"]
distortion1 = data["distortion1"]
intrinsics2 = data["intrinsics2"]
distortion2 = data["distortion2"]

print("Loaded calibration parameters.")

# Load one stereo pair
left_images = sorted(glob.glob('../images/left*.jpg'))
right_images = sorted(glob.glob('../images/right*.jpg'))

# Pick one pair 
imgL = cv2.imread(left_images[0])
imgR = cv2.imread(right_images[0])

# Undistort images
undistorted_L = cv2.undistort(imgL, intrinsics1, distortion1)
undistorted_R = cv2.undistort(imgR, intrinsics2, distortion2)

# Show results
cv2.imshow("Original Left", imgL)
cv2.imshow("Undistorted Left", undistorted_L)

cv2.imshow("Original Right", imgR)
cv2.imshow("Undistorted Right", undistorted_R)

cv2.waitKey(0)
cv2.destroyAllWindows()