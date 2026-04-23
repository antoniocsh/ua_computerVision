import numpy as np
import cv2
import glob

# Load calibration
data = np.load("stereoParams.npz")

K1 = data["intrinsics1"]
D1 = data["distortion1"]
K2 = data["intrinsics2"]
D2 = data["distortion2"]
R  = data["R"]
T  = data["T"]

# Load stereo images
left_images = sorted(glob.glob('../images/left*.jpg'))
right_images = sorted(glob.glob('../images/right*.jpg'))

imgL = cv2.imread(left_images[0])
imgR = cv2.imread(right_images[0])

h, w = imgL.shape[:2]

# Stereo Rectification
R1 = np.zeros((3,3))
R2 = np.zeros((3,3))
P1 = np.zeros((3,4))
P2 = np.zeros((3,4))
Q  = np.zeros((4,4))

cv2.stereoRectify(
    K1, D1,
    K2, D2,
    (w, h),
    R, T,
    R1, R2, P1, P2, Q,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# Convert to grayscale 
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

# StereoBM (Block Matching)
stereo = cv2.StereoBM_create(
    numDisparities=16 * 5,   # must be divisible by 16
    blockSize=21
)

print("Computing disparity...")

disparity = stereo.compute(grayL, grayR)

# Normalize for display
disparity = cv2.normalize(
    disparity,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX
)

disparity = np.uint8(disparity)

# Show results
cv2.imshow("Left Rectified", rectL)
cv2.imshow("Right Rectified", rectR)
cv2.imshow("Disparity Map", disparity)

cv2.waitKey(0)
cv2.destroyAllWindows()