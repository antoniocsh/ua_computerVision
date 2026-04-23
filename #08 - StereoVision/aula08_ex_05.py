import numpy as np
import cv2
import glob

# Load calibration parameters
data = np.load("stereoParams.npz")

K1 = data["intrinsics1"]
D1 = data["distortion1"]
K2 = data["intrinsics2"]
D2 = data["distortion2"]
R  = data["R"]
T  = data["T"]

print("Calibration loaded.")

# Load one stereo pair
left_images = sorted(glob.glob('../images/left*.jpg'))
right_images = sorted(glob.glob('../images/right*.jpg'))

imgL = cv2.imread(left_images[0])
imgR = cv2.imread(right_images[0])

height, width = imgL.shape[:2]

# Rectification
R1 = np.zeros((3,3))
R2 = np.zeros((3,3))
P1 = np.zeros((3,4))
P2 = np.zeros((3,4))
Q  = np.zeros((4,4))

cv2.stereoRectify(
    K1, D1,
    K2, D2,
    (width, height),
    R, T,
    R1, R2, P1, P2, Q,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=-1
)

# Compute maps
print("InitUndistortRectifyMap")

map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, (width, height), cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, (width, height), cv2.CV_32FC1
)

# Apply remap
rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# Draw horizontal lines
def draw_lines(img):
    h, w = img.shape[:2]
    img_copy = img.copy()

    for y in range(0, h, 25):
        cv2.line(img_copy, (0, y), (w, y), (0, 255, 0), 1)

    return img_copy

rectL_lines = draw_lines(rectL)
rectR_lines = draw_lines(rectR)

# Show results
cv2.imshow("Rectified Left", rectL_lines)
cv2.imshow("Rectified Right", rectR_lines)

cv2.waitKey(0)
cv2.destroyAllWindows()