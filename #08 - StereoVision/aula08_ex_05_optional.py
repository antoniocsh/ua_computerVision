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

# Rectification setup
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
    alpha=-1
)

map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

# Copies for drawing
dispL = rectL.copy()
dispR = rectR.copy()

# Draw horizontal highlight line
def draw_row(img, y, color):
    h, w = img.shape[:2]
    cv2.line(img, (0, y), (w, y), color, 2)

# Mouse callbacks
def mouse_left(event, x, y, flags, param):
    global dispL, dispR

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click: ({x}, {y})")

        color = np.random.randint(0, 255, 3).tolist()

        dispL = rectL.copy()
        dispR = rectR.copy()

        # Highlight clicked row in LEFT
        draw_row(dispL, y, color)

        # Same row in RIGHT (rectified => same y)
        draw_row(dispR, y, color)

        # Mark point
        cv2.circle(dispL, (x, y), 5, color, -1)

        cv2.imshow("Left Rectified", dispL)
        cv2.imshow("Right Rectified", dispR)


def mouse_right(event, x, y, flags, param):
    global dispL, dispR

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Right click: ({x}, {y})")

        color = np.random.randint(0, 255, 3).tolist()

        dispL = rectL.copy()
        dispR = rectR.copy()

        # Highlight clicked row in RIGHT
        draw_row(dispR, y, color)

        # Same row in LEFT
        draw_row(dispL, y, color)

        cv2.circle(dispR, (x, y), 5, color, -1)

        cv2.imshow("Left Rectified", dispL)
        cv2.imshow("Right Rectified", dispR)

# Show images + callbacks
cv2.imshow("Left Rectified", rectL)
cv2.imshow("Right Rectified", rectR)

cv2.setMouseCallback("Left Rectified", mouse_left)
cv2.setMouseCallback("Right Rectified", mouse_right)

print("Click on either image to highlight corresponding row.")

cv2.waitKey(-1)
cv2.destroyAllWindows()