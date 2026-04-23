import numpy as np
import cv2
import glob

# Load calibration parameters
data = np.load("stereoParams.npz")

K1 = data["intrinsics1"]
D1 = data["distortion1"]
K2 = data["intrinsics2"]
D2 = data["distortion2"]
F  = data["F"]

print("Calibration loaded.")

# Load one stereo pair
left_images = sorted(glob.glob('../images/left*.jpg'))
right_images = sorted(glob.glob('../images/right*.jpg'))

imgL = cv2.imread(left_images[0])
imgR = cv2.imread(right_images[0])

# Undistort images
imgL = cv2.undistort(imgL, K1, D1)
imgR = cv2.undistort(imgR, K2, D2)

# Copies for drawing
imgL_draw = imgL.copy()
imgR_draw = imgR.copy()

# Mouse callbacks + Draw
def draw_line(img, line, color):
    a, b, c = line
    h, w = img.shape[:2]

    if abs(b) > 1e-6:
        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(c + a * w) / b)
    else:
        # vertical line case
        x0 = x1 = int(-c / a)
        y0, y1 = 0, h

    cv2.line(img, (x0, y0), (x1, y1), color, 2)

def mouse_left(event, x, y, flags, param):
    global imgL_draw, imgR_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left click: ({x}, {y})")

        p = np.asarray([x, y], dtype=np.float32)
        color = np.random.randint(0, 255, 3).tolist()

        # Compute epiline in RIGHT image
        lineR = cv2.computeCorrespondEpilines(p.reshape(-1,1,2), 1, F)
        lineR = lineR.reshape(-1,3)[0]

        imgL_draw = imgL.copy()
        imgR_draw = imgR.copy()

        # Draw point in LEFT image
        cv2.circle(imgL_draw, (x, y), 5, color, -1)

        # Draw corresponding line ONLY in RIGHT image
        draw_line(imgR_draw, lineR, color)

        cv2.imshow("Left Image", imgL_draw)
        cv2.imshow("Right Image", imgR_draw)


def mouse_right(event, x, y, flags, param):
    global imgL_draw, imgR_draw

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Right click: ({x}, {y})")

        p = np.asarray([x, y], dtype=np.float32)
        color = np.random.randint(0, 255, 3).tolist()

        # Compute epiline in LEFT image
        lineL = cv2.computeCorrespondEpilines(p.reshape(-1,1,2), 2, F)
        lineL = lineL.reshape(-1,3)[0]

        imgL_draw = imgL.copy()
        imgR_draw = imgR.copy()

        # Draw point in RIGHT image
        cv2.circle(imgR_draw, (x, y), 5, color, -1)

        # Draw corresponding line ONLY in LEFT image
        draw_line(imgL_draw, lineL, color)

        cv2.imshow("Left Image", imgL_draw)
        cv2.imshow("Right Image", imgR_draw)


# Show images and set callbacks
cv2.imshow("Left Image", imgL_draw)
cv2.imshow("Right Image", imgR_draw)

cv2.setMouseCallback("Left Image", mouse_left)
cv2.setMouseCallback("Right Image", mouse_right)

print("\nClick on either image to see epipolar lines.")

cv2.waitKey(-1)
cv2.destroyAllWindows()