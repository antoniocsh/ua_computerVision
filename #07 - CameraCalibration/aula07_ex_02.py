# Aula_01_ex_01.py
# Chessboard Calibration + 3D Projection

import numpy as np
import cv2
import glob

# Chessboard size (internal corners)
board_h = 9
board_w = 6

# termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

def FindAndDisplayChessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret:
        # refine corners to subpixel accuracy
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)

    return ret, corners


# Prepare object points (3D world points)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Load images
images = sorted(glob.glob('..//images//left*.jpg'))

if len(images) == 0:
    print("No images found!")
    exit()

img_shape = None

# Detect corners
for fname in images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)

    if ret:
        objpoints.append(objp.copy())
        imgpoints.append(corners)

        if img_shape is None:
            img_shape = img.shape[:2]

if len(objpoints) == 0:
    print("No valid chessboard detections!")
    exit()

# Calibration
ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_shape[::-1],
    None,
    None
)

# Print results
print("Intrinsics:")
print(intrinsics)

print("Distortion:")
print(distortion)

for i in range(len(tvecs)):
    print(f"Translation {i}:")
    print(tvecs[i])
    print(f"Rotation {i}:")
    print(rvecs[i])

# Save calibration
# np.savez('camera.npz', intrinsics=intrinsics, distortion=distortion)

# 3D CUBE PROJECTION
img = cv2.imread(images[0])

# Define cube (size = 3 units)
cube = np.float32([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, -1],
    [1, 1, -1],
    [0, 1, -1]
])

# Project cube points into image
imgpts, _ = cv2.projectPoints(
    cube,
    rvecs[0],
    tvecs[0],
    intrinsics,
    distortion
)

imgpts = np.int32(imgpts).reshape(-1, 2)

# Draw cube base
img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 0, 255), 2)

# Draw cube top
img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 2)

# Draw vertical edges
for i in range(4):
    img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 0, 255), 2)

cv2.imshow("3D Cube", img)
cv2.waitKey(0)
cv2.destroyAllWindows()