 # Aula_01_ex_01.py
 #
 # Cheesboard Calibration
 #
 # Paulo Dias

import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def FindAndDisplayChessboard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret == True:
        # refinement criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # refine corners
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),   # search window
            (-1, -1),   # zero zone
            criteria
        )

        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

    return ret, corners


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w*board_h,3), np.float32)
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Read images
images = sorted(glob.glob('..//images//left*.jpg'))
img_shape = None
corners=[]
for fname in images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        if img_shape is None:
            img_shape = img.shape[:2]
print("corners: ", corners)

ret2, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape[::-1], None, None)

print("Intrinsics:")
print(intrinsics)

print("Distortion:")
print(distortion)

for i in range(len(tvecs)):
    print("Translations(%d):" % i)
    print(tvecs[i])
    print("Rotation(%d):" % i)
    print(rvecs[i])

# np.savez('camera.npz', intrinsics=intrinsics, distortion=distortion)

cv2.waitKey(-1)
cv2.destroyAllWindows()