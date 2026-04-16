import numpy as np
import cv2

board_w = 6
board_h = 9

# LOAD CAMERA PARAMETERS
with np.load('camera.npz') as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']

print("Intrinsics:")
print(intrinsics)

print("Distortion:")
print(distortion)

# LOAD IMAGE
img = cv2.imread('..//images//left01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FIND CHESSBOARD
ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

if not ret:
    print("Chessboard not found!")
    exit()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

# =========================
# 3D OBJECT POINTS
# =========================
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# SOLVE PNP
ret, rvec, tvec = cv2.solvePnP(
    objp,
    corners,
    intrinsics,
    distortion
)

print("\nRotation Vector (rvec):")
print(rvec)

print("\nTranslation Vector (tvec):")
print(tvec)

# # DRAW AXES
# axis = np.float32([
#     [0, 0, 0],
#     [3, 0, 0],
#     [0, 3, 0],
#     [0, 0, -3]
# ])

# imgpts, _ = cv2.projectPoints(axis, rvec, tvec, intrinsics, distortion)
# imgpts = np.int32(imgpts).reshape(-1, 2)

# origin = tuple(imgpts[0])
# # bgr 
# img = cv2.line(img, origin, tuple(imgpts[1]), (255, 0, 0), 3)
# img = cv2.line(img, origin, tuple(imgpts[2]), (0, 255, 0), 3)
# img = cv2.line(img, origin, tuple(imgpts[3]), (0, 0, 255), 3)

cv2.imshow("Pose Estimation", img)
cv2.waitKey(0)
cv2.destroyAllWindows()