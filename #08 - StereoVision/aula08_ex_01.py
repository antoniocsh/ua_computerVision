import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# Prepare object points (3D points in real world)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Storage for stereo calibration
objPoints = []        # 3D points
left_corners = []    # 2D points in left images
right_corners = []   # 2D points in right images

def FindCorners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret:
        cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(200)

    return ret, corners


# Load stereo image pairs
left_images = sorted(glob.glob('../images/left*.jpg'))
right_images = sorted(glob.glob('../images/right*.jpg'))

# Ensure same number of images
assert len(left_images) == len(right_images), "Mismatch in stereo pairs"

for left_fname, right_fname in zip(left_images, right_images):

    imgL = cv2.imread(left_fname)
    imgR = cv2.imread(right_fname)

    retL, cornersL = FindCorners(imgL)
    retR, cornersR = FindCorners(imgR)

    # Only use pair if BOTH succeed
    if retL and retR:
        objPoints.append(objp)
        left_corners.append(cornersL)
        right_corners.append(cornersR)

cv2.destroyAllWindows()

# Convert to numpy arrays (optional but useful)
objPoints = np.array(objPoints, dtype=np.float32)
left_corners = np.array(left_corners, dtype=np.float32)
right_corners = np.array(right_corners, dtype=np.float32)

print("Collected data:")
print("objPoints:", objPoints.shape)
print("left_corners:", left_corners.shape)
print("right_corners:", right_corners.shape)