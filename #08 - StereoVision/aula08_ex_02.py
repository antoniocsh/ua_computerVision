import numpy as np
import cv2
import glob

# Chessboard configuration
board_h = 9
board_w = 6

# Prepare object points (3D world coordinates)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)


objPoints = []        # 3D points
left_corners = []     # 2D points (left images)
right_corners = []    # 2D points (right images)

# Corner detection function
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

assert len(left_images) == len(right_images), "Mismatch in stereo pairs"

print("Total pairs found:", len(left_images))

# Detect corners
for left_fname, right_fname in zip(left_images, right_images):

    imgL = cv2.imread(left_fname)
    imgR = cv2.imread(right_fname)

    retL, cornersL = FindCorners(imgL)
    retR, cornersR = FindCorners(imgR)

    # Only keep valid stereo detections
    if retL and retR:
        objPoints.append(objp)
        left_corners.append(cornersL)
        right_corners.append(cornersR)

cv2.destroyAllWindows()

# Convert to numpy arrays
objPoints = np.array(objPoints, dtype=np.float32)
left_corners = np.array(left_corners, dtype=np.float32)
right_corners = np.array(right_corners, dtype=np.float32)

# Fix shape (remove extra dimension)
left_corners = np.squeeze(left_corners)
right_corners = np.squeeze(right_corners)

print("\nCollected data:")
print("objPoints:", objPoints.shape)
print("left_corners:", left_corners.shape)
print("right_corners:", right_corners.shape)

# Stereo Calibration

# Image size (width, height)
image_size = cv2.imread(left_images[0]).shape[:2][::-1]

# Initial camera parameters (no guess)
intrinsics1 = np.eye(3, dtype=np.float64)
intrinsics2 = np.eye(3, dtype=np.float64)

distortion1 = np.zeros((5, 1))
distortion2 = np.zeros((5, 1))

# Termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

# Flag: same focal length
flags = cv2.CALIB_SAME_FOCAL_LENGTH

print("\nRunning stereo calibration...")

ret, intrinsics1, distortion1, intrinsics2, distortion2, R, T, E, F = cv2.stereoCalibrate(
    objPoints,
    left_corners,
    right_corners,
    intrinsics1,
    distortion1,
    intrinsics2,
    distortion2,
    image_size,
    criteria=criteria,
    flags=flags
)

# Results
print("\nCalibration completed!")
print("RMS error:", ret)

print("\nIntrinsic Matrix 1:\n", intrinsics1)
print("\nIntrinsic Matrix 2:\n", intrinsics2)

print("\nRotation (R):\n", R)
print("\nTranslation (T):\n", T)

# Save parameters
# np.savez("stereoParams.npz",
#          intrinsics1=intrinsics1,
#          distortion1=distortion1,
#          intrinsics2=intrinsics2,
#          distortion2=distortion2,
#          R=R, T=T, E=E, F=F)

print("\nParameters saved to stereoParams.npz")