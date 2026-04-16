import numpy as np
import cv2


# CHESSBOARD CONFIGURATION
board_w = 5   # inner corners per row
board_h = 7   # inner corners per column

square_size = 30  # mm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# OBJECT POINTSs
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# CHESSBOARD DETECTION
def find_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        cv2.drawChessboardCorners(frame, (board_w, board_h), corners, ret)

    return ret, corners, frame


cap = cv2.VideoCapture(0)

collected = 0
max_images = 10

print("Press SPACE to capture calibration image. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret_corners, corners, vis = find_corners(frame)

    cv2.imshow("Calibration", vis)

    key = cv2.waitKey(1) & 0xFF

    # SPACE = capture image
    if key == 32 and ret_corners:
        objpoints.append(objp.copy())
        imgpoints.append(corners)

        collected += 1
        print(f"Captured {collected}/{max_images}")

        cv2.waitKey(500)

    # Q = quit early
    if key == ord('q') or collected >= max_images:
        break

cap.release()
cv2.destroyAllWindows()


# CALIBRATION
cap_test = cv2.VideoCapture(0)
ret, frame = cap_test.read()
img_shape = frame.shape[:2]
cap_test.release()

ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_shape[::-1],
    None,
    None
)

# RESULTS
print("\nIntrinsics:")
print(intrinsics)

print("\nDistortion:")
print(distortion)

for i in range(len(rvecs)):
    print(f"\nImage {i} Rotation:")
    print(rvecs[i])
    print(f"Image {i} Translation:")
    print(tvecs[i])

# SAVE CALIBRATION
np.savez('my_camera.npz',
         intrinsics=intrinsics,
         distortion=distortion)

print("\nCalibration saved to camera.npz")

cv2.destroyAllWindows()