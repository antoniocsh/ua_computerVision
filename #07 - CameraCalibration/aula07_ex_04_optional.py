import numpy as np
import cv2

# CHESSBOARD SETTINGS
board_w = 5
board_h = 7
square_size = 30

# LOAD CAMERA PARAMETERS
with np.load('my_camera.npz') as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']

print("Camera loaded")

# OBJECT POINTS (3D)
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
objp *= square_size

# CUBE MODEL
cube_size = square_size  # 30 mm

cube = np.float32([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, -cube_size],
    [cube_size, 0, -cube_size],
    [cube_size, cube_size, -cube_size],
    [0, cube_size, -cube_size]
])
# CAMERA
cap = cv2.VideoCapture(0)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # DETECT CHESSBOARD
    ret, corners = cv2.findChessboardCorners(gray, (board_w, board_h), None)

    if ret:
        # refine corners
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    
        # SOLVE PNP (POSE)
    
        _, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            intrinsics,
            distortion
        )

    
        # PROJECT CUBE
    
        imgpts, _ = cv2.projectPoints(
            cube,
            rvec,
            tvec,
            intrinsics,
            distortion
        )

        imgpts = np.int32(imgpts).reshape(-1, 2)

    
        # DRAW BASE
    
        frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 0, 255), 3)

        # DRAW TOP
        frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)

        # DRAW VERTICAL LINES
        for i in range(4):
            frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[i + 4]), (0, 0, 255), 3)


    # SHOW RESULT

    cv2.imshow("Live AR Cube", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()