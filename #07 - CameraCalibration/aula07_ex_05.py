import cv2
import numpy as np

# LOAD CAMERA PARAMETERS
with np.load('my_camera.npz') as data:
    intrinsics = data['intrinsics']
    distortion = data['distortion']

print("Camera loaded")

# DICTIONARIES
dict_6x6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

params = cv2.aruco.DetectorParameters()

detector_6x6 = cv2.aruco.ArucoDetector(dict_6x6, params)
detector_4x4 = cv2.aruco.ArucoDetector(dict_4x4, params)

# MARKER SIZE (same assumption for both here)
marker_length = 0.03  # meters

objPoints = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

# CAMERA
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    # DETECT IN BOTH DICTIONARIES

    corners1, ids1, _ = detector_6x6.detectMarkers(frame)
    corners2, ids2, _ = detector_4x4.detectMarkers(frame)

    all_corners = []
    all_ids = []
    all_sources = []  # track which dict they came from

    if ids1 is not None:
        all_corners += corners1
        all_ids += list(ids1.flatten())
        all_sources += ["6x6"] * len(ids1)

    if ids2 is not None:
        all_corners += corners2
        all_ids += list(ids2.flatten())
        all_sources += ["4x4"] * len(ids2)


    # DRAW + POSE

    for i in range(len(all_ids)):
        corners = all_corners[i]

        ret, rvec, tvec = cv2.solvePnP(
            objPoints,
            corners[0],
            intrinsics,
            distortion
        )

        # color depends on dictionary
        color = (0, 255, 0) if all_sources[i] == "6x6" else (255, 0, 0)

        cv2.drawFrameAxes(
            frame,
            intrinsics,
            distortion,
            rvec,
            tvec,
            marker_length * 0.5
        )

        cv2.aruco.drawDetectedMarkers(frame, [corners], np.array([[all_ids[i]]]))

    cv2.imshow("BOTH ArUco Dictionaries", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()