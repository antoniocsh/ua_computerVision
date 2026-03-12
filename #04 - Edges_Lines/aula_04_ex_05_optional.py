import cv2
import numpy as np

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:

    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 1, 128)

    # Show original camera feed
    cv2.imshow("Camera", frame)

    # Show Canny results
    cv2.imshow("Canny (1 , 255)", edges)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


capture.release()
cv2.destroyAllWindows()