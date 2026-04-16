import numpy as np
import cv2
import sys


# 1. Load image
try:
    img1 = cv2.imread("myimage" + sys.argv[1] + ".jpg")
except:
    img1 = cv2.imread("myimage1.jpg")

DIM_X = int(sys.argv[2]) if len(sys.argv) > 2 else 6

DIM_Y = int(sys.argv[3]) if len(sys.argv) > 3 else 9

img = img1.copy()
img_display = img.copy()

# 2. Store clicked points
points = []

def select_points(event, x, y, flags, param):
    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        # Draw point
        cv2.circle(img_display, (x, y), 50, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(points)), (x+50, y+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 6)

        cv2.imshow("Select Corners", img_display)

# 3. Select 4 corners
cv2.namedWindow("Select Corners", cv2.WINDOW_NORMAL)
cv2.imshow("Select Corners", img_display)
cv2.setMouseCallback("Select Corners", select_points)

print("Click the 4 corners (in order: TL, TR, BR, BL)")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to numpy
src_pts = np.array(points, dtype=np.float32)

# 4. Define real-world dimensions
# Choose a scale (e.g., 20 pixels per cm)
scale = 200

width = int(DIM_X * scale)
height = int(DIM_Y * scale)

dst_pts = np.array([
    [0, 0],              # Top-left
    [width, 0],          # Top-right
    [width, height],     # Bottom-right
    [0, height]          # Bottom-left
], dtype=np.float32)

# 5. Compute homography
H, mask = cv2.findHomography(src_pts, dst_pts)

print("Homography matrix:\n", H)

# 6. Warp perspective
warped = cv2.warpPerspective(img, H, (width, height))

# 7. Show results
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Rectified", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", img)
cv2.imshow("Rectified", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()