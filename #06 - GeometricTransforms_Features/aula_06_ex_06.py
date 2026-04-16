import numpy as np
import cv2
import sys

try:
    number = int(sys.argv[1]) - 1
except:
    number = 0
if not (0 <= number < 4):
    number = 0
    
print("Using image number:", number + 1)
# 1. Load image
img1 = cv2.imread("../images/homography_1.jpg")
img2 = cv2.imread("../images/homography_2.jpg")  # For reference
img3 = cv2.imread("../images/homography_3.jpg")  # For reference
img4 = cv2.imread("../images/homography_4.jpg")  # For reference
imgs = [img1, img2, img3, img4]

img = imgs[number].copy()   

img_display = img.copy()

# 2. Store clicked points
points = []

def select_points(event, x, y, flags, param):
    global points, img_display

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        # Draw point
        cv2.circle(img_display, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(points)), (x+10, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Select Corners", img_display)

# 3. Select 4 corners
cv2.imshow("Select Corners", img_display)
cv2.setMouseCallback("Select Corners", select_points)

print("Click the 4 corners of the book (in order: TL, TR, BR, BL)")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to numpy
src_pts = np.array(points, dtype=np.float32)

# 4. Define real-world dimensions
# Book size: 17.5 cm × 23.5 cm
# Choose a scale (e.g., 20 pixels per cm)
scale = 20

width = int(17.5 * scale)
height = int(23.5 * scale)

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
cv2.imshow("Original Image", img)
cv2.imshow("Rectified Book", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()