import numpy as np
import cv2
import math

# 1. Load images
img1 = cv2.imread("image.jpeg")       # Source
img2 = cv2.imread("image_tf.jpg")     # Destination

# Convert to grayscale (recommended for feature detection)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. ORB features
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# 3. FLANN matcher (for ORB → LSH index)
FLANN_INDEX_LSH = 6

index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1
)

search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# IMPORTANT: ORB descriptors must be uint8
des1 = np.uint8(des1)
des2 = np.uint8(des2)

# KNN match (k=2 for ratio test)
matches = flann.knnMatch(des1, des2, k=2)

# 4. Lowe's ratio test
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Total matches:", len(matches))
print("Good matches:", len(good_matches))

# 5. Draw matches
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, good_matches[:50], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.namedWindow('Good Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Good Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 6. Convert to numpy arrays
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# 7. Estimate affine transform (RANSAC)
M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

print("\nTransformation matrix:\n", M)

# 8. Warp image
warp_dst = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))

cv2.imshow("Warped Image", warp_dst)
cv2.imshow("Target Image", img2)

# 9. Difference image
diff = cv2.absdiff(img2, warp_dst)
cv2.imshow("Difference", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 10. Extract parameters
a, c, tx = M[0]
b, d, ty = M[1]

print("\n--- Transformation Parameters ---")

print("tx =", tx)
print("ty =", ty)

sx = np.sign(a) * np.sqrt(a**2 + b**2)
sy = np.sign(d) * np.sqrt(c**2 + d**2)

print("sx =", sx)
print("sy =", sy)

psi = math.atan2(b, a)
print("Rotation (radians):", psi)
print("Rotation (degrees):", math.degrees(psi))

# 11. Inliers info
print("\nInliers:", np.sum(mask), "/", len(good_matches))