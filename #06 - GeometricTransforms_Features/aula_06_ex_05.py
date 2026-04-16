import numpy as np
import cv2
import math

# Load images
img1 = cv2.imread("image.jpeg")       # Source
img2 = cv2.imread("image_tf.jpg")     # Destination

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# Sort matches by distance (best first)
matches = sorted(matches, key=lambda x: x.distance)


img_matches_10 = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:10], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

img_matches_100 = cv2.drawMatches(
    img1, kp1, img2, kp2, matches[:100], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

img_vis = np.vstack((img_matches_10, img_matches_100))

cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Matches', img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
# M = cv2.getAffineTransform(src_pts, dst_pts)

print("Transformation matrix:\n", M)

# 6. Warp source image
warp_dst = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))

cv2.imshow("Warped Image", warp_dst)
cv2.imshow("Target Image", img2)

# 7. Compute difference
diff = cv2.absdiff(img2, warp_dst)
cv2.imshow("Difference", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 8. Extract transformation parameters
a, c, tx = M[0]
b, d, ty = M[1]

print("\n--- Transformation Parameters ---")

# Translation
print("tx =", tx)
print("ty =", ty)

# Scale
sx = np.sign(a) * np.sqrt(a**2 + b**2)
sy = np.sign(d) * np.sqrt(c**2 + d**2)

print("sx =", sx)
print("sy =", sy)

# Rotation
psi = math.atan2(b, a)
print("Rotation (radians):", psi)
print("Rotation (degrees):", math.degrees(psi))
