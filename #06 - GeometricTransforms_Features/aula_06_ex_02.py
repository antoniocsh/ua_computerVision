import numpy as np
import cv2
import math

src = cv2.imread("image.jpeg")
dst = cv2.imread("image_tf.jpg")

srcPts = []
dstPts = []

def select_src(event, x, y, flags, param):
    global srcPts, src
    if event == cv2.EVENT_LBUTTONDOWN and len(srcPts) < 3:
        srcPts.append((x, y))
        cv2.circle(src, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(src, str(len(srcPts)), (x+10, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("Source", src)

def select_dst(event, x, y, flags, param):
    global dstPts, dst
    if event == cv2.EVENT_LBUTTONDOWN and len(dstPts) < 3:
        dstPts.append((x, y))
        cv2.circle(dst, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(dst, str(len(dstPts)), (x+10, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Destination", dst)

cv2.imshow("Source", src)

cv2.setMouseCallback("Source", select_src)

print("Select 3 points in SOURCE image")
cv2.waitKey(0)

cv2.imshow("Destination", dst)
cv2.setMouseCallback("Destination", select_dst)

print("Select the SAME 3 points in DEST image")
cv2.waitKey(0)

cv2.destroyAllWindows()

np_srcPts = np.array(srcPts, dtype=np.float32)
np_dstPts = np.array(dstPts, dtype=np.float32)

M = cv2.getAffineTransform(np_srcPts, np_dstPts)

print("Transformation matrix:\n", M)

warp_dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

cv2.imshow("Warped Image", warp_dst)
cv2.imshow("Target Image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

a, c, tx = M[0]
b, d, ty = M[1]

# Translation
print("tx =", tx)
print("ty =", ty)

# Scale (optional)
sx = np.sign(a) * np.sqrt(a**2 + b**2)
sy = np.sign(d) * np.sqrt(c**2 + d**2)

print("sx =", sx)
print("sy =", sy)

# Rotation
psi = math.atan2(b, a)
print("Rotation (radians):", psi)
print("Rotation (degrees):", math.degrees(psi))

diff = cv2.absdiff(dst, warp_dst)

cv2.imshow("Difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()