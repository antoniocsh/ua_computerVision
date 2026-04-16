import cv2
import numpy as np

img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE) 
# img = cv2.imread('../images/wdg2.bpm', cv2.IMREAD_GRAYSCALE)

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        flood = img.copy()
        h, w = img.shape
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(
            flood, mask, (x, y), 255,
            loDiff=5, upDiff=5,
            flags=cv2.FLOODFILL_FIXED_RANGE
        )

        cv2.imshow("Image", flood)

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click)

cv2.waitKey(0)
cv2.destroyAllWindows()