# aula_04_exe_06.py

import cv2
import numpy as np

def main():
    # Load image
    image = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE )
    flood_image = image.copy()
    h, w = image.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)

    seed_point = (430, 30)

    new_value = 255

    lo_diff = 5
    up_diff = 5

    #flood fill
    retval, _, _, rect = cv2.floodFill(
        flood_image,
        mask,
        seed_point,
        new_value,
        loDiff=lo_diff,
        upDiff=up_diff,
        flags=cv2.FLOODFILL_FIXED_RANGE
    )

    cv2.imshow("Original Image", image)
    cv2.imshow("Flood Filled Image", flood_image)

    print(f"Number of pixels filled: {retval}")
    print(f"Bounding rectangle: {rect}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()