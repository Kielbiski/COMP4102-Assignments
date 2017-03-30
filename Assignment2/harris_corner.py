import numpy as np
import cv2
import math

APETURE_SIZE = 3
BLOCK_SIZE = 5
K = 0.04
MAX_CORNERS = 100000
EIGENVALUE_THRESHOLD = 320000
wdSource = "Conrners"
wdResult = "Result"
tbThreshName = "Threshold"
tbDistanceName = "Distance"

filename = "checker.jpg"
final_corner_output_name = "checkers_corners.jpg"

# Get the Image
src = cv2.imread(filename, 1)

# Convert to greyscale
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gray = cv2.blur(src_gray, (3, 3))

# Get the two derivative from the sobel code
dx = cv2.Sobel(src_gray, cv2.CV_64F, 1, 0, ksize=APETURE_SIZE)
dy = cv2.Sobel(src_gray, cv2.CV_64F, 0, 1, ksize=APETURE_SIZE)

dx = cv2.convertScaleAbs(dx)
dy = cv2.convertScaleAbs(dy)

height, width, _ = src.shape

corners = []
# Iterating over all pixels except edge cases
# Note numpy access list y, x
for x in range(1, width-1):
    for y in range(1, height-1):
        sums = []
        for window_x in range(x-1, x+2):
            for window_y in range(y - 1, y + 2):
                # Calculate window values
                a = dx.item(window_y, window_x)
                a = a * a
                b = dx.item(window_y, window_x) * dy.item(window_y, window_x)
                d = dy.item(window_y, window_x)
                d = d * d
                # Add it to our sumation
                sums.append(np.array([[a, b], [b, d]], np.float64))

        # Get the summation from the window
        C = np.sum(sums, axis=0)

        # get out a b and d out of the matrix C
        a = C.item(0, 0)
        b = C.item(0, 1)
        d = C.item(1, 1)

        # Calculate our eigenvalues
        e1 = ((a + d) / 2.0) + math.sqrt(math.pow(b, 2) + math.pow((a - d) / 2.0, 2))
        e2 = ((a + d) / 2.0) - math.sqrt(math.pow(b, 2) + math.pow((a - d) / 2.0, 2))

        # calculate a score to determine if its a possible corner
        # taken from :
        # http://docs.opencv.org/2.4/doc/tutorials/features2d/trackingmotion/harris_detector/harris_detector.html

        r = (e1*e2) - (K * math.pow(e1 + e2, 2))

        # add the location of the corner to the corner list
        if r > EIGENVALUE_THRESHOLD:
            corners.append((x, y))

print "Corners: " + repr(len(corners))

for x, y in corners:
    cv2.circle(src, (x, y), 1, (255, 0, 0))
# Generate the windows
cv2.namedWindow(wdSource)
cv2.namedWindow(tbThreshName)
cv2.namedWindow(wdResult)
cv2.imshow(tbThreshName, dx)
cv2.imshow(wdResult, dy)
cv2.imshow(wdSource, src)
cv2.imwrite(final_corner_output_name, src)
cv2.waitKey(0)


