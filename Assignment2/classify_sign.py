import numpy as np
import cv2

# Adapted from classify_sign_template.cpp

WARPED_XSIZE = 200
WARPED_YSIZE = 300

canny_thresh = 154

VERY_LARGE_VALUE = 100000
NO_MATCH = 0
STOP_SIGN = 1
SPEED_LIMIT_40_SIGN = 2
SPEED_LIMIT_80_SIGN = 3

sign_recog_result = NO_MATCH
speed_40 = cv2.imread("speed_40.bmp", cv2.IMREAD_UNCHANGED)
speed_80 = cv2.imread("speed_80.bmp", cv2.IMREAD_UNCHANGED)

# you run your program on these three examples (uncomment the two lines below)
sign_name = "stop4"
# sign_name = "speedsign12"
# sign_name = "speedsign3"
# sign_name = "speedsign4"

final_sign_input_name = sign_name + ".jpg"
final_sign_output_name = sign_name + "_result" + ".jpg"
src = cv2.imread(final_sign_input_name, 1)

# Convert image to gray and blur it
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# Blurring the image to make a low pass filter to smooth edges
src_gray = cv2.blur(src_gray, (3, 3))

# here you add the code to do the recognition, and set the variable
# sign_recog_result to one of STOP_SIGN, SPEED_LIMIT_40_SIGN, SPEED_LIMIT_80_SIGN, or NO_MATCH

# find the edges using canny edge detection first
canny_output = cv2.Canny(src_gray, canny_thresh, canny_thresh*2, apertureSize=3)

largest_area = 0
largest_contour = None
# find the contours, _ is a throwaway, what is it
_, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(0, 0))
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, len(cnt) * 0.02, True)
    temp = cv2.contourArea(approx, False)

    # Get the largest Contour
    if temp > largest_area:
        largest_contour = approx
        largest_area = temp

largest_contour_img = np.copy(src)
cv2.drawContours(largest_contour_img, [largest_contour], -1, (0, 255, 0))
large_contour_window = "Largest Contour"
cv2.namedWindow(large_contour_window)
cv2.imshow(large_contour_window, largest_contour_img)

# do we have a stop sign or do we keep going
if len(largest_contour) is 8:
    sign_recog_result = STOP_SIGN

# Potentially have a speed sign do more recognition
elif len(largest_contour) is 4:
    # determine the top left, bottom right, bottom left, and top right for the endpoints of our square
    # math taken from:
    # http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    pts = largest_contour.reshape(4, 2)

    # The endpoints of our contour in clockwise order
    rect1 = np.zeros((4, 2),np.float32)

    # the top-left point has the smallest sum whereas the bottom right has the largest sum
    s = pts.sum(axis=1)
    rect1[0] = pts[np.argmin(s)]
    rect1[2] = pts[np.argmax(s)]

    # compute the difference between the points --- the top-right will have the minimum difference
    # and the bottom left will have the maximum difference
    diff = np.diff(pts, axis=1)
    rect1[1] = pts[np.argmin(diff)]
    rect1[3] = pts[np.argmax(diff)]

    # Getting our reference rect in counterclockwise order
    rect2 = np.array([[0, 0],
                      [WARPED_XSIZE, 0],
                      [WARPED_XSIZE, WARPED_YSIZE],
                      [0, WARPED_YSIZE]], np.float32)

    # Get the Perspective transformation Matrix
    perspective = cv2.getPerspectiveTransform(rect1, rect2)
    warp_img = cv2.warpPerspective(src_gray, perspective, (WARPED_XSIZE, WARPED_YSIZE))

    warp_window = "Warp Perspective Image"
    cv2.namedWindow(warp_window)
    cv2.imshow(warp_window, warp_img)

    # comparing the images
    if warp_img.data == speed_40.data:
        sign_recog_result = SPEED_LIMIT_40_SIGN
    elif warp_img.data == speed_80.data:
        sign_recog_result = SPEED_LIMIT_80_SIGN

else:
    sign_recog_result = NO_MATCH


text = ""
if sign_recog_result is SPEED_LIMIT_40_SIGN:
    text = "Speed 40"
elif sign_recog_result is SPEED_LIMIT_80_SIGN:
    text = "Speed 80"
elif sign_recog_result is STOP_SIGN:
    text = "Stop"
elif sign_recog_result is NO_MATCH:
    text = "Fail"

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.75
thickness = 3
textOrig = (10, 130)

cv2.putText(src, text, textOrig, font, fontScale, (255, 255, 255))
source_window = "Result"

cv2.namedWindow(source_window)
cv2.imshow(source_window, src)
cv2.imwrite(final_sign_output_name, src)
cv2.waitKey(0)

