import numpy as np
import cv2

img1 = cv2.imread('keble_a_half.bmp',0)          # queryImage
img2 = cv2.imread('keble_b_long.bmp',0) # trainImage

detector = cv2.AKAZE_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

src_pts = np.float32([kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
dst_pts = np.float32([kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)



##################################################################################
# Added Code
#################################################################################
# Calculate the homography between the src and dst points
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# numpy array has their dimensions backwards
height, width = img2.shape

# use the homography to warp image 2 onto image1
warp_img = cv2.warpPerspective(img1, M, (width, height))
pano_img = cv2.bitwise_or(warp_img, img2)

cv2.imshow('Warp', warp_img)
cv2.imwrite('q2-warp.jpg', warp_img)
cv2.imshow('Panorama', pano_img)
cv2.imwrite('q2-panorama.jpg', warp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
