#!/usr/bin/env python

import cv2
import sys
import numpy
import math
from numpy import matrix


R = numpy.array([[0.902701, 0.051530, 0.427171],
                 [0.182987, 0.852568, -0.489535],
                 [-0.389418, 0.520070, 0.760184]],
                numpy.float32)

rvec = cv2.Rodrigues(R)[0]

cameraMatrix = numpy.array([[-1000.000000, 0.000000, 0.000000],
                            [0.000000, -2000.000000, 0.000000],
                            [0.000000, 0.000000, 1.000000]],numpy.float32)

tvec = numpy.array([10,15,20], numpy.float32)

objectPoints = numpy.array([[0.1251, 56.3585, 19.3304],
                            [80.8741, 58.5009, 47.9873],
                            [35.0291, 89.5962, 82.2840],
                            [74.6605, 17.4108, 85.8943],
                            [71.0501, 51.3535, 30.3995],
                            [1.4985, 9.1403, 36.4452],
                            [14.7313, 16.5899, 98.8525],
                            [44.5692, 11.9083, 0.4669],
                            [0.8911, 37.7880, 53.1663],
                            [57.1184, 60.1764, 60.7166]], numpy.float32)

imagepoints,jac = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, None)


#######################################################
# Question 1
#######################################################


def computeprojectionmatrix(object_points, image_points):
    # Constructing our A
    A = numpy.zeros((2*len(objectPoints), 12))

    r = 0

    for obj_row, img_row in zip(object_points, image_points):
        A[r] = [obj_row[0], obj_row[1], obj_row[2], 1,
                0, 0, 0, 0,
                -img_row[0][0] * obj_row[0], -img_row[0][0] * obj_row[1], -img_row[0][0]*obj_row[2], -img_row[0][0]]
        A[r+1] = [0, 0, 0, 0,
                  obj_row[0], obj_row[1], obj_row[2], 1,
                  -img_row[0][1] * obj_row[0], -img_row[0][1] * obj_row[1], -img_row[0][1] * obj_row[2], -img_row[0][1]]
        r += 2

    u, s, v = numpy.linalg.svd(A)

    #print s

    # Find the smallest singular value in s
    smallest = s[0]
    s_index = 0

    for i, val in enumerate(s):
        if val < smallest:
            smallest = val
            s_index = i

    # Found the smallest index, now get the value of V at that index
    m = v[s_index]

    # our result came out as a 1d array, but since its our projection matrix we need it to be a in matrix form
    return m.reshape((3, 4))


def decomposeprojectionmatrix(m):
    r = numpy.zeros((3, 3))
    k = numpy.zeros((3, 3))
    t = numpy.zeros((1, 3))

    # get our scaling value
    y = math.sqrt(m[2][0]**2 + m[2][1]**2 + m[2][2]**2)

    # Using a normalized M offset from the scale
    m_normal = m / y

    # Check our Tz, negate our normalized m if it is negative
    Tz = m_normal[2][3]

    if Tz < 0:
        m_normal = -m_normal

    Tz = m_normal[2][3]

    # Setting the bottom row of our rotation matrix
    for i in range(0, 3):
        r[2][i] = m_normal[2][i]

    q1 = numpy.transpose([m_normal[0][0], m_normal[0][1], m_normal[0][2]])
    q2 = numpy.transpose([m_normal[1][0], m_normal[1][1], m_normal[1][2]])
    q3 = numpy.transpose([m_normal[2][0], m_normal[2][1], m_normal[2][2]])
    q4 = numpy.transpose([m_normal[0][3], m_normal[1][3], m_normal[2][3]])

    # Find our parameters

    ox = numpy.dot(numpy.transpose(q1), q3)
    oy = numpy.dot(numpy.transpose(q2), q3)

    fx = math.sqrt(numpy.dot(numpy.transpose(q1), q1) - ox**2)
    fy = math.sqrt(numpy.dot(numpy.transpose(q2), q2) - oy ** 2)

    # Setting row 1 and 2 of our rotation matrix
    for i in range(0, 3):
        r[0][i] = (ox * m_normal[2][i] - m_normal[0][i]) / fx
        r[1][i] = (oy * m_normal[2][i] - m_normal[1][i]) / fy

    Tx = (ox * Tz - m_normal[0][3])/fx
    Ty = (oy * Tz - m_normal[1][3])/fy

    # assuming Sx and Sy is 1 our K becomes
    k[0] = [-fx, 0, ox]
    k[1] = [0, -fy, oy]
    k[2] = [0, 0, 1]

    # calculating our translation matrix
    t[0][0] = Tx
    t[0][1] = Ty
    t[0][2] = Tz

    return k, r, t

M = computeprojectionmatrix(objectPoints, imagepoints)
K, R, T = decomposeprojectionmatrix(M)

with open('assign3-out', 'w') as f:
    f.write('Initial Camera Matrix\n')
    f.write(str(cameraMatrix))
    f.write('\nTranslation\n')
    f.write(str(tvec))
    f.write('\nInitial Rotation\n')
    f.write(str(R))
    f.write('\nInitial ObjectPoints\n')
    f.write(str(objectPoints))
    f.write('\nImage Points\n')
    f.write(str(imagepoints))
    f.write("\n\nComputed Camera Calibration Matrix\n")
    f.write(str(K))
    f.write("\nComputed Rotation Matrix\n")
    f.write(str(R))
    f.write("\nComputed Translation Matrix\n")
    f.write(str(T))
