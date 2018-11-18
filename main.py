import numpy as np
import math as m
import cv2
import cv2.aruco as aruco


def class MyPoint:
    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]


def CalculateDistanceBetweenPoints(p1, p2):
    a = m.pow(abs(p1[0] - p2[0]))
    b = m.pow(abs(p1[1] - p2[1]))
    distance = m.sqrt(a + b)
    return distance


def CalculateMiddlePointBetweenPoints(p1, p2):
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    return MyPoit

#image = cv2.imread("1.png")


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, image = cap.read()
    if(ret == True):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image, aruco_dict, parameters=parameters)
        #print(corners, ids, rejectedImgPoints)
        if(len(corners) > 0):
            print(corners[0])
        aruco.drawDetectedMarkers(image, corners, ids)
        #aruco.drawDetectedMarkers(image, rejectedImgPoints, borderColor=(100, 0, 240))

        cv2.imshow('frame', image)

    cv2.waitKey(50)
cv2.destroyAllWindows()
