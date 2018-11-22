import numpy as np
import math as m
import cv2
import configuration as configuration
import cv2.aruco as aruco


class ArucoInfo:
    def __init__(self, points, markerId):
        self.uL = points[0]
        self.uR = points[1]
        self.dR = points[2]
        self.dL = points[3]
        self.markerId = markerId
        self.sample50cm = 15

    def getPoints(self):
        return [self.uL, self.uR, self.dL, self.dR]


class PointInfo:
    def __init__(self, coordinates):
        self.x = coordinates[0]
        self.y = coordinates[1]

    def __getitem__(self, index):
        if(index == 0):
            return self.x
        elif(index == 1):
            return self.y
        else:
            return None

    def getCoordinates(self):
        return [self.x, self.y]


def calculateDistanceBetweenPoints(p1, p2):
    a = m.pow(abs(p1[0] - p2[0]), 2)
    b = m.pow(abs(p1[1] - p2[1]), 2)
    distance = m.sqrt(a + b)
    return distance


def calculateMiddlePointBetweenPoints(p1, p2):
    x = int((p1[0] + p2[0]) / 2)
    y = int((p1[1] + p2[1]) / 2)
    return PointInfo([x, y])


def projectArucoMarker(imgW, imgH, marker=None):
    img_size = (imgH, imgW, 3)
    img = np.ones(img_size) * 255
    if(marker != None):
        cv2.line(img, (marker.uL[0], marker.uL[1]), (marker.uR[0], marker.uR[1]), (0, 0, 0))
        cv2.line(img, (marker.uL[0], marker.uL[1]), (marker.dL[0], marker.dL[1]), (0, 0, 0))
        cv2.line(img, (marker.uR[0], marker.uR[1]), (marker.dR[0], marker.dR[1]), (0, 0, 0))
        cv2.line(img, (marker.dL[0], marker.dL[1]), (marker.dR[0], marker.dR[1]), (0, 0, 0))

        upperMiddlePoint = calculateMiddlePointBetweenPoints(marker.uL, marker.uR)
        downMiddlePoint = calculateMiddlePointBetweenPoints(marker.dL, marker.dR)

        middlePoint = calculateMiddlePointBetweenPoints(upperMiddlePoint, downMiddlePoint)

        middleDistance = calculateDistanceBetweenPoints(upperMiddlePoint.getCoordinates(), downMiddlePoint.getCoordinates()) / configuration.MARKER_SIZE
        estimatedDistance = round(marker.sample50cm * 50 / middleDistance, 2)

        estimatedAngle = round(((middlePoint.x * (configuration.CAMERA_ANGLE * 2) / imgW) - configuration.CAMERA_ANGLE) / 2, 1)

        cv2.line(img, (upperMiddlePoint.x, upperMiddlePoint.y), (downMiddlePoint.x, downMiddlePoint.y), (0, 0, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Length: ' + str(middleDistance), (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, 'Estimated distance: ' + str(estimatedDistance) + ' cm', (10, 50), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Estimated angle: ' + str(estimatedAngle) + ' degrees', (10, 70), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Aruco projection", img)


def projectArucoPosition(imgW, imgH):
    img_size = (imgH, imgW, 3)
    img = np.ones(img_size) * 255


# image = cv2.imread("1.png")


def mainLoop():
    markers = []
    width = 0
    height = 0
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, image = cap.read()
        if(ret == True):
            markers = []
            height = len(image)
            width = len(image[0])
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                image, aruco_dict, parameters=parameters)
            # print(corners, ids, rejectedImgPoints)
            if(len(corners) > 0):
                for i in range(len(corners)):
                    marker = ArucoInfo(corners[i][0], ids[i][0])
                    markers.append(marker)
                    print(corners[i])
                    projectArucoMarker(width, height, marker)
                aruco.drawDetectedMarkers(image, corners, ids)
            else:
                projectArucoMarker(width, height)
            # aruco.drawDetectedMarkers(image, rejectedImgPoints, borderColor=(100, 0, 240))

            cv2.imshow('frame', image)

        cv2.waitKey(50)
    cv2.destroyAllWindows()


mainLoop()
