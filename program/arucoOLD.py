import numpy as np
import math as m
import cv2
import configuration as configuration
import cv2.aruco as aruco
import decimal


class FloorPlan():
    def __init__(self, pixelSize_ONE_METER):
        self.ONE_METER_SCALE = pixelSize_ONE_METER

    def calculateSizeToPixels(self, centimeters):
        return self.ONE_METER_SCALE * (centimeters/100)


class ArucoPointPosition:
    def __init__(self, position, rotation, size=configuration.MARKER_SIZE):
        self.position = position
        self.rotation = rotation
        self.size = size

    def getMarkerLine(self, scale):
        return ((self.position[0]-40, self.position[1]), (self.position[0]+40, self.position[1]))

    def getMiddlePoint(self):
        return calculateMiddlePointBetweenPoints(self.getMarkerLine(1)[0], self.getMarkerLine(1)[1])


class ArucoInfo:
    def __init__(self, points, markerId):
        self.uL = points[0]
        self.uR = points[1]
        self.dR = points[2]
        self.dL = points[3]
        self.markerId = markerId
        self.estimatedAngle = None
        self.estimatedDistance = None
        #self.sample50cm = 15

        self.sample50cm = 5.6  # theta 360

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


def projectArucoMarker(imgW, imgH, floor, marker=None):
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
        rightDistance = calculateDistanceBetweenPoints(marker.uR, marker.dR) / configuration.MARKER_SIZE
        estimatedDistanceMiddle = decimal.Decimal(round(marker.sample50cm * 50 / middleDistance, 2))
        estimatedDistanceRight = decimal.Decimal(round(marker.sample50cm * 50 / rightDistance, 2))

        halfMarkerSize = decimal.Decimal(configuration.MARKER_SIZE/2)

        shorterLength = min(estimatedDistanceMiddle, estimatedDistanceRight)
        longerLength = max(estimatedDistanceMiddle, estimatedDistanceRight)
        if(shorterLength + halfMarkerSize > longerLength):

            cameraAngle = decimal.Decimal((m.pow(estimatedDistanceMiddle, 2) + m.pow(halfMarkerSize, 2) - m.pow(estimatedDistanceRight, 2))) / decimal.Decimal((2 * halfMarkerSize * estimatedDistanceMiddle))

            estimatedAngle = m.acos(cameraAngle)   # round(((middlePoint.x * (configuration.CAMERA_ANGLE * 2) / imgW) - configuration.CAMERA_ANGLE) / 2, 1)
            marker.estimatedAngle = decimal.Decimal(m.degrees(estimatedAngle)) - decimal.Decimal(90)
        else:
            marker.estimatedAngle = 0
        marker.estimatedDistance = estimatedDistanceMiddle

        cv2.line(img, (upperMiddlePoint.x, upperMiddlePoint.y), (downMiddlePoint.x, downMiddlePoint.y), (0, 0, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Length: ' + str(middleDistance), (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, 'Estimated distance: ' + str(estimatedDistanceMiddle) + ' cm', (10, 50), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Estimated angle: ' + str(marker.estimatedAngle) + ' degrees', (10, 70), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    projectArucoPosition(floor, marker)
    cv2.imshow("Aruco projection", img)


def projectArucoPosition(floor, marker):
    markerPos = ArucoPointPosition((1222, 743), 90)
    if not hasattr(projectArucoPosition, "lastPoint"):
        projectArucoPosition.lastPoint = None
    img = cv2.imread('m2ms.png',  cv2.IMREAD_COLOR)
    cv2.line(img, markerPos.getMarkerLine(1)[0], markerPos.getMarkerLine(1)[1], (0, 0, 255), 5)
    if(marker != None):
        middlePoint = markerPos.getMiddlePoint()
        angleRadians = (m.pi / 180.0) * (float(marker.estimatedAngle) + markerPos.rotation)
        posPointX = int(middlePoint[0] + m.cos(angleRadians) * floor.calculateSizeToPixels(int(marker.estimatedDistance)))
        posPointY = int(middlePoint[1] + m.sin(angleRadians) * floor.calculateSizeToPixels(int(marker.estimatedDistance)))
        projectArucoPosition.lastPoint = (posPointX, posPointY)
        cv2.circle(img, (posPointX, posPointY), 10, (0, 255, 0), -1)
    elif(projectArucoPosition.lastPoint != None):
        cv2.circle(img, projectArucoPosition.lastPoint, 10, (0, 210, 50), -1)

    img = image_resize(img, width=1200)
    cv2.imshow("Display frame", img)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
# image = cv2.imread("1.png")


def mainLoop():
    floor = FloorPlan(111)  # cm
    markers = []
    width = 0
    height = 0
    cap = cv2.VideoCapture(1)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    while(cap.isOpened()):
        ret, image = cap.read()
        if(ret == True):
            markers = []
            height = len(image)
            width = len(image[0])
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                image, aruco_dict, parameters=parameters)
            # print(corners, ids, rejectedImgPoints)
            if(len(corners) > 0):
                for i in range(len(corners)):
                    marker = ArucoInfo(corners[i][0], ids[i][0])
                    markers.append(marker)
                    # print(corners[i])
                    projectArucoMarker(width, height, floor, marker)
                #aruco.drawDetectedMarkers(image, corners, ids)
            else:
                projectArucoMarker(width, height, floor)
            #aruco.drawDetectedMarkers(image, rejectedImgPoints, borderColor=(100, 0, 240))

            cv2.imshow('frame', image)

        cv2.waitKey(10)
    cv2.destroyAllWindows()


mainLoop()
