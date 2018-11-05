import numpy as np
import cv2
import cv2.aruco as aruco

#image = cv2.imread("1.png")

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, image = cap.read()
    if(ret == True):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            image, aruco_dict, parameters=parameters)
        print(corners, ids, rejectedImgPoints)
        aruco.drawDetectedMarkers(image, corners, ids)
        #aruco.drawDetectedMarkers(image, rejectedImgPoints, borderColor=(100, 0, 240))

        cv2.imshow('frame', image)

    cv2.waitKey(50)
cv2.destroyAllWindows()
