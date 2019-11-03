
"""detections"""
from pathlib import Path
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import face_utils
import dlib

HOG = None
FACE_CASCADE = None
FACES_COUNTER = None
FACE_DETECTOR = None
FACE_PREDICTOR = None


# INITS
def init_face_cascade():
    """init_face_cascade"""
    global FACE_CASCADE
    global FACES_COUNTER
    path = "\\datasets\\haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(str(Path(__file__).resolve().parent) + path)
    FACES_COUNTER = 0
    print("FACE_CASCADE Initialized!")


def init_face_prediction():
    global FACE_DETECTOR
    global FACE_PREDICTOR
    path = "\\datasets\\shape_predictor_68_face_landmarks.dat"
    FACE_DETECTOR = dlib.get_frontal_face_detector()
    FACE_PREDICTOR = dlib.shape_predictor(str(Path(__file__).resolve().parent) + path)
    print("FACE_DETECTOR Initialized!")


def init_hog():
    global HOG
    HOG = cv2.HOGDescriptor()
    HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("HOG Initialized!")


# FUNCTIONS

def detectFace(frame):
    global FACES_COUNTER
    global FACE_CASCADE
    if FACE_CASCADE is None:
        init_face_cascade()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detect = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if(len(faces_detect) > 0 and len(faces_detect) <= 2):
        FACES_COUNTER += len(faces_detect)
        print("detect OK")

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces_detect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 128), 2)

    return frame


def detectLandmarks(frame):
    global FACES_COUNTER
    global FACE_PREDICTOR
    global FACE_DETECTOR
    if FACE_PREDICTOR is None or FACE_DETECTOR is None:
        init_face_prediction()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = FACE_DETECTOR(gray, 0)

    print(len(rects))
    FACES_COUNTER += len(rects)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = FACE_PREDICTOR(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame


def detectMotion(frame1, frame2, THRESHOLD):
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
    img, c, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, c, -1, (0, 255, 0), 2)

    return frame1


def detectPedestrian(image):
    global HOG
    if HOG is None:
        init_hog()
    orig = image.copy()

    (rects, weights) = HOG.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)

    return image
