
from head_orientation import detect_head_orientation
from tf_pose.networks import get_graph_path
from tf_pose.estimator import TfPoseEstimator
import dlib
"""detections"""
from pathlib import Path
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import face_utils
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HOG = None
FACE_CASCADE = None
FACES_COUNTER = 0
FACE_DETECTOR = None
FACE_PREDICTOR = None
TF_POSE_ESTIMATOR = None
IMAGE_H = None
IMAGE_W = None
CROP_X = None
CROP_HEIGHT = None
CROP_Y = None
CROP_WIDTH = None
LANDMARK_MEMORY_THRESHOLD = 15
LANDMARK_MEMORY_COUNTER = 0
LANDMARK_MEMORY = None

# INITS
def init_face_cascade():
    """init_face_cascade"""
    global FACE_CASCADE
    global FACES_COUNTER
    path = "\\models\\haarcascade_frontalface_default.xml"
    FACE_CASCADE = cv2.CascadeClassifier(str(Path(__file__).resolve().parent) + path)
    FACES_COUNTER = 0
    print("FACE_CASCADE Initialized!")


def init_face_prediction():
    global FACE_DETECTOR
    global FACE_PREDICTOR
    path = "\\models\\shape_predictor_68_face_landmarks.dat"
    FACE_DETECTOR = dlib.get_frontal_face_detector()
    FACE_PREDICTOR = dlib.shape_predictor(str(Path(__file__).resolve().parent) + path)
    print("FACE_DETECTOR Initialized!")


def init_hog():
    global HOG
    HOG = cv2.HOGDescriptor()
    HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("HOG Initialized!")


def init_TF_pose_estimator():
    global TF_POSE_ESTIMATOR
    model = "mobilenet_thin"
    TF_POSE_ESTIMATOR = TfPoseEstimator(get_graph_path(model), target_size=(216, 216))
    print("TF Pose estimator Initialized!")


# FUNCTIONS

def detect_face(frame):
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

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces_detect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 128), 2)

    return frame


def detect_landmarks(frame):
    global FACE_PREDICTOR
    global FACE_DETECTOR
    if FACE_PREDICTOR is None or FACE_DETECTOR is None:
        init_face_prediction()

    crop, crop_x, crop_y = get_crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    landmarks = FACE_DETECTOR(gray, 0)

    return landmark_filter(landmarks)

def landmark_filter(landmarks):
    global LANDMARK_MEMORY_THRESHOLD
    global LANDMARK_MEMORY_COUNTER
    global LANDMARK_MEMORY

    if len(landmarks) != 0:
        LANDMARK_MEMORY = landmarks
        LANDMARK_MEMORY_COUNTER = 0
    else:
        LANDMARK_MEMORY_COUNTER += 1
        if LANDMARK_MEMORY_COUNTER > LANDMARK_MEMORY_THRESHOLD:
            LANDMARK_MEMORY = landmarks
    return LANDMARK_MEMORY

def draw_no_face(frame, top, printImageStatistics):
    if printImageStatistics is True:
        cv2.putText(frame, "HEAD POSITION: NOT DETECTED", (10, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 3)


def get_crop(frame):
    global IMAGE_H
    global IMAGE_W
    global CROP_X
    global CROP_HEIGHT
    global CROP_Y
    global CROP_WIDTH
    if IMAGE_H is None:
        IMAGE_H, IMAGE_W = frame.shape[:2]
        CROP_X = int(IMAGE_W / 4)
        CROP_HEIGHT = int(IMAGE_H / 2)
        CROP_Y = int(IMAGE_H / 4)
        CROP_WIDTH = int(IMAGE_W / 2)

    return frame[CROP_Y:CROP_Y+CROP_HEIGHT, CROP_X:CROP_X+CROP_WIDTH], CROP_X, CROP_Y

def draw_landmarks(frame, rect):
    crop, crop_x, crop_y = get_crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    shape = FACE_PREDICTOR(gray, rect)
    shape = face_utils.shape_to_np(shape)
    p_1, p_2, angle = detect_head_orientation(frame, shape, crop_x, crop_y)
    for (x, y) in shape:
        cv2.circle(frame, (x + crop_x, y + crop_y), 2, (0, 255, 0), -1)
    return frame, p_1, p_2, angle


def detect_tf_pose(frame, params):
    global TF_POSE_ESTIMATOR
    if TF_POSE_ESTIMATOR is None:
        init_TF_pose_estimator()

    humans = TF_POSE_ESTIMATOR.inference(frame, resize_to_default=True, upsample_size=3.0)

    if params.showNativeOutput is True:
        image = TfPoseEstimator.draw_humans(frame, humans, imgcopy=True)
        cv2.imshow("TF_POSE native output result", image)
    return humans


def draw_TF_pose(frame, humans):
    frame = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
    return frame


# NOT USED
def detect_motion(frame1, frame2, THRESHOLD):
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
    img, c, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, c, -1, (0, 255, 0), 2)

    return frame1


# NOT USED
def detect_pedestrian(image):
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
