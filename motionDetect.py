import cv2
import numpy as np
import sys
from imutils.object_detection import non_max_suppression
from imutils import face_utils
import dlib
import time

# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    if image is None:
        return image
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


def main():
    HEIGHT = 1000  # 1440  # 1920
    THRESHOLD = 20
    INIT_FRAME = 2
    RECORD_VIDEO = False
    name = "auto_8.mp4"
    cap = cv2.VideoCapture("videos\\" + name)
    cap.set(1, INIT_FRAME)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # size = (frame_width, frame_height)
    r = HEIGHT / float(frame_height)
    size = (int(frame_width * r), HEIGHT)

    if(RECORD_VIDEO):
        out = cv2.VideoWriter('auto8_Landmarks_2K_29fps.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 29.0, size)
        # out = cv2.VideoWriter('pedestrian.mp4', 0x00000021, 20.0, size)
        # out = cv2.VideoWriter('kostool.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, size)

    if cap.isOpened():

        ret, frame = cap.read()

    else:
        ret = False

    ret, frame_prev = cap.read()

    frame_prev_resized = image_resize(frame_prev, None, HEIGHT)
    oldTime = time.time()
    timeSum = 0
    counter = 0

    while ret:
        ret, frame_current = cap.read()
        if (frame_current is None):
            break

        frame_current_resized = image_resize(frame_current, None, HEIGHT)

        # frame_motion = detectMotion(frame_prev_resized, frame_current_resized, THRESHOLD)
        # frame_pede = detectPedestrian(frame_current_resized)
        # frame_face = detectFace(frame_current_resized)
        frame_landmarks = detectLandmarks(frame_current_resized)

        # cv2.imshow("Motion", frame_motion)
        # cv2.imshow("Pedestrian", frame_pede)
        # cv2.imshow("Motion", frame_face)
        cv2.imshow("Landmarks", frame_landmarks)
        newTime = time.time()
        timeDif = newTime - oldTime
        print("Frame " + str(counter) + ": " + str(timeDif))
        oldTime = newTime
        timeSum += timeDif
        counter += 1

        if(RECORD_VIDEO):
            out.write(frame_landmarks)

        if cv2.waitKey(40) == 27:
            break

        frame_prev_resized = frame_current_resized
    cap.release()

    print("TOTAL TIME:")
    print(timeSum)
    print("AVERAGE TIME:")
    print(timeSum/counter)
    global faces

    print("FACES DETECTED:")
    print(faces)

    if(RECORD_VIDEO):
        out.release()
    cv2.destroyAllWindows()


def detectFace(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 128), 2)

    # Display the resulting frame
    return frame


def detectLandmarks(frame):
    global faces
    # detect faces in the grayscale image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 0)

    print(len(rects))
    faces += len(rects)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
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
     # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    orig = image.copy()

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (255, 0, 0), 2)

    # show some information on the number of bounding boxes

    return image


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
faces = 0


main()
