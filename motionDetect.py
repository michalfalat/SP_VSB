import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


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
    HEIGHT = 700
    THRESHOLD = 50
    INIT_FRAME = 10
    RECORD_VIDEO = True
    # window_name="Cam feed"
    # cv2.namedWindow(window_name)
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture("videos\kostol.mp4")
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
        out = cv2.VideoWriter('kostool.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, size)
        # out = cv2.VideoWriter('pedestrian.mp4', 0x00000021, 20.0, size)
        # out = cv2.VideoWriter('kostool.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, size)

    if cap.isOpened():

        ret, frame = cap.read()

    else:
        ret = False

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while ret:
        ret, frame = cap.read()
        frame1 = image_resize(frame1, None, HEIGHT)
        frame2 = image_resize(frame2, None, HEIGHT)
        # VideoFileOutput.write(frame)

        d = cv2.absdiff(frame1, frame2)

        grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(grey, (5, 5), 0)
        ret, th = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
        img, c, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame1, c, -1, (0, 255, 0), 2)
        frame1 = detectPedestrian(frame1)

        # cv2.imshow("win1",frame2)
        cv2.imshow("inter", frame1)

        if(RECORD_VIDEO):
            out.write(frame1)

        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = cap.read()
    cap.release()

    if(RECORD_VIDEO):
        out.release()
    cv2.destroyAllWindows()


def detectPedestrian(image):
     # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    # image = imutils.resize(image, width=min(600, image.shape[1]))
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

    # show the output images
    # cv2.imshow("Before NMS", orig)
    # cv2.imshow("After NMS", image)
    return image


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


main()
