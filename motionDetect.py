import cv2
import numpy as np


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
    HEIGHT = 500
    THRESHOLD = 60
    #window_name="Cam feed"
    # cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture("videos\kostol.MP4")

    #filename = 'F:\sample.avi'
    # codec=cv2.VideoWriter_fourcc('X','V','I','D')
    # framerate=30
    #resolution = (500,500)

    #  VideoFileOutput = cv2.VideoWriter(filename,codec,framerate,resolution)

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

        # cv2.imshow("win1",frame2)
        cv2.imshow("inter", frame1)

        if cv2.waitKey(40) == 27:
            break
        frame1 = frame2
        ret, frame2 = cap.read()
    cv2.destroyAllWindows()
    # VideoFileOutput.release()
    cap.release()


main()
