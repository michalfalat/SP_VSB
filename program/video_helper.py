import cv2
import time
from image_helper import image_resize
from detections import detectFace


def process_video(inputVideoName, recordVideo, maxHeight=1920, fps=29.0, outputVideoName="outputVideo"):
    HEIGHT = maxHeight
    THRESHOLD = 20
    INIT_FRAME = 2
    RECORD_VIDEO = recordVideo
    cap = cv2.VideoCapture(inputVideoName)
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
    finalOutputPath = "programOutputs\\" + outputVideoName + ".mp4"

    if(RECORD_VIDEO):
        out = cv2.VideoWriter(finalOutputPath, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
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
        frame_face = detectFace(frame_current_resized)
        # frame_landmarks = detectLandmarks(frame_current_resized)

        # cv2.imshow("Motion", frame_motion)
        # cv2.imshow("Pedestrian", frame_pede)
        cv2.imshow("Motion", frame_face)
        # cv2.imshow("Landmarks", frame_landmarks)
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
