import cv2
import time
from image_helper import image_resize
from detections import detectFace, FACES_COUNTER, detectTFPose, detectLandmarks, detectMotion, detectPedestrian
from detections import draw_landmarks, draw_TF_pose
from seat_belt import detect_seat_belt, draw_seatbelt_lines, print_seatbelt_info


def process_video(inputVideoName, recordVideo, maxHeight=1000, fps=29.0, outputVideoName="outputVideo"):
    HEIGHT = maxHeight
    THRESHOLD = 50
    INIT_FRAME = 0
    RECORD_VIDEO = recordVideo
    PRINT_STATISTICS = True
    cap = cv2.VideoCapture(inputVideoName)
    cap.set(1, INIT_FRAME)
    out = None

    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # size = (frame_width, frame_height)
    r = HEIGHT / float(frame_height)
    size = (int(frame_width * r), HEIGHT)
    finalOutputPath = "video_records\\" + outputVideoName + ".mp4"

    if(RECORD_VIDEO == True):
        out = cv2.VideoWriter(finalOutputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
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

        # frame = detectMotion(frame_prev_resized, frame_current_resized, THRESHOLD)
        # frame = detectPedestrian(frame_current_resized)
        # face = detectFace(frame_current_resized)

        frame = frame_current_resized
        humans = detectTFPose(frame_current_resized)
        landmarks = detectLandmarks(frame_current_resized)

        for human in humans:
            seatbelt_lines, offset = detect_seat_belt(frame_current_resized, human)
            frame = draw_seatbelt_lines(frame, seatbelt_lines, offset)
            frame = print_seatbelt_info(frame, seatbelt_lines, (50, 50))

        frame = draw_TF_pose(frame, humans)
        frame = draw_landmarks(frame, landmarks)

        cv2.imshow("Result", frame)

        if (PRINT_STATISTICS):
            newTime = time.time()
            timeDif = newTime - oldTime
            print("Frame " + str(counter) + ": " + str(timeDif))
            oldTime = newTime
            timeSum += timeDif
        counter += 1

        if(RECORD_VIDEO == True):
            print("writing frame")
            out.write(frame)

        if cv2.waitKey(40) == 27:
            break

        frame_prev_resized = frame_current_resized
    cap.release()

    if (PRINT_STATISTICS):
        print("TOTAL TIME:")
        print(timeSum)
        print("AVERAGE TIME:")
        print(timeSum/counter)

        print("FACES DETECTED:")
        print(FACES_COUNTER)

    if(RECORD_VIDEO == True):
        out.release()
    cv2.destroyAllWindows()
