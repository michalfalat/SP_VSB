import cv2
import time
from image_helper import image_resize
from detections import detectFace, FACES_COUNTER, detectTFPose, detectLandmarks, detectMotion, detectPedestrian
from detections import draw_landmarks, draw_TF_pose
from seat_belt import detect_seat_belt, draw_seatbelt_lines, print_seatbelt_info
from head_orientation import draw_head_orientation, print_head_orientation_info
from openpose_detector import detectOPPose
from pose_unifier import get_coordinates, get_human_image
from nn import save_train_frame, evaluate, draw_nn_result
from filters import nn_filter


def process_video(inputVideoName, recordVideo, maxHeight=800, fps=59.0, outputVideoName="outputVideo"):
    HEIGHT = maxHeight
    THRESHOLD = 50
    INIT_FRAME = 100
    RECORD_VIDEO = recordVideo
    PRINT_STATISTICS = True
    cap = cv2.VideoCapture(inputVideoName)
    cap.set(1, INIT_FRAME)
    out = None
    textFromTop = 50

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
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    while length:
        length  -= 1
        ret, frame_current = cap.read()
        if (frame_current is None):
            print(frame_current)
            print("No more frames")
            continue
        textFromTop = 50
        frame_current_resized = image_resize(frame_current, None, HEIGHT)

        # frame = detectMotion(frame_prev_resized, frame_current_resized, THRESHOLD)
        # frame = detectPedestrian(frame_current_resized)
        # face = detectFace(frame_current_resized)

        frame = frame_current_resized
        humansTF = detectTFPose(frame_current_resized)
        frameNN = get_human_image(frame_current_resized, humansTF[0], "TF", True)
        result = evaluate(frameNN)
        nn_result_text, nn_color, nn_className = result.process_result()

        nn_result_text, nn_color, nn_className = nn_filter(nn_result_text, nn_color, nn_className)

        # save_train_frame(frameNN, "various", 64)
        # humansOP = detectOPPose(frame_current_resized)
        # get_human_image(frame_current_resized, humansOP[0], "OP", True)
        # landmarks = detectLandmarks(frame_current_resized)
        

        # for (i, rect) in enumerate(landmarks):
        #     frame, p1, p2, angle = draw_landmarks(frame, rect)
        #     res = draw_head_orientation(frame, p1, p2, angle)
        #     print_head_orientation_info(frame, res, (10, textFromTop))
        # textFromTop +=50


        # for human in humans:
        #     seatbelt_lines, offset = detect_seat_belt(frame_current_resized, human)
        #     frame = draw_seatbelt_lines(frame, seatbelt_lines, offset)
        #     frame = print_seatbelt_info(frame, seatbelt_lines, (10, textFromTop))
        # textFromTop +=50

        frame = draw_nn_result(frame, nn_result_text, (10, textFromTop), nn_color)
        textFromTop += 50
        frame = draw_TF_pose(frame, humansTF)

        cv2.imshow("Result", frame)
        cv2.imshow("Result NN", frameNN)

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
        print("Saving Video to: " + finalOutputPath)
        out.release()
    cv2.destroyAllWindows()
