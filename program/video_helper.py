from enum import Enum
import time
import cv2
from image_helper import image_resize
from detections import detectFace, FACES_COUNTER, detectTFPose, detectLandmarks, detectMotion, detectPedestrian
from detections import draw_landmarks, draw_TF_pose
from seat_belt import detect_seat_belt, draw_seatbelt_lines, draw_seatbelt_info
from head_orientation import draw_head_orientation, draw_head_orientation_info
from openpose_detector import detectOPPose
from pose_unifier import get_coordinates, get_human_image
from nn import save_train_frame, evaluate, draw_nn_result
from filters import nn_filter


class DETECTION_TYPE(Enum):
    TF_POSE = 'TF_POSE',
    OP_POSE = 'OP_POSE'


class TASK_TYPE(Enum):
    POSE_DETECT = 'POSE_DETECT',
    POSE_EVALUATE = 'POSE_EVALUATE',
    SEATBELT = 'SEATBELT',
    HEAD = 'HEAD',
    ALL = 'ALL'


def process_video(args):
    cap = cv2.VideoCapture(args.videoInput)
    cap.set(1, args.initFrame)
    out = None
    text_from_top = 50

    if cap.isOpened() is False:
        print("Unable to read camera feed")
        print('Problem: video ' + args.videoInput + ' was not found')
        return

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    ratio = args.resolutionHeight / float(frame_height)
    size = (int(frame_width * ratio), args.resolutionHeight)

    if args.recordVideo is True:
        out = cv2.VideoWriter(args.videoOutput, cv2.VideoWriter_fourcc(*'mp4v'), args.videoFps, size)

    if cap.isOpened():
        ret, frame = cap.read()

    else:
        ret = False

    ret, frame_prev = cap.read()

    old_time = time.time()
    time_sum = 0
    counter = 1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = length

    while length:
        length -= 1
        if args.printContinuosStatistics is True:
            print("Frame " + str(counter) + "/" + str(total_frames) + ":")
        ret, frame_current = cap.read()
        if frame_current is None:
            continue
        text_from_top = 50
        frame_current_resized = image_resize(frame_current, None, args.resolutionHeight)

        # frame_prev_resized = image_resize(frame_prev, None, args.resolutionHeight)
        # frame = detectMotion(frame_prev_resized, frame_current_resized, THRESHOLD)
        # frame = detectPedestrian(frame_current_resized)
        # face = detectFace(frame_current_resized)

        frame = frame_current_resized
        if args.framework == "TF_POSE":
            humans_detected = detectTFPose(frame_current_resized)

        if args.framework == "OP_POSE":
            humans_detected = detectOPPose(frame_current_resized)
        
        frame_nn = get_human_image(frame_current_resized, humans_detected[0], args.framework, True)
        result_nn = evaluate(frame_nn, 32)
        nn_result_text, nn_color, nn_class_name = result_nn.process_result()

        if args.useFiltering is True:
           nn_result_text, nn_color, nn_class_name = nn_filter(nn_result_text, nn_color, nn_class_name, args.printContinuosStatistics)

        if args.printContinuosStatistics is True:
            result_nn.print_info()

        # save_train_frame(frame_nn, "various", 64)
        # humansOP = detectOPPose(frame_current_resized)
        # get_human_image(frame_current_resized, humansOP[0], "OP", True)



        # HEAD ORIENTATION
        if args.detectHeadOrientation is True:
            landmarks = detectLandmarks(frame_current_resized, text_from_top, args.imagePrintStatistics)

            for (i, rect) in enumerate(landmarks):
                frame, p_1, p_2, angle = draw_landmarks(frame, rect)
                res = draw_head_orientation(frame, p_1, p_2, angle)

                if args.imagePrintStatistics is True:
                    draw_head_orientation_info(frame, res, (10, text_from_top))

            if args.imagePrintStatistics is True:
                text_from_top += 50



        # SEAT BELT DETECTION
        if args.detectSeatBelt is True:
            for human in humans_detected:
                seatbelt_lines, offset = detect_seat_belt(frame_current_resized, human, args.framework)
                frame = draw_seatbelt_lines(frame, seatbelt_lines, offset)

                if args.imagePrintStatistics is True:
                    frame = draw_seatbelt_info(frame, seatbelt_lines, (10, text_from_top))

            if args.imagePrintStatistics is True:
                text_from_top += 50

        if args.imagePrintStatistics is True:
            frame = draw_nn_result(frame, nn_result_text, (10, text_from_top), nn_color)
            text_from_top += 50

        # NATIVE DRAWING METHODS
        # if args.framework == "TF_POSE":
        # frame = draw_TF_pose(frame, humans_detected)

        # if args.framework == "OP_POSE":
        #     frame = draw_TF_pose(frame, humans_detected)

        frame = get_human_image(frame, humans_detected[0], args.framework)

        if args.showOutput is True:
            cv2.imshow("Result", frame)
            #cv2.imshow("Result NN", frame_nn)

        new_time = time.time()
        time_dif = new_time - old_time
        old_time = new_time
        time_sum += time_dif

        if args.printContinuosStatistics is True:
            print("Ellapsed time: " + str(time_dif) + "\n\n")

        counter += 1

        if args.recordVideo is True:
            out.write(frame)

        if cv2.waitKey(40) == 27:
            break

        # frame_prev_resized = frame_current_resized
    cap.release()

    if args.printFinalStatistics is True:

        print("\n\nFINAL STATISTICS:")
        print("USED FRAMEWORK:        \t\t" + args.framework)
        print("RESOLUTION (original): \t\t" + str(frame_width) + "x" + str(frame_height))
        print("RESOLUTION (proceeded):\t\t" + str(size[0]) + "x" + str(size[1]))
        print("TOTAL TIME:            \t\t" + str(time_sum))
        print("AVERAGE TIME:          \t\t" + str(time_sum/counter))

        # print("FACES DETECTED:")
        # print(FACES_COUNTER)

    if args.recordVideo is True:
        print("Saving Video to: " + args.videoOutput)
        out.release()

    cv2.destroyAllWindows()
