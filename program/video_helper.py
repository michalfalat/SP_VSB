import time
import cv2
from image_helper import image_resize
from detections import detect_tf_pose, detect_landmarks, draw_landmarks
from seat_belt import detect_seat_belt, draw_seatbelt_lines, draw_seatbelt_info
from head_orientation import draw_head_orientation, draw_head_orientation_info
from openpose_detector import detect_op_pose
from pose_unifier import get_human_image
from nn import save_train_frame, evaluate, draw_nn_result
from filters import nn_filter
from nn_result import NN_result_counter
import numbers


def process_video(args):
    cap = cv2.VideoCapture(args.videoInput)
    cap.set(1, args.initFrame)
    out = None
    text_from_top = 50

    if cap.isOpened() is False:
        print("Unable to read camera feed")
        print('Problem: video ' + args.videoInput + ' was not found')
        return

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
    counter = 0
    people_counter = 0
    skipped_frame_counter = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = length
    nn_result_counter = NN_result_counter()
    nn_result_filtered_counter = NN_result_counter()

    while length:
        length -= 1
        counter += 1
        if args.printContinuosStatistics is True:
            print("Frame " + str(counter) + "/" + str(total_frames) + ":")
        ret, frame_current = cap.read()
        if frame_current is None:
            skipped_frame_counter += 1
            continue

        text_from_top = 50
        frame_current_resized = image_resize(frame_current, None, args.resolutionHeight)

        frame = frame_current_resized
        if args.detectBody is True:
            if args.framework == "TF_POSE":
                humans_detected = detect_tf_pose(frame_current_resized, args)
                if humans_detected is None or len(humans_detected) < 1:
                    continue

            if args.framework == "OP_POSE":
                humans_detected = detect_op_pose(frame_current_resized, args)
                if humans_detected is None or humans_detected.size < 2 or isinstance(humans_detected, numbers.Number) is True:
                    continue

            if humans_detected[0] is not None:
                people_counter += 1
            frame_nn = get_human_image(frame_current_resized, humans_detected[0], args.framework, True)
            result_nn = evaluate(args, frame_nn, 32)
            nn_result_text, nn_color, nn_class_name = result_nn.process_result()

            nn_result_counter.increment(nn_class_name)

            if args.useFiltering is True:
                nn_result_text, nn_color, nn_class_name = nn_filter(nn_result_text, nn_color, nn_class_name, args.printContinuosStatistics)
                nn_result_filtered_counter.increment(nn_class_name)

            if args.printContinuosStatistics is True:
                result_nn.print_info()

        if args.saveTrainImage is True:
            save_train_frame(frame_nn, args.saveTrainImagePath, 64)
            cv2.imshow("Neural network - train image", frame_nn)

        # HEAD ORIENTATION
        if args.detectHeadOrientation is True:
            landmarks = detect_landmarks(frame_current_resized, text_from_top, args.imagePrintStatistics)

            for (i, rect) in enumerate(landmarks):
                frame, p_1, p_2, angle = draw_landmarks(frame, rect)
                res = draw_head_orientation(frame, p_1, p_2, angle, args)

                if args.imagePrintStatistics is True:
                    draw_head_orientation_info(frame, res, (10, text_from_top))

            if args.imagePrintStatistics is True:
                text_from_top += 50

        # SEAT BELT DETECTION
        # actually not used, but working correctly
        if args.detectSeatBelt is True:
            for human in humans_detected:
                seatbelt_lines, offset = detect_seat_belt(frame_current_resized, human, args.framework)
                frame = draw_seatbelt_lines(frame, seatbelt_lines, offset)

                if args.imagePrintStatistics is True:
                    frame = draw_seatbelt_info(frame, seatbelt_lines, (10, text_from_top))

            if args.imagePrintStatistics is True:
                text_from_top += 50

        if args.imagePrintStatistics is True and args.detectBody is True:
            frame = draw_nn_result(frame, nn_result_text, (10, text_from_top), nn_color)
            text_from_top += 50

        if args.detectBody is True:
            frame = get_human_image(frame, humans_detected[0], args.framework)

        if args.showOutput is True:
            cv2.imshow("Result", frame)

        new_time = time.time()
        time_dif = new_time - old_time
        old_time = new_time
        time_sum += time_dif

        if args.printContinuosStatistics is True:
            print("Ellapsed time: " + str(time_dif) + "\n\n")

        if args.recordVideo is True:
            out.write(frame)

        if cv2.waitKey(40) == 27:
            break

    cap.release()

    if args.printFinalStatistics is True:
        not_skipped_counter = counter - skipped_frame_counter
        print("\n\nFINAL STATISTICS:")
        print("USED FRAMEWORK:        \t\t" + args.framework)
        print("RESOLUTION (original): \t\t" + str(frame_width) + "x" + str(frame_height))
        print("RESOLUTION (proceeded):\t\t" + str(size[0]) + "x" + str(size[1]))
        print("TOTAL TIME:            \t\t" + str(time_sum))
        print("AVERAGE TIME:          \t\t" + str(time_sum/not_skipped_counter))
        print("TOTAL FRAMES:          \t\t" + str(not_skipped_counter))
        print("HUMANS DETECTED:       \t\t" + str(people_counter))
        print("HUMANS DETECTED [%]:   \t\t" + str(people_counter / not_skipped_counter * 100) + "%")

        print("\n-----NN ANALYZATOR RESULT-----")
        print("STEERING:              \t\t" + str(nn_result_counter.steering))
        print("STEERING [%]:          \t\t" + str(nn_result_counter.steering / not_skipped_counter * 100) + "%")
        print("SHIFTING:              \t\t" + str(nn_result_counter.shifting))
        print("SHIFTING [%]:          \t\t" + str(nn_result_counter.shifting / not_skipped_counter * 100) + "%")
        print("WRONG:              \t\t" + str(nn_result_counter.wrong))
        print("WRONG [%]:          \t\t" + str(nn_result_counter.wrong / not_skipped_counter * 100) + "%")

        print("\n-----NN ANALYZATOR RESULT - WITH FILTER-----")
        print("STEERING:              \t\t" + str(nn_result_filtered_counter.steering))
        print("STEERING [%]:          \t\t" + str(nn_result_filtered_counter.steering / not_skipped_counter * 100) + "%")
        print("SHIFTING:              \t\t" + str(nn_result_filtered_counter.shifting))
        print("SHIFTING [%]:          \t\t" + str(nn_result_filtered_counter.shifting / not_skipped_counter * 100) + "%")
        print("WRONG:              \t\t" + str(nn_result_filtered_counter.wrong))
        print("WRONG [%]:          \t\t" + str(nn_result_filtered_counter.wrong / not_skipped_counter * 100) + "%")

    if args.recordVideo is True:
        print("Saving Video to: " + args.videoOutput)
        out.release()

    cv2.destroyAllWindows()
