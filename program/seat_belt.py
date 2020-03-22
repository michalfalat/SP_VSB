import cv2
import numpy as np
import math
from pose_unifier import get_coordinates


def detect_seat_belt(frame, human, framework):
    crop_img, crop_offset = crop_seatbelt_area(frame, human, framework)
    return detect_seatbelt_lines(crop_img, frame), crop_offset


def crop_seatbelt_area(frame, human, framework):
    npimg = np.copy(frame)
    image_h, image_w = npimg.shape[:2]
    coordinates = get_coordinates(frame, human, framework)
    # if len(human.body_parts) == 0:
    #     return None, (0, 0)
    try:
        # shoulder_left = human.body_parts[2]
        # shoulder_right = human.body_parts[5]
        # hip_left = human.body_parts[8]
        # hip_right = human.body_parts[11]

        pos_shoulder_left = coordinates[2]
        pos_shoulder_right = coordinates[3]
        pos_hip_left = coordinates[4]
        pos_hip_right = coordinates[5]

        # pos_shoulder_left = (int(shoulder_left.x * image_w + 0.5), int(shoulder_left.y * image_h + 0.5))
        # pos_shoulder_right = (int(shoulder_right .x * image_w + 0.5), int(shoulder_right .y * image_h + 0.5))
        # pos_hip_left = (int(hip_left.x * image_w + 0.5), int(hip_left.y * image_h + 0.5))
        # pos_hip_right = (int(hip_right .x * image_w + 0.5), int(hip_right .y * image_h + 0.5))
        crop_diameter = 10
        crop_diameter_left = 60
        width = abs(pos_shoulder_right[0] - pos_shoulder_left[0])
        height = abs(pos_hip_right[1] - pos_shoulder_left[1])

        more_left_shoulder = pos_shoulder_left if pos_shoulder_left[0] < pos_shoulder_right[0] else  pos_shoulder_right

        crop_img = frame[more_left_shoulder[1]+crop_diameter:more_left_shoulder[1]+height - crop_diameter, more_left_shoulder[0]+crop_diameter_left:more_left_shoulder[0]+width - crop_diameter]
        crop_offset = (more_left_shoulder[0]+crop_diameter_left, more_left_shoulder[1]+crop_diameter)

        return crop_img, crop_offset
    except:
        return None, (0, 0)


def detect_seatbelt_lines(frame, originalFrame):
    if frame is None or not frame.any():
        return []

    alpha = 1.3 # calc from image darkness

    # cv2.imshow('frame before ', frame)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

    # cv2.imshow('frame after ', frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_DEFAULT)
    sobel_x = cv2.Sobel(frame_gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0,  borderType=border)
    sobel_y = cv2.Sobel(frame_gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=border)

    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)

    weighted = cv2.addWeighted(abs_x, 0.5,  abs_y, 0.5, 0)
    retval, thresholded = cv2.threshold(weighted, 100, 255, cv2.THRESH_BINARY)


    # cv2.imshow('thresholded', thresholded)
    # cv2.imshow('frame', frame)

    coef_y, coef_x = originalFrame.shape[:2]
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = (int(coef_y / 55))  # 45  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = coef_y / 20  # 50  minimum number of pixels making up a line
    max_line_gap = coef_y / 14  # 70  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(thresholded, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is None:
        return []

    seatbelt_lines = filter_seatbelt_lines(lines)
    return seatbelt_lines


def filter_seatbelt_lines(lines):
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            dy = y2 - y1
            dx = x2 - x1
            if dx == 0:
                continue
            theta = math.atan(dy/dx)
            theta *= 180 / np.pi
            if(abs(theta) > 20 and abs(theta) < 80):
               filtered_lines.append(line)
            # else:
               # cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return filtered_lines


def draw_seatbelt_lines(frame, lines, crop_offset):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (crop_offset[0] + x1, crop_offset[1] + y1), (crop_offset[0] + x2, crop_offset[1] + y2), (0, 0, 255), 3)
    return frame


def draw_seatbelt_info(frame, seatbelt_lines, pos):
    if len(seatbelt_lines) > 2:
        cv2.putText(frame, "SEATBELT ON", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 10), 3)
    else:
        cv2.putText(frame, "SEATBELT OFF", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame
