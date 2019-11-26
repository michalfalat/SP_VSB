import cv2
import numpy as np
import math


def detect_seat_belt(frame, human):
    crop_img, crop_offset = crop_seatbelt_area(frame, human)
    return detect_seatbelt_lines(crop_img, frame), crop_offset


def crop_seatbelt_area(frame, human):
    npimg = np.copy(frame)
    image_h, image_w = npimg.shape[:2]
    if(len(human.body_parts) < 12):
        return None, (0, 0)
    shoulder1 = human.body_parts[2]
    shoulder2 = human.body_parts[5]
    hip1 = human.body_parts[8]
    hip2 = human.body_parts[11]

    pos_shoulder1 = (int(shoulder1.x * image_w + 0.5), int(shoulder1.y * image_h + 0.5))
    pos_shoulder2 = (int(shoulder2.x * image_w + 0.5), int(shoulder2.y * image_h + 0.5))
    pos_hip1 = (int(hip1.x * image_w + 0.5), int(hip1.y * image_h + 0.5))
    pos_hip2 = (int(hip2.x * image_w + 0.5), int(hip2.y * image_h + 0.5))
    crop_diameter = 20
    width = abs(pos_shoulder2[0] - pos_shoulder1[0])
    height = abs(pos_hip2[1] - pos_shoulder1[1])

    crop_img = frame[pos_shoulder1[1]+crop_diameter:pos_shoulder1[1]+height - crop_diameter, pos_shoulder1[0]+crop_diameter:pos_shoulder1[0]+width - crop_diameter]
    crop_offset = (pos_shoulder1[0]+crop_diameter, pos_shoulder1[1]+crop_diameter)

    return crop_img, crop_offset


def detect_seatbelt_lines(frame, originalFrame):
    if frame is None or not frame.any():
        return []
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    border = cv2.borderInterpolate(0, 1, cv2.BORDER_DEFAULT)
    sobel_x = cv2.Sobel(frame_gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0,  borderType=border)
    sobel_y = cv2.Sobel(frame_gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=border)

    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)

    weighted = cv2.addWeighted(abs_x, 0.5,  abs_y, 0.5, 0)
    retval, thresholded = cv2.threshold(weighted, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('thresholded', thresholded)
    cv2.imshow('frame', frame)

    coef_y, coef_x = originalFrame.shape[:2]
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = (int(coef_y / 50))  # 45  # minimum number of votes (intersections in Hough grid cell)
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
            theta = math.atan(dy/dx)
            theta *= 180 / np.pi
            # print(str(theta) + ' degrees')
            if(abs(theta) > 20 and abs(theta) < 70):
               filtered_lines.append(line)
            # else:
               # cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # print(len(filtered_lines))
    return filtered_lines


def draw_seatbelt_lines(frame, lines, crop_offset):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(frame, (crop_offset[0] + x1, crop_offset[1] + y1), (crop_offset[0] + x2, crop_offset[1] + y2), (0, 0, 255), 3)
    return frame


def print_seatbelt_info(frame, seatbelt_lines, pos):
    if(len(seatbelt_lines) > 3):
        cv2.putText(frame, "SEATBELT ON", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 10), 3)
    else:
        cv2.putText(frame, "SEATBELT OFF", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame
