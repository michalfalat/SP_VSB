import numpy as np
import cv2


def get_coordinates(frame, human, poseType):
    npimg = np.copy(frame)
    image_h, image_w = npimg.shape[:2]
    if(poseType == "OP"):
        head = (int(human[0][0]), int((human[0][1] + human[1][1])/2))
        neck = (int(human[1][0]), int(human[1][1]))
        shoulder_left = (int(human[5][0]), int(human[5][1]))
        shoulder_right = (int(human[2][0]), int(human[2][1]))
        hip_left = (int(human[11][0]), int(human[11][1]))
        hip_right = (int(human[8][0]), int(human[8][1]))
        elbow_left = (int(human[6][0]), int(human[6][1]))
        elbow_right = (int(human[3][0]), int(human[3][1]))
        wrist_left = (int(human[7][0]), int(human[7][1]))
        wrist_right = (int(human[4][0]), int(human[4][1]))
        knee_left = (int(human[12][0]), int(human[12][1]))
        knee_right = (int(human[9][0]), int(human[9][1]))
        return [head, neck, shoulder_left, shoulder_right, hip_left, hip_right, elbow_left, elbow_right, wrist_left, wrist_right, knee_left, knee_right]

    elif (poseType == "TF"):
        if(len(human.body_parts) == 0):
            return [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
        try:
            head = (int(human.body_parts[0].x * image_w + 0.5), int(human.body_parts[0].y * image_h + 0.5))
        except:
            head = (0, 0)

        try:
            neck = (int(human.body_parts[1].x * image_w + 0.5), int(human.body_parts[1].y * image_h + 0.5))
        except:
            neck = (0, 0)

        try:
            shoulder_left = (int(human.body_parts[5].x * image_w + 0.5), int(human.body_parts[5].y * image_h + 0.5))
        except:
            shoulder_left = (0, 0)

        try:
            shoulder_right = (int(human.body_parts[2].x * image_w + 0.5), int(human.body_parts[2].y * image_h + 0.5))
        except:
            shoulder_right = (0, 0)

        try:
            hip_left = (int(human.body_parts[11].x * image_w + 0.5), int(human.body_parts[11].y * image_h + 0.5))
        except:
            hip_left = (0, 0)

        try:
            hip_right = (int(human.body_parts[8].x * image_w + 0.5), int(human.body_parts[8].y * image_h + 0.5))
        except:
            hip_right = (0, 0)

        try:
            elbow_left = (int(human.body_parts[6].x * image_w + 0.5), int(human.body_parts[6].y * image_h + 0.5))
        except:
            elbow_left = (0, 0)

        try:
            elbow_right = (int(human.body_parts[3].x * image_w + 0.5), int(human.body_parts[3].y * image_h + 0.5))
        except:
            elbow_right = (0, 0)

        try:
            wrist_left = (int(human.body_parts[7].x * image_w + 0.5), int(human.body_parts[7].y * image_h + 0.5))
        except:
            wrist_left = (0, 0)

        try:
            wrist_right = (int(human.body_parts[4].x * image_w + 0.5), int(human.body_parts[4].y * image_h + 0.5))
        except:
            wrist_right = (0, 0)

        try:
            knee_left = (int(human.body_parts[12].x * image_w + 0.5), int(human.body_parts[12].y * image_h + 0.5))
        except:
            knee_left = (0, 0)

        try:
            knee_right = (int(human.body_parts[9].x * image_w + 0.5), int(human.body_parts[9].y * image_h + 0.5))
        except:
            knee_right = (0, 0)

        return [head, neck, shoulder_left, shoulder_right, hip_left, hip_right, elbow_left, elbow_right, wrist_left, wrist_right, knee_left, knee_right]


def get_human_image(frame, human, poseType, for_nn=False):
    if(for_nn):
        frameCopy, coordinates = get_nn_image(frame, human, poseType)
        color = (255, 255, 255)
    else:
        coordinates = get_coordinates(frame, human, poseType)
        frameCopy = frame.copy()
        color = (220, 220, 90)
    lineWidth = 8
    headWidth = 30

    cv2.circle(frameCopy, coordinates[0], 15, color, headWidth)

    if coordinates_exists(coordinates[0], coordinates[1]):  # head to neck
        cv2.line(frameCopy, coordinates[0], coordinates[1], color, lineWidth)

    if coordinates_exists(coordinates[1], coordinates[2]):  # head to left shoulder
        cv2.line(frameCopy, coordinates[1], coordinates[2], color, lineWidth)

    if coordinates_exists(coordinates[1], coordinates[3]):  # head to right shoulder
        cv2.line(frameCopy, coordinates[1], coordinates[3], color, lineWidth)

    if coordinates_exists(coordinates[1], coordinates[4]):  # neck to left hip
        cv2.line(frameCopy, coordinates[1], coordinates[4], color, lineWidth)

    if coordinates_exists(coordinates[1], coordinates[5]):  # neck to right hip
        cv2.line(frameCopy, coordinates[1], coordinates[5], color, lineWidth)

    if coordinates_exists(coordinates[4], coordinates[10]):  # left hip to left knee
        cv2.line(frameCopy, coordinates[4], coordinates[10], color, lineWidth)

    if coordinates_exists(coordinates[5], coordinates[11]):  # right hip to right knee
        cv2.line(frameCopy, coordinates[5], coordinates[11], color, lineWidth)

    if coordinates_exists(coordinates[2], coordinates[6]):   # left shoulder to left elbow
        cv2.line(frameCopy, coordinates[2], coordinates[6], color, lineWidth)

    if coordinates_exists(coordinates[3], coordinates[7]):   # right shoulder to right elbow
        cv2.line(frameCopy, coordinates[3], coordinates[7], color, lineWidth)

    if coordinates_exists(coordinates[6], coordinates[8]):  # left elbow to left wrist
        cv2.line(frameCopy, coordinates[6], coordinates[8], color, lineWidth)

    if coordinates_exists(coordinates[7], coordinates[9]):  # head to neck
        cv2.line(frameCopy, coordinates[7], coordinates[9], color, lineWidth)  # right elbow to right wrist

    # cv2.imshow("lines - " + poseType, frameCopy)
    # cv2.waitKey(0)
    # print( coordinates)
    return frameCopy


def coordinates_exists(coord1, coord2):
    return coord1 != None and coord2 != None and coord1[0] > 0 and coord1[1] > 0 and coord2[0] > 0 and coord2[1] > 0


def get_nn_image(frame, human, poseType):
    coordinates = get_coordinates(frame, human, poseType)
    min_x = int(coordinates[0][0])
    max_x = int(coordinates[0][0])
    min_y = int(coordinates[0][1])
    max_y = int(coordinates[0][1])
    for i in range(len(coordinates)):
        if(coordinates[i][0] > 0 and coordinates[i][1] > 0):
            if(coordinates[i][0] < min_x):
                min_x = int(coordinates[i][0])
            elif(coordinates[i][0] > max_x):
                max_x = int(coordinates[i][0])

            if(coordinates[i][1] < min_y):
                min_y = int(coordinates[i][1])
            elif(coordinates[i][1] > max_y):
                max_y = int(coordinates[i][1])
    # print(min_x)
    # print(min_y)
    # print(max_x)
    # print(max_y)
    width = max_x - min_x
    height = max_y - min_y

    # coordinates = list(coordinates)

    offset = 30

    for i in range(len(coordinates)):
        coordinates[i] = list(coordinates[i])
        coordinates[i][0] = int(coordinates[i][0] - min_x+offset)
        coordinates[i][1] = int(coordinates[i][1] - min_y+offset)
        coordinates[i] = tuple(coordinates[i])

    # print(width)
    # print(height)
    # crop_img = frame[min_y-offset:min_y+height+offset, min_x-offset:min_x+width+offset].copy()
    crop_img = np.zeros((height+2*offset, width+2*offset, 1), dtype="uint8")

    # cv2.imshow("cropped", crop_img)

    return crop_img, coordinates
