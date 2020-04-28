import cv2
import numpy as np
import math

def detect_head_orientation(frame, shape, x_offset, y_offset):

    #2D image points of face
    input_image = frame.shape
    image_points = np.array([
        shape[33],     # Nose tip
        shape[8],      # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corne
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points_3d = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner

    ])

    # camera matrix calcualtion
    focal_length = input_image[1]
    center = (input_image[1]/2, input_image[0]/2)
    camera_matrix = np.array(
        [
            [focal_length,  0,            center[0]],
            [0,             focal_length, center[1]],
            [0,             0,            1]
        ], dtype="double"
    )

    # define lens disortion
    dist_coeffs = np.zeros((4, 1))

    # estimates the object pose given a set of object points, their corresponding image projections, as well as the camera matrix and the distortion coefficients
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # print ("Rotation Vector:\n {0}".format(rotation_vector))
    # print ("Translation Vector:\n {0}".format(translation_vector))

    nose_end_point_2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(x_offset + image_points[0][0]), y_offset + int(image_points[0][1]))
    p2 = (int(x_offset + nose_end_point_2D[0][0][0]), y_offset + int(nose_end_point_2D[0][0][1]))
    angle = int(round(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))))
    return (p1, p2, angle)

def draw_head_orientation(frame, p_src, p_dst, angle_dst, args):
    angle1 = 2
    angle2 = 100
    length = 110
    result = False
    p_dst_filtered = filter_outbound_lines(frame, p_src, p_dst)

    if args.detectHeadLimit is False:
        cv2.line(frame, p_src, p_dst_filtered, (255, 0, 20), 4)
        result = True
    elif(angle_dst < angle1 or angle_dst > angle2):
        cv2.line(frame, p_src, p_dst_filtered, (10, 0, 255), 4)
    else:
        cv2.line(frame, p_src, p_dst_filtered, (255, 0, 0), 3)
        result = True

    p1_x = int(round(p_src[0] + length * np.cos(angle1 * np.pi / 180.0)))
    p1_y = int(round(p_src[1] + length * np.sin(angle1 * np.pi / 180.0)))

    p2_x = int(round(p_src[0] + length * np.cos(angle2 * np.pi / 180.0)))
    p2_y = int(round(p_src[1] + length * np.sin(angle2 * np.pi / 180.0)))

    if args.detectHeadLimit is True:
        cv2.line(frame, p_src, (p1_x, p1_y), (0, 127, 255), 2)
        cv2.line(frame, p_src, (p2_x, p2_y), (0, 127, 255), 2)

    return result


def filter_outbound_lines(frame, p_src, p_dst):
    image_h, image_w = frame.shape[:2]
    if p_dst is None or p_dst[0] < 0 or p_dst[1] < 0 or p_dst[0] > image_w or p_dst[1] > image_h:
        return p_src
    else:
        return p_dst


def draw_head_orientation_info(frame, result, label_position):
    # skip head orientation
    if result == -1:
        cv2.putText(frame, "HEAD POSITION: SKIP ", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 127), 3)

    # head is in given range
    elif result:
        cv2.putText(frame, "HEAD POSITION: OK ", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 10), 3)

    # head is out of range
    else:
        cv2.putText(frame, "HEAD POSITION: OUT OF RANGE", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame
