import cv2
import numpy as np
import math

def detect_head_orientation(frame, shape):

    size = frame.shape
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
        shape[33],     # Nose tip
        shape[8],     # Chin
        shape[36],     # Left eye left corner
        shape[45],     # Right eye right corne
        shape[48],     # Left Mouth corner
        shape[54]      # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner

    ])


    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double"
    )

    # print ("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # print ("Rotation Vector:\n {0}".format(rotation_vector))
    # print ("Translation Vector:\n {0}".format(translation_vector))


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)


    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    angle = int(round(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))))

    return (p1, p2, angle)

    # draw_head_orientation(frame, p1, p2, angle)



def draw_head_orientation(frame, p_src, p_dst, angle_dst):
    angle1 = 2
    angle2 = 100
    length = 110
    result = False

    print(angle_dst)

    if(angle_dst < angle1 or angle_dst > angle2):
        cv2.line(frame, p_src, p_dst, (10, 0, 255), 3)
    else:
        cv2.line(frame, p_src, p_dst, (255, 0, 0), 2)
        result = True

    p1_x = int(round(p_src[0] + length * np.cos(angle1 * np.pi / 180.0)))
    p1_y = int(round(p_src[1] + length * np.sin(angle1 * np.pi / 180.0)))

    p2_x = int(round(p_src[0] + length * np.cos(angle2 * np.pi / 180.0)))
    p2_y = int(round(p_src[1] + length * np.sin(angle2 * np.pi / 180.0)))

    cv2.line(frame, p_src, (p1_x, p1_y), (0, 127, 255), 1)
    cv2.line(frame, p_src, (p2_x, p2_y), (0, 127, 255), 1)

    return result


def print_head_orientation_info(frame, res, pos):
    if(res):
        cv2.putText(frame, "HEAD POSITION: OK ", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 10), 3)
    else:
        cv2.putText(frame, "HEAD POSITION: OUT OF RANGE", pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame
