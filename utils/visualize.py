import cv2
import numpy as np

l_pair = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
    (17, 18), (18, 19), (19, 11), (19, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
    (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
    (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
    (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
    (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
    (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
    (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
    (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
    (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
    (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
]

p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot

line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
              (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]


def vis_joints(image: np.ndarray, keypoints: np.ndarray, output_path: str):
    """
    Saves an image with the drawn on keypoints and corners of the bounding box

    Taken from
        https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/vis.py

    Parameters
    ----------
    image: np.ndarray
        the image we want to save
    keypoints: np.ndarray
        the keypoints to draw on with a size of num_keypoints x 3 x 2
    output_path: str
        the path to save the image
    """
    part_line = {}
    kp_x = keypoints[:, 0, 0]
    kp_y = keypoints[:, 1, 0]
    kp_scores = keypoints[:, 0, 1]

    # Draw keypoints
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= 0.6:
            continue
        cor_x, cor_y = int(kp_x[n]), int(kp_y[n])
        part_line[n] = (int(cor_x), int(cor_y))
        if n < len(p_color):
            cv2.circle(image, (int(cor_x), int(cor_y)), 2, p_color[n], -1)
        else:
            cv2.circle(image, (int(cor_x), int(cor_y)), 1, (255,255,255), 2)

    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            if i < len(line_color):
                cv2.line(image, start_xy, end_xy, line_color[i], 2)
            else:
                cv2.line(image, start_xy, end_xy, (255,255,255), 1)

    cv2.imwrite(output_path, image)


def vis_bbox(image: np.ndarray, bbox: np.ndarray, output_path: str):
    """
    Saves an image with the drawn on bounding box

    Parameters
    ----------
    image: np.ndarray
        the image we want to save
    bbox: np.ndarray
        the bounding box (x_min, y_min, x_max, y_max)
    output_path: str
        the path to save the image
    """
    # Draw Bounding Box Points
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    px1, py1 = xmin, ymin
    px2, py2 = xmin, ymax
    px3, py3 = xmax, ymin
    px4, py4 = xmax, ymax
    cv2.circle(image, (int(px1), int(py1)), 5, (255, 255, 255), 2)
    cv2.circle(image, (int(px2), int(py2)), 5, (255, 255, 255), 2)
    cv2.circle(image, (int(px3), int(py3)), 5, (255, 255, 255), 2)
    cv2.circle(image, (int(px4), int(py4)), 5, (255, 255, 255), 2)

    # Draw the Bound Box Lines
    cv2.line(image, (int(px1), int(py1)), (int(px2), int(py2)), (255, 255, 255), 2)
    cv2.line(image, (int(px1), int(py1)), (int(px3), int(py3)), (255, 255, 255), 2)
    cv2.line(image, (int(px2), int(py2)), (int(px4), int(py4)), (255, 255, 255), 2)
    cv2.line(image, (int(px3), int(py3)), (int(px4), int(py4)), (255, 255, 255), 2)

    cv2.imwrite(output_path, image)


def vis_heatmap(image: np.ndarray, heatmap: np.ndarray, output_path):
    """
    Saves the image with the heatmap superimposed over it

    Taken from
        https://stackoverflow.com/questions/46020894/superimpose-heatmap-on-a-base-image-opencv-python

    Parameters
    ----------
    image: np.ndarray
        the image we want to save
    heatmap: np.ndarray
        the heatmap corresponding to the joint points
    output_path: str
        the path to save the image
    """
    print(heatmap)
    heatmap = heatmap.astype('uint8')
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)

    cv2.imwrite(output_path, super_imposed_img)
