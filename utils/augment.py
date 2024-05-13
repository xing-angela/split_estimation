import cv2
import random
import numpy as np

######################### Augmentation Factors #########################
ROTATION_FACTOR = 80
FLIP = True
########################################################################

def transform_point(point, transform_matrix):
    """
    Transforms a point according the the transform matrix

    Parameters
    ----------
    point: np.array
        the point to transform (x, y) coordinate
    transform_matrix: np.array
        the transformation matrix

    Returns
    -------
    np.array
        the transformed (x, y) coordinate
    """
    homogenous_point = np.array([point[0], point[1], 1.]).T
    homogenous_point = np.dot(transform_matrix, homogenous_point)
    return homogenous_point[:2]

def augment_data(img, label, joint_pairs, exists_bbox):
    """
    Augments the data (scaling, rotating, and flipping)

    Parameters
    ----------
    img: np.array
        the image
    label: dict
        the labels of the for the image (bbox, width, height, joints_3d)
    joint_pairs: list
        the pairs of indices that represent a joint
    exists_bbox: bool
        boolean indicating if the bbox exists

    Returns
    -------
    (np.array, dict)
        image and annotations of the sample at the index
        annotation = {
            bbox, 
            width,
            height,
            joints_3d
        }
    """
    h, w, _ = img.shape
    joints_3d = label["joints_3d"]
    joint_shape = joints_3d.shape

    if exists_bbox:
        bbox = label["bbox"]

    # flips the data
    if FLIP and random.random() < 0.5:
        # flips the image
        img = cv2.flip(img, 1)

        # flips the joint key points
        flip_joints = joints_3d.copy()
        flip_joints[:, 0, 0] = w - flip_joints[:, 0, 0]
        for pair in joint_pairs:
            flip_joints[pair[0], :, 0], flip_joints[pair[1], :, 0] = \
                flip_joints[pair[1], :, 0], flip_joints[pair[0], :, 0].copy()
            flip_joints[pair[0], :, 1], flip_joints[pair[1], :, 1] = \
                flip_joints[pair[1], :, 1], flip_joints[pair[0], :, 1].copy()
        flip_joints[:, :, 0] *= flip_joints[:, :, 1]
        joints_3d = flip_joints

        # flips the bbox
        if exists_bbox:
            orig_bbox = bbox.copy()
            bbox_w = orig_bbox[2] - orig_bbox[0]
            bbox[2] = w - orig_bbox[0]
            bbox[0] = bbox[2] - bbox_w

    # rotates the data
    if random.random() < 0.5:
        rot_angle = random.uniform(-ROTATION_FACTOR, ROTATION_FACTOR)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rot_angle, 1.0)

        # rotates the image
        img = cv2.warpAffine(img, rotation_matrix, (w, h))
        
        # rotates the keypoints
        for i in range(len(joints_3d)):
            if joints_3d[i, 0, 1] > 0.0:
                joints_3d[i, 0:2, 0] = transform_point(joints_3d[i, 0:2, 0], rotation_matrix)
        
        # rotates the bbox
        if exists_bbox:
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_points = np.array([[x_min, y_min],
                                    [x_min, y_max],
                                    [x_max, y_min],
                                    [x_max, y_max]])
            for i in range(len(bbox_points)):
                bbox_points[i] = transform_point(bbox_points[i], rotation_matrix)

            x_min = np.clip(min(bbox_points[:, 0]), 0, w)
            y_min = np.clip(min(bbox_points[:, 1]), 0, h)
            x_max = np.clip(max(bbox_points[:, 0]), 0, w)
            y_max = np.clip(max(bbox_points[:, 1]), 0, h)
            
            bbox = [x_min, y_min, x_max, y_max]

    new_label = {
        'width': w,
        'height': h,
        'joints_3d': joints_3d
    }
    if exists_bbox:
        new_label['bbox'] = bbox

    return img, new_label

