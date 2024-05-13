import cv2
import torch
import numpy as np

def joint_to_heatmap(heatmap_size, image_size, joints_3d, sigma=2):
    """
    Generates a heatmap from the inputted joints_3d (using gaussians)
    
    Parameters
    ----------
    heatmap_size: (int, int)
        the size of the desired heat map (H, W)
    image_size: (int, int)
        the size of the image (H, W)
    joints_3d: np.ndarray
        an array of the joint points (size of 26 x 3 x 2)
    sigma: int
        the sigma value for the gaussians
    
    Returns
    -------
    np.ndarray
        the heatmap of the joint points (size of 26 x height x width)
    np.ndarray
        the masks of the key points taken from the visibility of the join in the array
    """
    num_joints = len(joints_3d)
    joints = joints_3d[:, :, 0]
    visibility = joints_3d[:, :, 1]
    joint_masks = np.ones((num_joints, 1), dtype=np.float32)
    joint_masks[:, 0] = visibility[:, 0]
    
    heatmaps = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    combined_heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)

    scale_x, scale_y = heatmap_size[1] / image_size[1], heatmap_size[0] / image_size[0]

    for j in range(num_joints):
        # scales the joint key points
        x = max(min(int(joints[j, 0]), image_size[1]-1), 0)
        y = max(min(int(joints[j, 1]), image_size[0]-1), 0)
        scaled_x = max(min(int(joints[j, 0] * scale_x), heatmap_size[1]-1), 0)
        scaled_y = max(min(int(joints[j, 1] * scale_y), heatmap_size[0]-1), 0)

        # checks if the point is visible
        if joint_masks[j] > 0:
            heatmaps[j, scaled_y, scaled_x] = 1.0
            heatmaps[j, :, :] = cv2.GaussianBlur(heatmaps[j, :, :], (0, 0), sigma)

            max_val = heatmaps[j, :, :].max()
            if max_val > 0:
                heatmaps[j, :, :] /= max_val

            combined_heatmap[y, x] = 1.0
        
        combined_heatmap = cv2.GaussianBlur(combined_heatmap, (0, 0), sigma)
        max_val = combined_heatmap.max()
        if max_val > 0:
            combined_heatmap /= max_val
    
    return heatmaps, np.expand_dims(joint_masks, -1), combined_heatmap


def heatmap_to_joint(heatmaps, image_size):
    """
    Converts a heatmap of joint points back to joint coordinates.
    
    Parameters
    ----------
    heatmaps : np.array
        The heatmap array of joint points (size of 26 x height x width)
    image_size : (int, int)
        The size of the image (H, W)
    
    Returns
    -------
    np.array
        The array of joint coordinates (size of 26 x 3 x 2)
    """
    num_joints = heatmaps.shape[0]
    heatmap_size = (heatmaps.shape[1], heatmaps.shape[2])
    scale_x, scale_y = image_size[1] / heatmap_size[1], image_size[0] / heatmap_size[0]
    joints_3d = np.zeros((num_joints, 3, 2), dtype=np.float32)

    for j in range(num_joints):
        # find the index of the maximum value in the heatmap
        y, x = np.unravel_index(np.argmax(heatmaps[j]), heatmaps[j].shape)
        # convert heatmap coordinates back to original image coordinates
        real_x = x * scale_x
        real_y = y * scale_y

        # Assume visibility is 1 since heatmap exists
        joints_3d[j, :, 0] = [real_x, real_y, 0]
        joints_3d[j, :, 1] = [1, 1, 1]
    return joints_3d