import cv2
import torch
import numpy as np

def joint_to_heatmap(heatmap_size: (int, int), image_size: (int, int), joints_3d: np.ndarray, sigma=2):
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
        x, y = int(joints[j, 0] * scale_x), int(joints[j, 1] * scale_y)

        # checks the bounds of the points
        if joint_masks[j] > 0:
            heatmaps[j, y, x] = 255.0
            heatmaps[j, :, :] = cv2.GaussianBlur(heatmaps[j, :, :], (0, 0), sigma)

            combined_heatmap[int(joints[j, 1]), int(joints[j, 0])] = 255.0
        
        combined_heatmap = cv2.GaussianBlur(combined_heatmap, (0, 0), sigma)
    
    return heatmaps, joint_masks, combined_heatmap