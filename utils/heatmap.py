import torch
import numpy as np

def joint_to_heatmap(heatmap_size: (int, int), image_size: (int, int), joints_3d: np.ndarray):
    """
    Generates a heatmap from the inputted joints_3d (using gaussians)
    
    Taken from the generate_target function:
        https://github.com/microsoft/human-pose-estimation.pytorch/blob/49f3f4458c9d5917c75c37a6db48c6a0d7cd89a1/lib/dataset/JointsDataset.py#L169
    
    Parameters
    ----------
    heatmap_size: (int, int)
        the size of the desired heat map (H, W)
    image_size: (int, int)
        the size of the image (H, W)
    joints_3d: np.ndarray
        an array of the joint points (size of 26 x 3 x 2)
    
    Returns
    -------
    np.ndarray
        the heatmap of the joint points (size of 26 x height x width)
    """
    num_joints = len(joints_3d)
    joints = joints_3d[:, :, 0]
    visibility = joints_3d[:, :, 1]
    
    joint_mask = np.ones((num_joints, 1), dtype=np.float32)
    joint_mask[:, 0] = visibility[:, 0]

    heatmaps = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)

    sigma = 2
    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        stride_x = image_size[0] / heatmap_size[0]
        stride_y = image_size[1] / heatmap_size[1]
        mu_x = int(joints[joint_id][0] / stride_x + 0.5)
        mu_y = int(joints[joint_id][1] / stride_y + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            joint_mask[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # print(f"gx: {g_x}, g_y: {g_y}")
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
        # print(f"img_x: {img_x}, img_y: {img_y}")

        v = joint_mask[joint_id]
        if v > 0.5:
            heatmaps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps