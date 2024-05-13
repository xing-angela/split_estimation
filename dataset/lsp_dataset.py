import os
import cv2
import torch
import scipy.io
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from utils.augment import augment_data
from utils.heatmap import joint_to_heatmap
from utils.visualize import vis_joints, vis_bbox, vis_heatmap

JOINT_PAIRS = [[1, 0], [2, 1], [2, 3], [3, 4], [4, 5], [7, 6], [8, 7], [9, 10], 
               [10, 11], [12, 3], [12, 8], [12, 9], [13, 12]]

class LSPDataset(Dataset):
    """
    Leeds Sports Dataset (LSP) dataset is an image dataset where each
    image has been annotated with 14 body key points
    """

    num_joints = 14

    def __init__(self, data_dir, img_dir, img_size):
        """
        Initializes the LSP Dataset

        https://github.com/axelcarlier/lsp

        Parameters
        ----------
        data_dir: str
            path of the data directory
        img_dir: str
            path of image directory within the data directory
        img_size: str
            target size of images after scaling
        """

        self.data_dir = data_dir
        self.img_dir = img_dir
        self.img_size = img_size

        # loads the images and labels
        data = self.load_data()
        self.images = data[0]
        self.joints_3d = data[1]

    def __getitem__(self, index):
        """
        Returns the image and annotations of the dataset at the specific index

        Parameters
        ----------
        index : int
            index of a sample

        Returns
        -------
        (torch.Tensor, dict)
            image and annotations of the sample at the index
            annotation = {
                bbox, 
                width, 
                height, 
                heatmaps, 
                masks
            }
        """
        img_path = os.path.join(self.data_dir, self.img_dir, self.images[index])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        joints_3d = self.joints_3d[index]
        
        img_h, img_w, _ = img.shape
        target_w, target_h = self.img_size
        scale_x, scale_y = target_w / img_w, target_h / img_h
        
        # scales the image to the correct size
        scaled_img = cv2.resize(img, (target_w, target_h))

        # scales the joint points
        scaled_joints = joints_3d.copy()
        for i in range(self.num_joints):
            if scaled_joints[i, 0, 1] > 0.0:
                scaled_joints[i, 0, 0] = scaled_joints[i, 0, 0] * scale_x
                scaled_joints[i, 1, 0] = scaled_joints[i, 1, 0] * scale_y

        # further augments the data
        scale_label = {
            'width': target_w,
            'height': target_h,
            'joints_3d': scaled_joints
        }
        aug_img, aug_label = augment_data(scaled_img, scale_label, JOINT_PAIRS, False)
        aug_h, aug_w = aug_label['height'], aug_label['width']
        aug_joints = aug_label['joints_3d']

        # ensures the augmented image is the right shape and converts to tensor
        aug_img = np.transpose(aug_img, (2, 0, 1))  # C x H x W
        aug_img = torch.from_numpy(aug_img).float()
        if aug_img.max() > 1:
            aug_img /= 255

        # gets the heatmap and the mask of the joints
        heatmaps, masks, combined_heatmap = joint_to_heatmap((64, 48), (aug_h, aug_w), aug_joints)
        # heatmaps, masks, combined_heatmap = joint_to_heatmap((32, 24), (aug_h, aug_w), aug_joints, sigma=3)

        # visualization
        # if flip and rotate:
        #     print("flip and rotate")
        # outfolder = "/users/axing2/data/users/axing2/split_estimation/test_lsp_data"
        # if not os.path.exists(outfolder):
        #     os.makedirs(outfolder)
        # cv2.imwrite(os.path.join(outfolder, "non_aug.jpg"), scaled_img)
        # cv2.imwrite(os.path.join(outfolder, "original.jpg"), aug_img)
        # vis_joints(aug_img.copy(), aug_joints, JOINT_PAIRS, os.path.join(outfolder, "joints.jpg"))

        aug_label = {
            'width': aug_w,
            'height': aug_h,
            'heatmaps': torch.Tensor(heatmaps),
            'masks': torch.Tensor(masks)
        }

        return aug_img, aug_label

    def __len__(self):
        """
        Returns the number of samples

        Returns
        -------
        int
            number of samples
        """
        return len(self.images)

    def load_data(self):
        """
        Loads the images from the image directory
        """

        images = sorted(os.listdir(os.path.join(self.data_dir, self.img_dir)))

        joints_mat = scipy.io.loadmat(os.path.join(self.data_dir, "joints.mat"))["joints"]
        
        print("----- Loading Dataset")
        all_joints = []
        for i in tqdm(range(len(images))):
            joints = joints_mat[:, :, i]
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            # z coordinate should just be 0 for 2d image
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for j in range(self.num_joints):
                joints_3d[j, 0, 0] = joints[j, 0]
                joints_3d[j, 1, 0] = joints[j, 1]
                visible = joints[j, 2]
                joints_3d[j, :2, 1] = visible

            all_joints.append(joints_3d)

        print(f"----- Loaded {len(images)} samples")

        return images, np.array(all_joints)