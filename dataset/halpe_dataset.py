import os
import cv2
import json
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from utils.augment import augment_data
from utils.heatmap import joint_to_heatmap
from utils.visualize import vis_joints, vis_bbox, vis_heatmap

JOINT_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
               [20, 21], [22, 23], [24, 25]]

class HalpeDataset(Dataset):
    """
    Halpe Dataset for the 26 body key points (exludes key points 
    that are on the face and the hand)
    """
    num_joints = 26
    
    def __init__(self, data_dir, img_dir, img_size):
        """
        Initializes the Halpe Dataset

        Parameters
        ----------
        data_dir: str
            path of the data directory
        img_dir: str
            path of image directory within the data directory
        img_size: str
            target size of images after scaling
        images: list
            a list of {img_path, img_id}
        labels: list
            a list of {bbox, width, height, joints_3d}
            * note that the joints3d is a list of 26 joints
        """
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.img_size = img_size
        
        # loads the images and labels
        data = self.load_from_json()
        self.images = data[0]
        self.labels = data[1]

    
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
        img_path = self.images[index]['img_path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        bbox = label['bbox']
        joints_3d = label['joints_3d']
        
        img_h, img_w, _ = img.shape
        target_w, target_h = self.img_size
        scale_x, scale_y = target_w / img_w, target_h / img_h
        
        # scales the image to the correct size
        scaled_img = cv2.resize(img, (target_w, target_h))

        # scales the bbox
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

        scale_x_min = x_min * scale_x
        scale_y_min = y_min * scale_y
        scale_x_max = x_max * scale_x
        scale_y_max = y_max * scale_y

        scaled_bbox = [scale_x_min, scale_y_min, scale_x_max, scale_y_max]

        # scales the joint points
        scaled_joints = joints_3d.copy()
        for i in range(self.num_joints):
            if scaled_joints[i, 0, 1] > 0.0:
                scaled_joints[i, 0, 0] = scaled_joints[i, 0, 0] * scale_x
                scaled_joints[i, 1, 0] = scaled_joints[i, 1, 0] * scale_y

        # further augments the data
        scale_label = {
            'bbox': scaled_bbox,
            'width': target_w,
            'height': target_h,
            'joints_3d': scaled_joints
        }
        aug_img, aug_label = augment_data(scaled_img, scale_label, JOINT_PAIRS, True)
        aug_h, aug_w = aug_label['height'], aug_label['width']
        aug_bbox = aug_label['bbox']
        aug_joints = aug_label['joints_3d']

        # ensures the augmented image is the right shape and converts to tensor
        aug_img = np.transpose(aug_img, (2, 0, 1))  # C x H x W
        aug_img = torch.from_numpy(aug_img).float()
        if aug_img.max() > 1:
            aug_img /= 255

        # gets the heatmap and the mask of the joints
        # heatmaps, masks, combined_heatmap = joint_to_heatmap((64, 48), (aug_h, aug_w), aug_joints)
        heatmaps, masks, combined_heatmap = joint_to_heatmap((32, 24), (aug_h, aug_w), aug_joints, sigma=3)

        # visualization
        # if flip and rotate:
        #     print("flip and rotate")
        #     outfolder = "/users/axing2/data/users/axing2/split_estimation/test_aug"
        #     cv2.imwrite(os.path.join(outfolder, "non_aug.jpg"), scaled_img)
        #     cv2.imwrite(os.path.join(outfolder, "original.jpg"), aug_img)
        #     vis_joints(aug_img.copy(), aug_joints, os.path.join(outfolder, "joints.jpg"))
        #     vis_bbox(aug_img.copy(), aug_bbox, os.path.join(outfolder, "bbox.jpg"))
        # for i in range(len(heatmaps)):
        #     heatmap = heatmaps[i]
        #     vis_heatmap(scaled_img.copy(), heatmap, os.path.join(outfolder, f"heatmap_{i}.jpg"))
        # vis_heatmap(aug_img.copy(), combined_heatmap, os.path.join(outfolder, f"heatmap_combined.jpg"))

        aug_label = {
            'bbox': aug_bbox,
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


    def load_from_json(self):
        """
        Loads the images and the labels from the data file
        Inspiration taken from 
            https://github.com/Fang-Haoshu/Halpe-FullBody/blob/master/vis.py
            https://github.com/MVIG-SJTU/AlphaPose/blob/c60106d19afb443e964df6f06ed1842962f5f1f7/alphapose/datasets/halpe_26.py
        """
        
        annot_file = os.path.join(self.data_dir, "halpe_train_v1.json")
        body_annot = json.load(open(annot_file))

        images = []
        labels = []

        # stores the dict of images to match images to ids
        imgs = {}
        for img in body_annot['images']:
            imgs[img['id']] = img

        print("----- Loading Dataset")
        for i, annot in enumerate(tqdm(body_annot['annotations'])):
            img_id = annot['image_id']
            img = imgs[img_id]
            img_w = img['width']
            img_h = img['height']
            img_path = os.path.join(self.data_dir, self.img_dir, img['file_name'])

            # keypoints list does not exist, the keypoints list is just 0s, 
            # or length of list is 0
            if ('keypoints' not in annot) or (max(annot['keypoints']) == 0) or (len(annot['keypoints']) == 0):
                continue

            # get the bounding box
            bbox = annot['bbox']
            # converts the bbox from x, y, w, h to x_min, y_min, x_max, y_max and 
            # ensures that the bounding box is within the bounds of the image
            w, h = np.maximum(bbox[2] - 1, 0), np.maximum(bbox[3] - 1, 0)
            x_min = np.minimum(img_w - 1, np.maximum(0, bbox[0]))
            y_min = np.minimum(img_h - 1, np.maximum(0, bbox[1]))
            x_max = np.minimum(img_w - 1, np.maximum(0, bbox[0] + w))
            y_max = np.minimum(img_h - 1, np.maximum(0, bbox[1] + h))
            # require non-zero box area
            if (x_max - x_min) * (y_max - y_min) <= 0 or x_max <= x_min or y_max <= y_min:
                continue
            
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            # z coordinate should just be 0 for 2d image
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = annot['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = annot['keypoints'][i * 3 + 1]
                if annot['keypoints'][i * 3 + 2] >= 0.35:
                    visible = 1
                else:
                    visible = 0
                joints_3d[i, :2, 1] = visible

            # no visible keypoint
            if np.sum(joints_3d[:, 0, 1]) < 1:
                continue

            images.append({'img_path': img_path, 'img_id': img_id})
            labels.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'width': img_w,
                'height': img_h,
                'joints_3d': joints_3d
            })

        print(f"----- Loaded {len(images)} samples")

        return images, labels

            