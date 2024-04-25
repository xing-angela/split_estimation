import os
import cv2
import json
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

class HalpeDataset(Dataset):
    """
    Halpe Dataset for the 26 body key points (exludes key points 
    that are on the face and the hand)
    """
    num_joints = 26
    
    def __init__(self, data_dir: str, img_dir: str, img_size: (int, int)):
        """
        Initializes the Halpe Dataset

        Parameters
        ----------
        data_dir: string
            path of the data directory
        img_dir: string
            path of image directory within the data directory
        img_size: string
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

    
    def __getitem__(self, index: int):
        """
        Returns the image and annotations of the dataset at the specific index

        Parameters
        ----------
        index : int
            index of a sample

        Returns
        -------
        tuple[np.ndarray, torch.Tensor]
            image and annotations of the sample at the index
        """
        img_path = self.images[index]['img_path']
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        test_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        bbox = label['bbox']
        joints_3d = label['joints_3d']
        
        img_h, img_w, _ = img.shape
        target_w, target_h = self.img_size
        scale_x, scale_y = target_w / img_w, target_h / img_h
        
        # scales the image to the correct size
        scaled_img = cv2.resize(img, (target_w, target_h))

        scaled_img = np.transpose(scaled_img, (2, 0, 1))  # C x H x W
        scaled_img = torch.from_numpy(scaled_img).float()
        if scaled_img.max() > 1:
            scaled_img /= 255

        # scales the bbox
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

        scale_x_min = x_min * scale_x
        scale_y_min = y_min * scale_y
        scale_x_max = x_max * scale_x
        scale_y_max = y_max * scale_y

        # scales the joint points
        for i in range(self.num_joints):
            if joints_3d[i, 0, 1] > 0.0:
                joints_3d[i, 0, 0] = joints_3d[i, 0, 0] * scale_x
                joints_3d[i, 1, 0] = joints_3d[i, 1, 0] * scale_y

        scaled_label = {
            'bbox': (scale_x_min, scale_y_min, scale_x_max, scale_y_max),
            'width': target_w,
            'height': target_h,
            'joints_3d': joints_3d
        }

        return scaled_img, scaled_label

    
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

            