import os
import cv2
import torch
import numpy as np
import mediapipe as mp

from tqdm import tqdm
from argparse import ArgumentParser
from models.fast_pose import FastPose
from utils.heatmap import heatmap_to_joint
from utils.visualize import vis_bbox, vis_joints

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

img_size = [192, 256]

HALPE_JOINT_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
    (17, 18), (18, 19), (19, 11), (19, 12),
    (11, 13), (12, 14), (13, 15), (14, 16),
    (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
]

LSP_JOINT_PAIRS = [
    (1, 0), (2, 1), (2, 3), (3, 4), (4, 5), (7, 6), (8, 7),
    (9, 10), (10, 11), (12, 3), (12, 8), (12, 9), (13, 12)
]

def parse_args():
    """
    Parses the command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--img_dir", required=True, type=str, help="Path of the images to test on")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output the testing results")
    parser.add_argument("--checkpoint", required=True, type=str, help="Name of the checkpoint file")
    parser.add_argument("--train_dataset", choices=["Halpe26", "LSP"], default="Halpe26", help="Name of the data used for training")
    parser.add_argument("--save_joints", action="store_true", help="Whether to store the joint keypoints")

    return parser.parse_args()

def crop_img(img, bbox):
    """
    Crops the image according to the bounding box and reshapes the 
        image to the correct size for the model
    
    Parameters
    ----------
    img: np.ndarray
        the image as a numpy array
    bbox: list
        the bounding box (x_min, y_min, x_max, y_max)
    
    Returns
    ----------
    np.ndarray
        the cropped and reshaped image
    """
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    cropped_img = img[y_min:y_max, x_min:x_max, :]

    return cropped_img

def get_mp_bbox(img):
    """
    Gets the bounding box of the person using mediapipe

    Code from:
        https://developers.google.com/mediapipe/solutions/vision/object_detector/python

    Parameters
    ----------
    img: np.array
        the image as a numpy array
    
    Returns
    ----------
    list
        the bounding box for the image
    """

    model_path = 'pretrained/efficientdet_lite0.tflite'

    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=10,
        running_mode=VisionRunningMode.IMAGE
    )

    with ObjectDetector.create_from_options(options) as detector:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = detector.detect(img)

        # find the person with the largest bounding box
        largest_width = 0
        person_id = 0
        for i, person in enumerate(detection_result.detections):
            if person.bounding_box.width > largest_width:
                largest_width = person.bounding_box.width
                person_id = i

        bbox = detection_result.detections[0].bounding_box
        x_origin = bbox.origin_x + (bbox.width / 2)
        y_origin = bbox.origin_y + (bbox.height / 2)

        # pad the bounding box
        padding = 0.1
        x_pad = (bbox.width * padding) / 2
        y_pad = (bbox.height * padding) / 2

        x_min = int(x_origin - (bbox.width / 2) - x_pad)
        y_min = int(y_origin - (bbox.height / 2) - y_pad)
        x_max = int(x_origin + (bbox.width / 2) + x_pad)
        y_max = int(y_origin + (bbox.height / 2) + y_pad)

        out_bbox = [x_min, y_min, x_max, y_max]

        return out_bbox


def main():
    # parse command line arguments
    args = parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # create the model
    print("----- Loading the Model")
    if args.train_dataset == "Halpe26":
        model = FastPose(26)
    else:
        model = FastPose(14)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE), strict=False)
    model.eval()

    # load the images into a dataset
    img_paths = sorted(os.listdir(args.img_dir))

    for i, img_name in enumerate(tqdm(img_paths)):
        img_path = os.path.join(args.img_dir, img_name)

        with torch.no_grad():
            # decrease resolution of original image
            target_w = 720
            orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if orig_img.shape[1] > target_w:
                resize_factor = orig_img.shape[1] / target_w
                target_h = int(orig_img.shape[0] / resize_factor)
                orig_img = cv2.resize(orig_img, (target_w, target_h))
            
            # crop and resize the image to the desired shape
            bbox = get_mp_bbox(orig_img)
            cropped_img = crop_img(orig_img, bbox)

            if not np.all(cropped_img.shape):
                print(f"Bounding box detection failed. Skipping this image {img_name}.")
                continue

            target_w = 129
            if cropped_img.shape[1] > target_w:
                resize_factor = cropped_img.shape[1] / target_w
                target_h = int(cropped_img.shape[0] / resize_factor)
                scale_cropped = cv2.resize(cropped_img, (target_w, target_h))
            else:
                scale_cropped = cropped_img.copy()

            scaled_img = cv2.resize(scale_cropped, (img_size[0], img_size[1]))

            input_img = np.transpose(scaled_img, (2, 0, 1))  # C x H x W
            input_img = torch.from_numpy(input_img).float()
            if input_img.max() > 1:
                input_img /= 255
            input_img = input_img.to(DEVICE)

            # makes the batch size be 1 for the cropped image
            input_img = input_img.unsqueeze(0)
            
            heatmap = model(input_img)
            heatmap = heatmap.cpu().data.numpy()
            heatmap = heatmap[0]

            joints = heatmap_to_joint(heatmap, cropped_img.shape[:2])

            # visualizes the joint keypoints
            output_path = os.path.join(args.output_path, img_name)
            output_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            if args.train_dataset == "Halpe26":
                vis_joints(output_img, joints, HALPE_JOINT_PAIRS, output_path)
            else:
                vis_joints(output_img, joints, LSP_JOINT_PAIRS, output_path)

            # stores the joint keypoints
            if args.save_joints:
                kp_x = joints[:, 0, 0]
                kp_y = joints[:, 1, 0]

                keypoints = np.column_stack((kp_x, kp_y))

                basename = img_name.split(".")[0]

                file = os.path.join(args.output_path, basename)
                np.save(file, keypoints)


if __name__ == "__main__":
    main()