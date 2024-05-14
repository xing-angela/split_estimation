import os
import cv2
import numpy as np

from argparse import ArgumentParser

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
    parser.add_argument("--joint_dir", required=True, type=str, help="Path of joint keypoints")
    parser.add_argument("--output_path", required=True, type=str, help="Path to output the testing results")

    return parser.parse_args()

def main():
    # parse command line arguments
    args = parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load the images into a dataset
    img_paths = sorted(os.listdir(args.img_dir))

    for i, img_name in enumerate(img_paths):
        # load the image
        img_path = os.path.join(args.img_dir, img_name)
        img = cv2.imread(img_path)

        # load the keypoints
        basename = img_name.split(".")[0]
        keypoint_path = os.path.join(args.joint_dir, f"{basename}.npy")
        keypoints = np.load(keypoint_path)

        joint1 = keypoints[1] - keypoints[0]
        joint2 = keypoints[4] - keypoints[5]

        # Calculate the angle between the joints
        dot_product = np.dot(joint1, joint2)

        norm_v1 = np.linalg.norm(joint1)
        norm_v2 = np.linalg.norm(joint2)

        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        # Adjust the angle based on the direction of the cross product
        cross_product = np.cross(joint1, joint2)
        if cross_product > 0.0:
            angle_deg = 360 - np.degrees(angle_rad)

        print(f"{basename}: {angle_deg}")

        if angle_deg >= 180:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        # visualizes the points used in the angle calculation
        for i, kp in enumerate(keypoints):
            if i == 0 or i == 1 or i == 4 or i == 5: 
                x = kp[0]
                y = kp[1]
                cor_x, cor_y = int(x), int(y)
                cv2.circle(img, (int(cor_x), int(cor_y)), 2, color, 2)
        
        cv2.line(img, [int(keypoints[1][0]), int(keypoints[1][1])], [int(keypoints[0][0]), int(keypoints[0][1])], color, 1)
        cv2.line(img, [int(keypoints[4][0]), int(keypoints[4][1])], [int(keypoints[5][0]), int(keypoints[5][1])], color, 1)

        # writes the angle onto the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 20)
        fontScale = 0.5
        thickness = 2
        lineType = 2

        cv2.putText(img, f"{str(int(angle_deg))} degrees", 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            color,
            thickness,
            lineType)
        
        output_path = os.path.join(args.output_path, img_name)
        cv2.imwrite(output_path, img)


if __name__ == "__main__":
    main()