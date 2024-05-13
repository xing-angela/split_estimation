import cv2
import numpy as np

def vis_joints(image, keypoints, joint_pairs, output_path):
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
    joint_pairs: list paris
        the key point pairs to visualize the line
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
        cv2.circle(image, (int(cor_x), int(cor_y)), 2, (255,255,255), 2)

    for i, (start_p, end_p) in enumerate(joint_pairs):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
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


def vis_heatmap(image: np.ndarray, heatmap: np.ndarray, output_path: str):
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
    heatmap = heatmap / heatmap.max()
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, image, 0.5, 0)

    cv2.imwrite(output_path, super_imposed_img)


def vis_loss(losses: list[float], output_path: str):
    """
    Saves the plot with the loss for every epoch

    Parameters
    ----------
    losses: list[float]
        a list of the losses per epoch
    output_path: str
        the path to save the plot
    """
    from matplotlib import pyplot as plt 

    x = np.arange(1, len(losses) + 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Loss per Epoch")
    plt.plot(x, losses)

    plt.savefig(output_path, dpi=600)

def vis_acc(accuracies: list[float], output_path: str):
    """
    Saves the plot with the loss for every epoch

    Parameters
    ----------
    losses: list[float]
        a list of the losses per epoch
    output_path: str
        the path to save the plot
    """
    from matplotlib import pyplot as plt 
    
    x = np.arange(1, len(accuracies) + 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.plot(x, accuracies)

    plt.savefig(output_path, dpi=600)