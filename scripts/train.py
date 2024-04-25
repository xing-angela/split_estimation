import torch

from tqdm import tqdm
from argparse import ArgumentParser
from models.fast_pose import FastPose
from utils.dataset import HalpeDataset
from torch.utils.data import DataLoader
from utils.heatmap import joint_to_heatmap

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", required=True, type=str, help="Training data directory")
    parser.add_argument("--batch_size", default=48, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--img_dir", default="images/train2015", type=str, help="Directory of train images")

    parser.add_argument("--img_width", default=256, type=int, help="Size of the image width")
    parser.add_argument("--img_height", default=192, type=int, help="Size of the image height")

    return parser.parse_args()

def train(args, dataloader, model):

    print("----- Begin Training")
    for epoch in range(args.num_epochs):

        print(f"----- Epoch {epoch}")
        for i, (image, label) in enumerate(tqdm(dataloader)):
            output = model(image)

            img_w = label["width"]
            img_h = label["height"]
            joints_3d = label["joints_3d"]
            heatmaps = torch.Tensor([joint_to_heatmap((64, 48), (img_h[i], img_w[i]), joint) for i, joint in enumerate(joints_3d)])
            print(f"output shape: {output.shape}")
            print(f"heatmap shape: {heatmaps.shape}")
            return

def main():
    # parse command line arguments
    args = parse_args()

    # build the dataset
    img_size = (args.img_width, args.img_height)
    train_dataset = HalpeDataset(args.data_dir, args.img_dir, img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # initalize the model
    model = FastPose()

    loss = train(args, train_dataloader, model)

if __name__ == "__main__":
    main()