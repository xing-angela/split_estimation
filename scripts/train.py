import os
import torch

from torch import nn
from tqdm import tqdm
from models.unipose import UniPose
from argparse import ArgumentParser
from models.fast_pose import FastPose
from dataset.lsp_dataset import LSPDataset
from dataset.halpe_dataset import HalpeDataset
from torch.utils.data import DataLoader
from utils.accuracy import calc_accuracy
from utils.heatmap import joint_to_heatmap
from utils.visualize import vis_loss, vis_acc
from torchsummary import summary

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    """
    Parses the command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--data_dir", required=True, type=str, help="Training data directory")
    parser.add_argument("--img_dir", required=True, type=str, help="Directory of train images")
    parser.add_argument("--out_dir", default="./exp", help="The output directory")
    parser.add_argument("--exp_name", default="0", help="Experiment name")

    parser.add_argument("--model", choices=["AlphaPose", "UniPose"], default="AlphaPose", help="Name of the model to train")
    parser.add_argument("--dataset", choices=["Halpe26", "LSP"], default="Halpe26", help="Name of the data used for training")
    parser.add_argument("--batch_size", default=48, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=200, type=int, help="Number of epochs")

    parser.add_argument("--img_width", default=256, type=int, help="Size of the image width")
    parser.add_argument("--img_height", default=192, type=int, help="Size of the image height")

    parser.add_argument("--save_epoch", default=50, type=int, help="Saves model snapshot every n epochs")
    parser.add_argument("--plot_loss", action="store_true", help="Whether to plot the loss for the epochs")
    parser.add_argument("--plot_acc", action="store_true", help="Whether to plot the accuracy for the epochs")

    parser.add_argument("--from_epoch", default=0, type=int, help="Continues training from n epoch")

    return parser.parse_args()

def train(args, dataloader, model, loss_function, optimizer, start_epoch, end_epoch, print_info=True):
    """
    Trains the model

    Parameters
    ----------
    args: ArgumentParser
        the command line arguments
    dataloader: DataLoader
        the training data
    model: Module
        the model
    loss_function: LossFunction
        the loss function
    optimizer: Optimizer
        the optimizer
    start_epoch: int
        the start epoch
    end_epoch: int
        the end epoch
    print_info: bool
        indicates whether to print loss information after each epoch
    """

    exp_dir = os.path.join(args.out_dir, args.exp_name)

    epoch_average_losses = []
    epoch_average_acc = []

    print("----- Begin Training")
    for epoch in range(start_epoch, end_epoch):
        epoch_loss_sum = 0.0
        epoch_acc_sum = 0.0
        print(f"----- Epoch {epoch}")
        
        progress = tqdm(dataloader)
        for i, (images, labels) in enumerate(progress):
            img_w = labels["width"]
            img_h = labels["height"]
            heatmaps = labels["heatmaps"]
            masks = labels["masks"]

            # converts data to cuda
            images = images.to(DEVICE).requires_grad_()
            heatmaps = heatmaps.to(DEVICE)
            masks = masks.to(DEVICE)

            output = model(images)

            gt = heatmaps.mul(masks)
            output = output.mul(masks)

            # calculate loss and accuracy
            loss = 0.5 * loss_function(output, gt)
            accuracy = calc_accuracy(output, gt)

            # updating weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculates the current loss
            epoch_loss_sum += loss.item() * images.shape[0]
            epoch_acc_sum += accuracy * images.shape[0]

            progress.set_description(
                "loss: {loss:.8f} | accuracy: {accuracy:.4f}".format(
                    loss=loss.item(), 
                    accuracy=accuracy)
            )

        # calculates the average loss for the epoch
        epoch_average_losses.append(epoch_loss_sum / len(dataloader.dataset))
        epoch_average_acc.append(epoch_acc_sum / len(dataloader.dataset))

        if print_info:
            print(
                "Epoch: {} | Loss: {:.8f} | Accuracy: {:.4f}".format(
                    epoch, epoch_loss_sum / len(dataloader.dataset),
                    epoch_acc_sum / len(dataloader.dataset)
                )
            )

        # saves the model every x epochs
        if epoch % args.save_epoch == 0:
            torch.save(model.state_dict(), f'{exp_dir}/model_{epoch}.pth')

            # also saves the loss and accuracy plots if necessary
            if args.plot_loss:
                output_plot = f"{exp_dir}/epoch_loss.jpg"
                vis_loss(epoch_average_losses, output_plot)
            
            if args.plot_acc:
                output_plot = f"{exp_dir}/epoch_acc.jpg"
                vis_acc(epoch_average_acc, output_plot)

    # saves the model
    torch.save(model.state_dict(), f'{exp_dir}/model_final.pth')

    # saves the loss and accuracy plots if necessary
    if args.plot_loss:
        output_plot = f"{exp_dir}/epoch_loss.jpg"
        vis_loss(epoch_average_losses, output_plot)
    
    if args.plot_acc:
        output_plot = f"{exp_dir}/epoch_acc.jpg"
        vis_acc(epoch_average_acc, output_plot)
    
    return

def main():
    # parse command line arguments
    args = parse_args()

    # creates experiment directory
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # build the dataset
    img_size = (args.img_width, args.img_height)
    if args.dataset == "Halpe26":
        train_dataset = HalpeDataset(args.data_dir, args.img_dir, img_size)
    else:
        train_dataset = LSPDataset(args.data_dir, args.img_dir, img_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # initalize the model
    if args.model == "AlphaPose":
        if args.dataset == "Halpe26":
            model = FastPose(26)
        else:
            model = FastPose(14)
    else:
        model = UniPose((args.img_width, args.img_height))
    
    model = model.to(DEVICE)
    
    # allows from continuing model from a specific epoch by loading the previous weights
    if args.from_epoch != 0:
        start_epoch = args.from_epoch
        end_epoch = start_epoch + args.num_epochs
        checkpoint = f'{exp_dir}/model_{args.from_epoch}.pth'
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE), strict=False)
        else:
            print("Checkpoint does not exist. Exiting...")
            return
    else:
        start_epoch = 0
        end_epoch = start_epoch + args.num_epochs

    model.train()

    # MSE loss and Adam optimizer
    loss_function = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # train
    loss = train(args, train_dataloader, model, loss_function, optimizer, start_epoch, end_epoch)

if __name__ == "__main__":
    main()