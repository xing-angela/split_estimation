import torch

from torch import nn
from torchvision import models
from models.duc import DUC

class FastPose(nn.Module):
    def __init__(self, num_joints):
        """
        Initalizes the model layers

        ResNet --> DUC --> DUC --> DUC --> Conv2D

        Parameters
        ----------
        num_joints: int
            the number of joints to output in the heatmap
        """
        super(FastPose, self).__init__()

        self.num_joints = num_joints

        # use the Pytorch pretrained resnet model
        # note: the last 2 layers result in shape that doesn't fit the DUC 
        #       input, so we are using all layers except the Average Pool and 
        #       linear layers at the end
        #       the result from this layer should be 2048 x h/32 x w/32
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet50.eval()
        layers = list(resnet50.children())[:-2]
        self.resnet = nn.Sequential(*layers)

        self.duc_layer1 = DUC(2048)
        self.duc_layer2 = DUC(1024)
        self.duc_layer3 = DUC(512)
        self.conv_layer = nn.Conv2d(256, self.num_joints, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.resnet(x)
        out = self.duc_layer1(out)
        out = self.duc_layer2(out)
        out = self.duc_layer3(out)
        out = self.conv_layer(out)
        return out
