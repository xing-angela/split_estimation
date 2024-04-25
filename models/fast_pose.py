import torch
from torch import nn
from torchvision import models
from models.duc import DUC_Layer

class FastPose(nn.Module):
    def __init__(self):
        """
        Initalizes the model layers

        ResNet --> DUC --> DUC --> DUC --> Conv2D
        """

        super().__init__()

        self.num_joints = 26 # 26 joints for the Halpe data for only the body

        # use the Pytorch pretrained resnet model
        # note: the last 2 layers result in shape that doesn't fit the DUC 
        #       input, so we are using all layers except the Average Pool and 
        #       linear layers at the end
        #       the result from this layer should be 2048 x h/32 x w/32
        resnet50 = models.resnet50(pretrained=True)
        layers = list(resnet50.children())[:-2]
        self.resnet = nn.Sequential(*layers)

        self.duc_layer1 = DUC_Layer(2048)
        self.duc_layer2 = DUC_Layer(1024)
        self.duc_layer3 = DUC_Layer(512)
        self.conv_layer = nn.Conv2d(256, self.num_joints, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor):
        out = self.resnet(x)
        out = self.duc_layer1(out)
        out = self.duc_layer2(out)
        out = self.duc_layer3(out)
        out = self.conv_layer(out)
        return out
