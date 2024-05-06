from .model_utils import BuildUnet, BuildResidualUnet, BuildResNetEncoderUnet
import torch

model1 = BuildUnet(in_channels=3, classes=3, model_depth=6, input_layer_channels=64, k_size=3, st=1, pad=1, b=True, reverse_skip_concat=False)
model2 = BuildUnet(in_channels=3, classes=3, model_depth=6, input_layer_channels=64, k_size=3, st=1, pad=1, b=True, reverse_skip_concat=False)

class FF_block(torch.nn.Module):
    def __init__(self, block1, block2, in_channels, out_channels):
        super(FF_block, self).__init__()
        __concat_block = torch.concat([block1, block2], dim=1)
        __conv_block = torch.nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels)

    def forward(self, x):
        return self.model1(x), self.model2(x)

class BuildMTLModel(torch.nn.Module):
    def __init__(self, model1, model2):
        super(BuildMTLModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
    
        self.model1.children = []

    def forward(self, x):
        return self.model1(x), self.model2(x)