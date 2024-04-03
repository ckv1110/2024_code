import torchviz
import torch
from model_utils import Unet

model = Unet(in_channels=3, classes=10, reverse_skip_concat=False)
x = torch.randn((1, 3, 128, 128))
y = model(x)

torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))

input_names = ['Input Image']
output_names = ['Prediction']
torch.onnx.export(model, x, 'Unet_10.onnx', input_names=input_names, output_names=output_names)