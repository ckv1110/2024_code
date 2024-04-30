import torchviz
import torch
from modules.model import BuildResidualUnet, BuildUnet, BuildResNetEncoderUnet
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = 'resnet34'
model = BuildResNetEncoderUnet(in_channels=3, classes=10, resnet_type=resnet, weights=None, k_size=3, pad=1, b=True, reverse_skip_concat=False, residual_decoder=False).to(device)
# model = BuildResidualUnet(in_channels=3, classes=3, model_depth=5, input_layer_channels=64, k_size=3, st=2, pad=1, b=True, reverse_skip_concat=False).to(device)
# model = BuildUnet(in_channels=3, classes=3, model_depth=6, input_layer_channels=64, k_size=3, st=2, pad=1, b=True, reverse_skip_concat=False).to(device)
x = torch.randn((1, 3, 512, 512)).to(device)
y = model(x)

torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))

input_names = ['Input Image']
output_names = ['Prediction']
os.makedirs('./model_architectures', exist_ok=True)
torch.onnx.export(model, x, f'./model_architectures/{resnet}Unet.onnx', input_names=input_names, output_names=output_names)
# torch.onnx.export(model, x, './model_architectures/Unet_depth-6.onnx', input_names=input_names, output_names=output_names)