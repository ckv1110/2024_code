import torchviz
import torch
from modules.model import BuildResidualUnet, BuildUnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BuildResidualUnet(in_channels=3, classes=3, k_size=3, st=2, pad=1, b=True, reverse_skip_concat=False).to(device)
x = torch.randn((1, 3, 512, 512)).to(device)
y = model(x)

torchviz.make_dot(y.mean(), params=dict(model.named_parameters()))

input_names = ['Input Image']
output_names = ['Prediction']
torch.onnx.export(model, x, './model_architectures/ResidualUnet.onnx', input_names=input_names, output_names=output_names)