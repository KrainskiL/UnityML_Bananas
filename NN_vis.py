from model import QNetwork
import torchviz
import torch

nn = QNetwork(37,4,42)
torchviz.make_dot(nn(torch.randn(1,37)), params=dict(nn.named_parameters())).render("NN_arch", format="png")