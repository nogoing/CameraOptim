import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNet(nn.Module):

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)