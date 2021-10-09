import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 입력으로 받은 소스뷰의 (ResNet feature + Image Color + Segmentation mask)
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        resnet = models.resnet34(pretrained=True).eval()

        # ~ layer1 = [:5]   >> Feature map shape : (1/4*H, 1/4*W, 64)
        # layer2 = [5]      >> Feature map shape : (1/8*H, 1/8*W, 128)
        # layer3 = [6]      >> Feature map shape : (1/16*H, 1/16*W, 256)
        feature_layers = list(resnet.children())[:7]

        for layer in feature_layers:
            layer.requires_grad = False

        self.feature_layers = nn.ModuleList(feature_layers)

        self.conv1x1_1 = nn.Conv2d(64, 32, (1, 1))
        self.conv1x1_2 = nn.Conv2d(128, 32, (1, 1))
        self.conv1x1_3 = nn.Conv2d(256, 32, (1, 1))

    def forward(self, img, mask):
        feature_maps = []
        orig_img = img

        for i, layer in enumerate(self.feature_layers):
            img = layer(img)

            # Upsampling > 1x1 Conv2D (32ch) > l2 normaliation of the feature column
            if i == 4:
                feature_map = nn.Upsample(scale_factor=4, mode='bilinear')(img)
                feature_map = self.conv1x1_1(feature_map)
                feature_map = F.normalize(feature_map, p=2, dim=1)
                feature_maps.append(feature_map)
            elif i == 5:
                feature_map = nn.Upsample(scale_factor=8, mode='bilinear')(img)
                feature_map = self.conv1x1_2(feature_map)
                feature_map = F.normalize(feature_map, p=2, dim=1)
                feature_maps.append(feature_map)
            elif i == 6:
                feature_map = nn.Upsample(scale_factor=16, mode='bilinear')(img)
                feature_map = self.conv1x1_3(feature_map)
                feature_map = F.normalize(feature_map, p=2, dim=1)
                feature_maps.append(feature_map)

        feature_maps.append(orig_img)
        feature_maps.append(mask)
        
        # feature_maps = [ResNet Features, RGB Image, Segmentation Mask]
        # 각 feature의 shape은 (1, C, H, W) ?
        feature_map = torch.cat(feature_maps, dim=1))   # 채널 기준으로 이어붙임.

        return feature_map