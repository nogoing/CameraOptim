import torch
import pytorch3d

from network.model import NerFormer, FeatureNet
from ray_sampling import RaySampler
from point_sampling import sample_along_camera_ray

from co3d.dataset.dataset_zoo import dataset_zoo
from co3d.dataset.dataloader_zoo import dataloader_zoo




def train():
    device = "cuda"

    # CO3D dataset
    root_dir = "../dataset/CO3D"

    category = ["teddybear", "cake"]

    co3d_sequence = "singlesequence"
    singlesequence_id = 0

    dataset = dataset_zoo(...)
    data_loader = dataloader_zoo(...)

    train_loader = data_loader["train"]

    # network
    model = NerFormer()
    feature_net = FeatureNet()

    # criterion
    
