import torch
import torch.nn

from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCGridRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

from network.model import NerFormer, FeatureNet

# ray sampler 생성
ray_sampler = MonteCarloRaysampler(
    image_width=-1.0,
    image_heignt=1.0,
    n_rays_per_iamge=800,
    n_pts_per_ray=32,
    min_depth=0.3,
    max_depth=3.0,
)

# ray marcher 생성
ray_marcher = EmissionAbsorptionRaymarcher()

# 위의 두 개를 이용하여 renderer 생성
renderer = ImplicitRenderer(
    raysampler=ray_sampler,  raymarcher=ray_marcher, 
)

# opacity, color 결정 모델 생성
nerformer = NerFormer(d_z=100)

feature_net = FeatureNet() # 소스 이미지 모두 넣어서 feature + rgb + mask 맵을 생성해야 함.

optim = torch.optim.Adam((nerformer.parameters(), feature_net.parameters()), lr=1e-3)

for iter in range(10000):
    batch_idx = torch.randint(len(cameras))
    optim.zero_grad()
    
    # rendering
    rendered_pixels, sampled_rays = renderer(
        cameras=camera,
        volumetric_function=nerformer,
    )

    gt_pixels = sample_images_at_locations(gt_images[batch_idx], sampled_rays.xys)

    # 기존 NeRF의 original loss
    rgb_mse_loss = (rendered_pixels - gt_pixels).abs().mean()
    # Nerformer에서 새롭게 추가한 loss
    # 렌더링된 알파 마스크와 GT 마스크 사이의 BCE loss
    bce_loss = torch.nn.BCELoss(mask, gt_mask)
    
    total_loss = rgb_mse_loss + bce_loss
    total_loss.backward()

    optim.step()