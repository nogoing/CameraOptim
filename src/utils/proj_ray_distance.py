import torch
import pytorch3d
from src.models.modules.SuperGluePretrainedNetwork.models.matching import Matching


def init_superglue(rank):
    config = {
            "superpoint": {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": 1024,
            },
            "superglue": {
                "weights": "outdoor",
                "sinkhorn_iterations": 20,
                "match_threshold": 0.6,
            },
        }
    superglue = Matching(config).eval().to(rank)
    
    return superglue


def runSuperGlueSinglePair(superglue, img0, img1, rank):
    # Must have 3 channels
    assert img0.shape[2] == 3 and img1.shape[2] == 3
    
    # Convert RGB images to gray images 
    img0_gray = (
        0.2989 * (img0[:, :, 0])
        + 0.5870 * (img0[:, :, 1])
        + 0.1140 * (img0[:, :, 2])
    ).to(rank)
    
    img1_gray = (
        0.2989 * (img1[:, :, 0])
        + 0.5870 * (img1[:, :, 1])
        + 0.1140 * (img1[:, :, 2])
    ).to(rank)

    pred = superglue(
        {
            "image0": img0_gray[None, None, :, :],
            "image1": img1_gray[None, None, :, :]
        }
    )

    pred = {k: v[0] for k, v in pred.items()}

    match_src = torch.where(pred["matches0"] != -1)[0]
    match_trg = pred["matches0"][match_src]
    kps0, kps1 = pred["keypoints0"], pred["keypoints1"]

    matches = torch.stack([match_src, match_trg], dim=1)
    conf = pred['matching_scores0'][match_src]
    
    return {
            "kps0": kps0.detach(),
            "kps1": kps1.detach(),
            "matches": matches.detach(),
            "conf": conf
        }
    

def preprocess_match(match_result):
    kps0 = match_result["kps0"]
    kps1 = match_result["kps1"]
    matches = match_result["matches"]   # kps0, kps1의 몇 번째끼리 페어인지를 나타냄.

    if len(matches) == 0:
        return None, None

    kps0 = torch.stack([kps0[match_[0]] for match_ in matches])
    kps1 = torch.stack([kps1[match_[1]] for match_ in matches])

    return torch.stack([kps0, kps1])


def get_pair_rays(pair_cam_idxs, cameras, H, W, pairs):
    # pair_cam_idxs = (first_cam_idx, second_cam_idx)   ex) (0, 2)
    # cameras = src cameras
    # pairs = (2, correspondences, 2(xy))
    pair_rays_d = []
    pair_rays_o = []
    
    screen2ndc_true = pytorch3d.renderer.cameras.get_screen_to_ndc_transform(cameras, image_size=(H, W), with_xyflip=True)
    c2w_transform = cameras.get_world_to_view_transform().inverse()
    K_inv = cameras.get_projection_transform().inverse()
    for i, cam_idx in enumerate(pair_cam_idxs):   
        keypoint_xy = pairs[i]         # (correspondences, 2(xy))
        z_depth = torch.ones((keypoint_xy.shape[0], 1), device=cameras.device)
        keypoint_xyz = torch.cat((keypoint_xy, z_depth), dim=-1)
        
        keypoint_ndc = screen2ndc_true[cam_idx].transform_points(keypoint_xyz)# screen >> ndc

        cam_rays = K_inv[cam_idx].transform_points(keypoint_ndc)     # ndc >> cam

        
        rays_d = c2w_transform[cam_idx].transform_points(cam_rays)  # cam >> world

        pair_rays_d.append(rays_d)
        rays_o = cameras.get_camera_center()[cam_idx].repeat(rays_d.shape[0], 1)
        pair_rays_o.append(rays_o)
        
    pair_rays_d = torch.stack(pair_rays_d, dim=0)   # (2, correspondences, 3)
    pair_rays_o = torch.stack(pair_rays_o, dim=0)   # (2, 3)
    
    # 위의 rays_d는 rotation + tanslation까지 모두 변환된 것이기 때문에 
    # 해당 이미지 픽셀로 향하는 ray의 `방향`이 아니라
    # 해당 이미지 픽셀의 `위치`가 된다. 즉 3차원 point 좌표값이라는 것임.
    # 그래서 translation 값에 해당하는 rays_o를 한 번 빼서 `방향값`으로 만들어준다.
    pair_rays_d -= pair_rays_o
        
    return {"rays_d": pair_rays_d, "rays_o": pair_rays_o}


# 이미지 페어 한 쌍에서 correspondences 사이의 projected ray distance를 계산함.
def proj_ray_dist_loss_single(kps0_list, kps1_list, camera, pair_idxs, pair_rays, mode, H, W, eps=1e-10,
                              proj_ray_dist_threshold=10., only_projection_points=False, projection_point_idx=10):
    
    assert kps0_list[:, 0].max() < W and kps1_list[:, 0].max() < W
    assert kps0_list[:, 1].max() < H and kps1_list[:, 1].max() < H

    rays_d = pair_rays["rays_d"]    # (2, correspondences, 3)
    rays_o = pair_rays["rays_o"]    # (2, 3)
    
    if only_projection_points:
        rays_d = rays_d[:, [projection_point_idx]]
        rays_o = rays_o[:, [projection_point_idx]]
        kps0_list = kps0_list[[projection_point_idx], :]
        kps1_list = kps1_list[[projection_point_idx], :]

    rays_d = rays_d / (rays_d.norm(p=2, dim=-1) + eps).unsqueeze(-1)  # unit vector
    
    rays_d_0, rays_d_1 = rays_d[0].unsqueeze(0), rays_d[1].unsqueeze(0)      # (1, correspondences, 3)
    rays_o_0, rays_o_1 = rays_o[0].unsqueeze(0), rays_o[1].unsqueeze(0)      # (1, correspondences, 3)

    rays_d_0 = rays_d_0 / (rays_d_0.norm(p=2, dim=-1)[:, :, None] + eps)
    rays_d_1 = rays_d_1 / (rays_d_1.norm(p=2, dim=-1)[:, :, None] + eps) 

    r0_r1 = torch.einsum(
        "ijk, ijk -> ij", 
        rays_d_0, 
        rays_d_1
    )
    # 첫 번째 이미지에서의 각 특징점들로 향하는 step 거리
    t0 = (
        torch.einsum(
            "ijk, ijk -> ij", 
            rays_d_0, 
            rays_o_0 - rays_o_1
        ) - r0_r1
        * torch.einsum(
            "ijk, ijk -> ij", 
            rays_d_1, 
            rays_o_0 - rays_o_1
        )
    ) / (r0_r1 ** 2 - 1 + eps)
    # 두 번째 이미지에서의 각 특징점들로 향하는 step 거리
    t1 = (
        torch.einsum(
            "ijk, ijk -> ij", 
            rays_d_1, 
            rays_o_1 - rays_o_0
        ) - r0_r1
        * torch.einsum(
            "ijk, ijk -> ij", 
            rays_d_0, 
            rays_o_1 - rays_o_0
        )
    ) / (r0_r1 ** 2 - 1 + eps)

    # 각 이미지에서의 특징점들의 포인트 X_A, X_B
    p0 = t0[:, :, None] * rays_d_0 + rays_o_0
    p1 = t1[:, :, None] * rays_d_1 + rays_o_1

    # 페어 이미지의 screen으로 projection
    # X_AB, X_BA
    p0_norm_im1 = camera.transform_points_screen(p0, eps=1., image_size=(H, W))[pair_idxs[1]].reshape(p0.shape)
    p1_norm_im0 = camera.transform_points_screen(p1, eps=1., image_size=(H, W))[pair_idxs[0]].reshape(p1.shape)

    # p0_norm_im1_2d = p0_norm_im1[:, :, :2] / \
    #     (p0_norm_im1[:, :, 2, None] + eps)
    # p1_norm_im0_2d = p1_norm_im0[:, :, :2] / \
    #     (p1_norm_im0[:, :, 2, None] + eps)

    p0_norm_im1_2d = p0_norm_im1[:, :, :2]
    p1_norm_im0_2d = p1_norm_im0[:, :, :2]
        
    # Chirality check: remove rays behind cameras
    # First, flatten the correspondences
    # Find indices of valid rays
    valid_t0 = (t0 > 0).flatten()
    valid_t1 = (t1 > 0).flatten()
    valid = torch.logical_and(valid_t0, valid_t1)

    p1_norm_im0_2d, kps0_list = p1_norm_im0_2d[0, valid], kps0_list[valid]
    p0_norm_im1_2d, kps1_list = p0_norm_im1_2d[0, valid], kps1_list[valid]

    # loss 말고 프로젝션 위치 구하는 용...
    if only_projection_points:
        return p1_norm_im0_2d, p0_norm_im1_2d
    
    # Second, select losses that are valid
    loss0_list = (
        (p1_norm_im0_2d - kps0_list) ** 2
    ).sum(-1).flatten()
    loss1_list = (
        (p0_norm_im1_2d - kps1_list) ** 2
    ).sum(-1).flatten()

    if mode == "train":
        loss0_valid_idx = torch.logical_and(
            loss0_list < proj_ray_dist_threshold, 
            torch.isfinite(loss0_list)
        )
        loss1_valid_idx = torch.logical_and(
            loss1_list < proj_ray_dist_threshold, 
            torch.isfinite(loss1_list)
        )

        loss0 = loss0_list[loss0_valid_idx].mean()
        loss1 = loss1_list[loss1_valid_idx].mean()

        num_matches = torch.logical_and(
            loss0_valid_idx, loss1_valid_idx
        ).float().sum().item() 
        
        return 0.5 * (loss0 + loss1), num_matches
        
    else:
        loss0_invalid_idx = torch.logical_or(
            loss0_list > proj_ray_dist_threshold,
            torch.logical_not(torch.isfinite(loss0_list))
        )
        loss0_list[loss0_invalid_idx] = proj_ray_dist_threshold
        loss0 = loss0_list.mean()

        loss1_invalid_idx = torch.logical_or(
            loss1_list > proj_ray_dist_threshold,
            torch.logical_not(torch.isfinite(loss1_list))
        )
        loss1_list[loss1_invalid_idx] = proj_ray_dist_threshold
        loss1 = loss1_list.mean()

        return 0.5 * (loss0 + loss1), None
    
    # loss0 = loss0_list.mean()
    # loss1 = loss1_list.mean()
    
    # return 0.5 * (loss0 + loss1)