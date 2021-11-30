import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d

from collections import OrderedDict

from point_sampling import sample_along_camera_ray


# world points를 각 src view로 projection 한 다음, 해당 지점에서의 img features를 샘플링
# positional embedding 값과 concat 하여 네트워크에 들어가는 최종 input tensor를 구성하는 함수.
def feature_sampling(feature_maps, camera, pts, pe, img_size, src_idxs, padding_mode="zeros", interpolation_mode="bilinear"):
    # feature_maps: (N_src, 100, H, W).  100 = resnet feature + RGB + Segmentation Mask.
    # pts: (N_rays, N_samples, 3) ---> World Coordinate.
    # pe: (N_rays, N_samples, embedding_dim) ---> Positional Embedding of pts.

    N_src = feature_maps.shape[0]
    N_rays, N_samples = pts.shape[:2]

    # pts (world) >>>> projection to each src views
    proj_points = camera.transform_points_screen(pts.reshape(-1, 3), image_size=img_size)   # (1 + N_src, N_rays*N_samples, 3)
    src_proj_points = proj_points[src_idxs]   # (N_src, N_rays*N_samples, 3)

    screen_to_ndc_transforms = pytorch3d.renderer.cameras.get_screen_to_ndc_transform(camera, image_size=img_size, with_xyflip=True)[src_idxs]
    src_proj_points_ndc = screen_to_ndc_transforms.transform_points(src_proj_points)
    
    src_grid = src_proj_points_ndc.reshape(-1, N_rays, N_samples, 3)[..., :2]   # (N_src, N_rays, N_samples, 2)
    
    # (N_src, feature_dim, N_rays, N_samples)
    sampling_features = F.grid_sample(feature_maps, src_grid, align_corners=False, 
                                        padding_mode=padding_mode, mode=interpolation_mode)

    pe = pe.permute(2, 0, 1).unsqueeze(0).repeat(N_src, 1, 1, 1)    # (N_src, embedding_dim, N_rays, N_samples)

    # concat(img_feature, positional_embedding)
    sampling_features = torch.cat((sampling_features, pe), dim=1)

    input_tensor = sampling_features.permute(2, 3, 0, 1)

    return input_tensor


##########################################################################################
# NeRF에서의 렌더링 방식
##########################################################################################


# # coarse 단계에서 아웃풋으로 나온 depth 값을 weight로 변환
# def raw2outputs(raw, z_vals, mask, white_bkgd=False):
#     '''
#     param raw: raw network의 아웃풋 --> [N_rays, N_samples, 4(RGB + density)]
#     param z_vals: rays 상에 놓인 각 samples에서의 depth --> [N_rays, N_samples]
#     param ray_d: rays의 디렉션 벡터 --> [N_rays, 3]
    
#     return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
#     '''
#     rgb = raw[:, :, :3]     # color 값 : [N_rays, N_samples, 3]
#     sigma = raw[:, :, 3]    # density 값 : [N_rays, N_samples]

#     # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
#     # very different scales, and using interval can affect the model's generalization ability.
#     # Therefore we don't use the intervals for both training and evaluation.
#     sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

#     # point samples are ordered with increasing depth
#     # interval between samples
#     dists = z_vals[:, 1:] - z_vals[:, :-1]
#     dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

#     alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

#     # Eq. (3): T
#     T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
#     T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

#     # maths show weights, and summation of weights along a ray, are always inside [0, 1]
#     weights = alpha * T     # [N_rays, N_samples]
#     rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

#     if white_bkgd:
#         rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

#     mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
#     depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

#     ret = OrderedDict([('rgb', rgb_map),
#                        ('depth', depth_map),
#                        ('weights', weights),                # used for importance sampling of fine samples
#                        ('mask', mask),
#                        ('alpha', alpha),
#                        ('z_vals', z_vals)
#                        ])

#     return ret


def EARayMarching(ray_densities, ray_colors, z_vals):
    # z_vals: (N_rays, N_samples)
    # ray_densities: (N_rays, N_samples, 1)
    # ray_colors: (N_rays, N_samples, 3)

    # delta 계산
    delta = z_vals[..., 1:] - z_vals[..., :-1]      # (N_rays, N_samples-1)
    delta = torch.cat((delta, delta[..., -1:]), dim=-1)     # (N_rays, N_samples)

    # alpha 계산
    T = torch.exp(-delta * ray_densities[..., 0])   # (N_rays, N_samples)
    alphas = 1 - T + 1e-10          # (N_rays, N_samples)

    # absorption 계산
    absorption = torch.cumprod(T, dim=1)        # (N_rays, N_samples)

    # weights 계산
    # importance sample을 추출하는 fine 단계에서 사용한다.
    weights = absorption * alphas           # (N_rays, N_samples)

    # color 계산
    colors = torch.sum(weights.unsqueeze(2) * ray_colors, dim=-2)    # (N_rays, 3)
    
    # mask 계산
    masks = 1 - torch.prod((1 - T), dim=1, keepdim=True)  # (N_rays, 1)
    
    # depth_map = torch.sum(wegihts * z_vals, dim=-1)
    
    return {
            "rgb": colors, 
            "weights": weights, 
            "alphas": alphas,
            "mask": masks,
            # "depth":depth_map
            }


# coarse 단계에서 계산된 weight 값을 이용하여 fine sampling
def sample_pdf(bins, weights, N_samples, det=False):
    '''
    param bins: tensor of shape [N_rays, M+1], M is the number of bins
    param weights: tensor of shape [N_rays, M]
    param N_samples: number of samples along each ray
    param det: if True, will perform deterministic sampling
    
    return: [N_rays, N_samples]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples


# 여기서 Coarse-to-Fine 과정을 수행
# ray sampling 이후의 모든 과정...
def render_rays(ray_batch,
                coarse_model,
                fine_model,
                feature_maps,
                src_idxs,
                N_samples,
                PE,
                inv_uniform=False,
                N_importance=0,
                det=False,
                # white_bkgd=False
                ):
    '''
    param ray_batch: {"ray_o": [N_rays, 3] , "ray_d": [N_rays, 3], "depth_range": [near, far], "camera": }
    param model:  {'net_coarse':  , 'net_fine': }
    param N_samples: samples along each ray (for both coarse and fine model)
    param inv_uniform: if True, uniformly sample inverse depth for coarse model
    param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    param det: if True, will deterministicly sample depths
    
    return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''

    ret = {"outputs_coarse": None,
           "outputs_fine": None}
    #################################################################################
    ################################## Coarse step ##################################
    #################################################################################

    # pts: (N_rays, N_samples, 3)
    # z_vals: (N_rays, N_samples)
    pts, z_vals = sample_along_camera_ray(ray_batch["ray_o"], ray_batch["ray_d"], ray_batch["depth_range"],
                                            N_samples,
                                            inv_uniform=inv_uniform,
                                            det=det)

    N_rays, N_samples = pts.shape[:2]
    H, W = feature_maps.shape[-2:]

    # Positional Embedding
    positional_embedding = PE(pts)

    # Input tensor 구성
    input_tensor = feature_sampling(feature_maps, ray_batch["camera"], pts, positional_embedding, (H, W), src_idxs)

    coarse_densities, coarse_colors = coarse_model(input_tensor)
    outputs_coarse = EARayMarching(coarse_densities, coarse_colors, z_vals)
    ret["outputs_coarse"] = outputs_coarse

    #################################################################################
    ################################### Fine step ###################################
    #################################################################################
    if N_importance > 0:
        assert fine_model is not None

        # detach since we would like to decouple the coarse and fine networks
        weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]
        
        if inv_uniform:
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                    weights=torch.flip(weights, dims=[1]),
                                    N_samples=N_importance, det=det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:
            # take mid-points of depth samples
            z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                                   N_samples=N_importance, det=det)  # [N_rays, N_importance]

        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)
        N_total_samples = N_samples + N_importance

        viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
        ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
        pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]

        # Positional Embedding
        positional_embedding = PE(pts)

        # Input tensor 구성
        input_tensor = feature_sampling(feature_maps, ray_batch["camera"], pts, positional_embedding, (H, W), src_idxs)

        coarse_densities, coarse_colors = fine_model(input_tensor)
        outputs_fine = EARayMarching(coarse_densities, coarse_colors, z_vals)
        ret["outputs_fine"] = outputs_fine

    return ret


# 한 이미지 전체를 렌더링하는 함수.
def render_image(ray_sampler,
                 ray_batch,
                 src_idxs,
                 model,
                 feature_maps,
                 chunk_size,
                 N_samples,
                 PE,
                 inv_uniform=False,
                 N_importance=0,
                 det=False,
                 render_stride=1,
                 ):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                           ('outputs_fine', OrderedDict())])

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        ray_chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range']:
                ray_chunk[k] = ray_batch[k]
            # ray_o, ray_d, rgb, mask는 chunck_size씩 끊어서...
            elif ray_batch[k] is not None:
                ray_chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                ray_chunk[k] = None

        output = render_rays(ray_chunk, model, model, feature_maps, src_idxs, N_samples, PE, N_importance=N_importance)

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in output['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if output['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in output['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in output['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(output['outputs_coarse'][k].cpu())

        if output['outputs_fine'] is not None:
            for k in output['outputs_fine']:
                all_ret['outputs_fine'][k].append(output['outputs_fine'][k].cpu())
    
    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # chunk별로 나누어진 결과를 병합
    for k in all_ret['outputs_coarse']:
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                        rgb_strided.shape[1], -1))
        all_ret['outputs_coarse'][k] = tmp.squeeze()

    all_ret['outputs_coarse']["rgb"][all_ret['outputs_coarse']["mask"] == 0] = 1.

    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                        rgb_strided.shape[1], -1))

            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret