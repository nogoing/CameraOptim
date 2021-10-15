import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



############### Poisional Embedding
class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)

        return torch.cat((embed.sin(), embed.cos()), dim=-1)


# 타겟 이미지 내에서 랜덤으로 픽셀 샘플링
def sample_random_pixel(H, W, N_rand, sample_mode, center_ratio=0.8):
    rng = np.random.RandomState(234)

    if sample_mode == 'center':
        border_H = int(H * (1 - center_ratio) / 2.)
        border_W = int(W * (1 - center_ratio) / 2.)

        # pixel coordinates
        u, v = np.meshgrid(np.arange(border_H, H - border_H),
                            np.arange(border_W, W - border_W))
        u = u.reshape(-1)
        v = v.reshape(-1)

        select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
        select_inds = v[select_inds] + W * u[select_inds]

    elif sample_mode == 'uniform':
        # Random from one image
        select_inds = rng.choice(H*W, size=(N_rand,), replace=False)
    else:
        raise Exception("unknown sample mode!")

    return select_inds


# 타겟 이미지의 각 픽셀마다 rays_o, rays_d를 구함
def get_rays(H, W, intrinsics, c2w, render_stride):
    '''
    param H: image height
    param W: image width
    param intrinsics: 4 by 4 intrinsic matrix
    param c2w: 4 by 4 camera to world extrinsic matrix
    
    return: rays_o, rays_d
    '''
    # u --> x 인덱싱
    # v --> y 인덱싱
    u, v = np.meshgrid(np.arange(W)[::render_stride], np.arange(H)[::render_stride])

    # 이미지의 각 row들이 이어붙은 형태로 변환
    # (H, W) --> (H*W)
    u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5

    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # [3(x+y+z), H*W]
    pixels = torch.from_numpy(pixels)
    # batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

    # bmm : batch matrix-matrix product 
    # [B, N, M] tensor * [B, M, P] tensor >>> [B, N, P]
    # rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
    rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(pixels)).transpose(1, 2)
    rays_d = rays_d.reshape(-1, 3)
    rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
    
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


############### Coarse-to-Fine 샘플링
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


##########################################################################################
# NeRF에서의 렌더링 방식
##########################################################################################

# 여기서 coarse > fine 단계를 모두 수행
def render_rays(ray_batch,
                model,
                featmaps,
                projector,
                N_samples,
                inv_uniform=False,
                N_importance=0,
                det=False,
                white_bkgd=False):
    '''
    param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    param model:  {'net_coarse':  , 'net_fine': }
    param N_samples: samples along each ray (for both coarse and fine model)
    param inv_uniform: if True, uniformly sample inverse depth for coarse model
    param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    param det: if True, will deterministicly sample depths
    
    return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''

    ret = {'outputs_coarse': None,
           'outputs_fine': None}

    ################################## Coarse step ##################################

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(ray_o=ray_batch['ray_o'],
                                          ray_d=ray_batch['ray_d'],
                                          depth_range=ray_batch['depth_range'],
                                          N_samples=N_samples, inv_uniform=inv_uniform, det=det)
    N_rays, N_samples = pts.shape[:2]

    rgb_feat, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                 ray_batch['src_rgbs'],
                                                 ray_batch['src_cameras'],
                                                 featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x]
    pixel_mask = mask[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
    raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)   # [N_rays, N_samples, 4]
    outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask,
                                 white_bkgd=white_bkgd)
    ret['outputs_coarse'] = outputs_coarse

    ################################## Fine step ##################################
    if N_importance > 0:
        assert model.net_fine is not None
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

        rgb_feat_sampled, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                             ray_batch['src_rgbs'],
                                                             ray_batch['src_cameras'],
                                                             featmaps=featmaps[1])

        pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
        raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask)
        outputs_fine = raw2outputs(raw_fine, z_vals, pixel_mask,
                                   white_bkgd=white_bkgd)
        ret['outputs_fine'] = outputs_fine

    return ret


def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    '''
    param raw: raw network의 아웃풋 --> [N_rays, N_samples, 4(RGB + density)]
    param z_vals: rays 상에 놓인 각 samples에서의 depth --> [N_rays, N_samples]
    param ray_d: rays의 디렉션 벡터 --> [N_rays, 3]
    
    return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    rgb = raw[:, :, :3]     # color 값 : [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # density 값 : [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T     # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('mask', mask),
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret