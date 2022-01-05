import torch

# 주어진 N개의 ray를 따라서 point sampling 수행
def sample_along_camera_ray(ray_o, ray_d, depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    '''
    param ray_o: scene coordinate system에서 ray의 원점 : [N_rays, 3]
    param ray_d: scene coordinate system에서 ray의 방향 벡터 : [N_rays, 3]
    param depth_range: [near_depth, far_depth]
    param inv_uniform: if True, uniformly sampling inverse depth
    param det: if True, will perform deterministic sampling
    
    return: tensor of shape [N_rays, N_samples, 3]
    '''
    # 샘플들은 [near_depth, far_depth] 범위 안에 있어야 함.
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0]
    far_depth_value = depth_range[1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])
    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples]

    # 샘플 포인트 위치를 계산.
    # ray의 원점으로부터 ray의 방향으로 z_val만큼 전진한 것.
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3]
    
    return pts, z_vals