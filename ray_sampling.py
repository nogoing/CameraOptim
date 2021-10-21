import torch
import numpy as np


##########################################################################################
# 타겟 이미지에서 랜덤으로 ray를 샘플링해주는 객체
# Training 또는 Test 코드 레벨에서 생성된다.
##########################################################################################


class RaySampler(object):
    def __init__(self, data, device, H, W, intrinsics, c2w, resize_factor=1, render_stride=1):
        super().__init__()

        self.render_stride = render_stride
        self.device = device

        self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        self.camera = data['camera']
        self.rgb_path = data['rgb_path']
        self.depth_range = data['depth_range']

        self.H = int(H)
        self.W = int(W)
        self.intrinsics = intrinsics
        self.c2w_mat = c2w

        self.batch_size = len(self.camera)

        # 배치 내의 모든 이미지에 대해 전체 픽셀로 향하는 각 ray들을 정의
        self.rays_o, self.rays_d = self.get_rays(self.H, self.W, self.intrinsics, self.c2w_mat, render_stride)

    
    # 타겟 이미지의 각 픽셀마다 rays_o, rays_d를 구하는 함수.
    # __init__() 단계에서 실행됨.
    def get_rays(self, H, W, intrinsics, c2w, render_stride):
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

        # 이미지의 각 row들이 한 줄로 이어붙은 형태로 변환
        # (H, W) --> (H*W)
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5

        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # [3(x+y+z), H*W]? 아니면 homogeneous coordinate라 1을 추가한 건지?
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        # bmm : batch matrix-matrix product 
        # [B, N, M] tensor * [B, M, P] tensor >>> [B, N, P]
        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        # rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        
        return rays_o, rays_d

    # 타겟 이미지 내에서 픽셀을 랜덤으로 샘플링하는 함수.
    # sample_random 함수에서 실행됨.
    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        rng = np.random.RandomState(234)

        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                                np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform':
            # Random from one image
            select_inds = rng.choice(self.H * self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds


    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        '''
        param N_rand: number of rays to be casted

        return:
        '''

        # 타겟 이미지에서 픽셀을 랜덤으로 샘플링한다.
        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        # 앞서 구해 놓은 rays_o, rays_d에서 샘플링된 픽셀에 해당하는 것들을 가져온다.
        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
        else:
            rgb = None

        ret = {'ray_o': rays_o.cuda(),
                'ray_d': rays_d.cuda(),
                'camera': self.camera.cuda(),
                'depth_range': self.depth_range.cuda(),
                'rgb': rgb.cuda() if rgb is not None else None,
                'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
                'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
                'selected_inds': select_inds
        }
        
        return ret


# def get_rays_np(H, W, K, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

#     return rays_o, rays_d




# def ndc_rays(H, W, focal, near, rays_o, rays_d):
#     # Shift ray origins to near plane
#     t = -(near + rays_o[...,2]) / rays_d[...,2]
#     rays_o = rays_o + t[...,None] * rays_d
    
#     # Projection
#     o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
#     o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
#     o2 = 1. + 2. * near / rays_o[...,2]

#     d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
#     d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
#     d2 = -2. * near / rays_o[...,2]
    
#     rays_o = torch.stack([o0,o1,o2], -1)
#     rays_d = torch.stack([d0,d1,d2], -1)
    
#     return rays_o, rays_d