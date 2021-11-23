import torch
import numpy as np
import pytorch3d

##########################################################################################
# 타겟 이미지에서 랜덤으로 ray를 샘플링해주는 객체
# Training 또는 Test 코드 레벨에서 생성된다.

# return
# 타겟 카메라에서 샘플링 된 픽셀로 향하는 ray의 direction, origin과 해당 픽셀의 rgb값
##########################################################################################


class RaySampler(object):
    def __init__(self, data, target_cam_idx, device, resize_factor=1, render_stride=1):
        super().__init__()

        self.render_stride = render_stride
        self.device = device

        self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        self.rgb_path = data['rgb_path'] if 'rgb_path' in data.keys() else None

        self.camera = data['camera']
        self.target_cam_idx = target_cam_idx

        self.depth_range = data['depth_range'] if 'depth_range' in data.keys() else None

        self.H, self.W = self.rgb.shape[-2:]

        self.ndc_transform = pytorch3d.renderer.cameras.get_screen_to_ndc_transform(self.camera, image_size=(self.H, self.W), with_xyflip=True)[target_cam_idx]
        self.K_transform = self.camera.get_projection_transform()[target_cam_idx]
        self.c2w_transform = self.camera.get_world_to_view_transform()[target_cam_idx].inverse()

        # 배치 내의 모든 이미지에 대해 전체 픽셀로 향하는 각 ray들을 정의
        self.rays_o, self.rays_d = self.get_rays(render_stride)

    
    # 타겟 이미지의 각 픽셀마다 rays_o, rays_d를 구하는 함수.
    # __init__() 단계에서 실행됨.
    def get_rays(self, render_stride):
        # u --> x 인덱싱
        # v --> y 인덱싱
        u, v = np.meshgrid(np.arange(self.W)[::render_stride], np.arange(self.H)[::render_stride])

        # 이미지의 각 row들이 한 줄로 이어붙은 형태로 변환
        # (H, W) --> (H*W)
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5

        pixels = np.stack((u, v, np.ones_like(u)), axis=1)  # [3(x+y+z), H*W]
        pixels = torch.from_numpy(pixels)       # pixels --> Screen coord

        # Screen >>> NDC
        ndc_pixels = self.ndc_transform.transform_points(pixels)
        # NDC >>> Camera
        cam_rays = self.K_transform.inverse().transform_points(ndc_pixels)
        # Camera >>> World
        rays_d = self.c2w_transform.transform_points(cam_rays)
        rays_o = self.camera.get_camera_center()[self.target_cam_idx].unsqueeze(0).repeat(rays_d.shape[0], 1)

        # 위의 rays_d는 rotation + tanslation까지 모두 변환된 것이기 때문에 
        # 해당 이미지 픽셀로 향하는 ray의 `방향`이 아니라
        # 해당 이미지 픽셀의 `위치`가 된다. 즉 3차원 point 좌표값이라는 것임.
        # 그래서 translation 값에 해당하는 rays_o를 한 번 빼서 `방향값`으로 만들어준다.
        rays_d -= rays_o
        
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
            rgb = self.rgb.permute(1, 2, 0).reshape(-1, 3).unsqueeze(0)
            rgb = rgb[:, select_inds]
        else:
            rgb = None

        ret = {'ray_o': rays_o.cuda(),
                'ray_d': rays_d.cuda(),
                'camera': self.camera.cuda(),
                'depth_range': self.depth_range.cuda() if self.depth_range is not None else None,
                'rgb': rgb.cuda() if rgb is not None else None,
                # 'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
                # 'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
                # 'selected_inds': select_inds
        }
        
        return ret