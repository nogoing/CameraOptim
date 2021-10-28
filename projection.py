# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F


class Projector():
    def __init__(self, device):
        self.device = device

    def inbound(self, pixel_locations, h, w):
        '''
        픽셀이 이미지 상의 유효한 위치에 놓여 있는지 확인하는 함수.
        param pixel_locations: [..., 2]
        param h: height
        param w: weight
        
        return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)


    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        
        return normalized_pixel_locations


    def compute_projections(self, xyz, train_cameras):
        '''
        3D 포인트를 train_cameras(소스 카메라)의 이미지 스페이스로 프로젝션하는 함수
        param xyz: [..., 3]
        param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        
        return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)

        # 타겟 카메라의 intrinsics
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        # 타겟 카메라의 extrinsics
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        
        # (x, y, z) ---> (x, y, z, 1) : Homogeneous Coordinate로 변환.
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]

        # P_proj = K x c2w_inv(=w2c) x P
        # 소스뷰는 자신의 extrinsic(c2w_inverse)와 intrinsic을 곱하여 이미지 스페이스로 가져옴
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]

        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]

        # 픽셀 위치 (x, y)만 추출
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)

        # 카메라의 뒤로 projection된 포인트는 invalid로 판단.
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera
        
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape)


    def compute_angle(self, xyz, query_camera, train_cameras):
        '''
        param xyz: [..., 3]
        param query_camera: 타겟 카메라 [34, ] : img_size(2) + intrinsics(16) + extrinsics(16)
        param train_cameras: 소스 카메라 [n_views, 34]  : img_size(2) + intrinsics(16) + extrinsics(16)
        
        return: [n_views, ..., 4] : 앞 3개 채널 = query ray와 target ray 사이의 direction 차이를 나타내는 단위 벡터 / 마지막 채널 = 두 방향 벡터의 내적값
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        
        num_views = len(train_poses)

        # target 카메라의 pose를 소스 개수만큼 복사.
        query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]

        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)     # 단위벡터화

        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)     # 단위벡터화

        # 두 단위 벡터의 차이값을 계산
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)  # 단위벡터화
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)

        # 두 단위 벡터를 내적
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)

        # [두 ray의 차이를 나타내는 방향 벡터(3채널), 두 ray의 내적값(1채널)]
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, ) + original_shape + (4, ))

        return ray_diff


    # 메인으로 실행되는 함수
    def compute(self,  xyz, query_camera, train_imgs, train_cameras, featmaps):
        '''
        :param xyz: [n_rays, n_samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, n_views, h, w, 3] = 소스 이미지 n개
        :param train_cameras: [1, n_views, 34] = 소스 카메라 n개 / img_size(2) + intrinsics(16) + extrinsics(16)
        :param featmaps: [n_views, d, h, w] = 소스 이미지 n개의 resnet feature map

        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 ray_diff: [n_rays, n_samples, 4],
                 mask: [n_rays, n_samples, 1]
        '''
        assert (train_imgs.shape[0] == 1) \
               and (train_cameras.shape[0] == 1) \
               and (query_camera.shape[0] == 1), 'only support batch_size=1 for now'

        # 소스 이미지, 소스 카메라의 1이었던 batch 차원을 삭제
        train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
        # 타겟 이미지의 1이었던 batch 차원을 삭제
        query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # 채널 순서를 변경 ---> [n_views, 3, h, w]

        h, w = train_cameras[0][:2]

        # 쿼리 포인트 (x, y, z)가 각 소스 뷰들로 projection 되는 위치를 계산
        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]

        # 각 소스 이미지들로부터 프로젝션 된 위치의 RGB 컬러값을 샘플링
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # 각 소스 이미지의 faeture map들로부터 프로젝션 된 위치의 resnet feature 값을 샘플링
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        
        # [RGB 샘플링 값, resnet feature 샘플링 값]
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        ray_diff = self.compute_angle(xyz, query_camera, train_cameras)
        ray_diff = ray_diff.permute(1, 2, 0, 3)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   # [n_rays, n_samples, n_views, 1]

        return rgb_feat_sampled, ray_diff, mask