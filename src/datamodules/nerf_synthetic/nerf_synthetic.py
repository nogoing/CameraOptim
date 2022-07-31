import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pytorch3d
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras

import numpy as np

import os
import json
import numpy as np
from PIL import Image

from src.utils.utils import get_nearest_src


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta['camera_angle_x'])
    rgb_files = []
    w2cs = []

    # intrinsics: 모든 프레임에서 동일함
    img = Image.open(os.path.join(basedir, meta['frames'][0]['file_path'] + '.png'))
    H, W = img.height, img.width
    
    focal_screen = .5 * W / np.tan(.5 * camera_angle_x)
    s = min(H, W)
    focal_ndc = focal_screen * 2.0 / s
    focal_length = torch.tensor([focal_ndc, focal_ndc], dtype=torch.float32)
    
    principal_x_screen = W/2.
    principal_y_screen = H/2.
    principal_x_ndc = - (principal_x_screen - W / 2.0) * 2.0 / s
    principal_y_ndc = - (principal_y_screen - H / 2.0) * 2.0 / s
    principal_point = torch.tensor([principal_x_ndc, principal_y_ndc], dtype=torch.float32)

    # extrinsics: 각 프레임별
    for i, frame in enumerate(meta['frames']):
        rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'][2:] + '.png')
        rgb_files.append(rgb_file)
        # column major
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        w2c = torch.inverse(c2w)

        # X, Z축 반전 (blender >> pytorch3d)
        w2c[0] *= -1
        w2c[2] *= -1

        # row major
        w2c = w2c.T

        w2cs.append(w2c)

    return rgb_files, [focal_length]*len(w2cs),[principal_point]*len(w2cs), w2cs


class NerfSyntheticDataset(Dataset):
    def __init__(self, N_src, mode, val_len=10,
                 # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
                 scenes=()):
        self.folder_path = "/home/kmuvcl/dataset/nerf_synthetic/"
        
        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        
        self.mode = mode  # train / test / val
        self.N_src = N_src
        # self.testskip = args.testskip

        all_scenes = ('chair', 'drums', 'ficus', 'lego', 'hotdog', 'materials', 'mic', 'ship')
        
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = scenes.split(',')
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        
        self.render_rgb_files = []
        self.render_focal_lengths = []
        self.render_principals = []
        self.render_w2cs = []
               
        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_file = os.path.join(self.scene_path, 'transforms_{}.json'.format(mode))
            
            rgb_files, focal_length, principal_point, w2cs = read_cameras(pose_file)

            self.render_rgb_files.extend(rgb_files)
            self.render_focal_lengths.extend(focal_length)
            self.render_principals.extend(principal_point)
            self.render_w2cs.extend(w2cs)
        
        if mode == "val":
            val_ids = np.random.choice(len(self.render_rgb_files), val_len)
            self.render_rgb_files = [self.render_rgb_files[id] for id in val_ids]
            self.render_focal_lengths = [self.render_focal_lengths[id] for id in val_ids]
            self.render_principals = [self.render_principals[id] for id in val_ids]
            self.render_w2cs = [self.render_w2cs[id] for id in val_ids]
            
        self.PIL2Tensor = ToTensor()
        
    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        # target
        target_rgb_file = self.render_rgb_files[idx]
        target_focal_length = self.render_focal_lengths[idx]
        target_principal = self.render_principals[idx]
        target_w2c = self.render_w2cs[idx]
        
        # references
        # 소스뷰들은 train 집합에서 가져온다.
        ref_pose_file = os.path.join('/'.join(target_rgb_file.split('/')[:-2]), 'transforms_train.json')
        ref_rgb_files, ref_focal_length, ref_principal_point, ref_w2cs = read_cameras(ref_pose_file)
        
        if self.mode == 'train':
            id_render = int(os.path.basename(target_rgb_file)[:-4].split('_')[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1
        
        nearest_pose_ids = get_nearest_src(target_w2c,
                                            torch.stack(ref_w2cs, axis=0),
                                            int(self.N_src*subsample_factor),
                                            tgt_id=id_render,
                                            angular_dist_method='matrix')

        src_ids = np.random.choice(nearest_pose_ids, self.N_src, replace=False)
        
        ref_rgb_files = [ref_rgb_files[id] for id in src_ids]
        ref_focal_lengths = torch.stack([ref_focal_length[id] for id in src_ids])
        ref_principals = torch.stack([ref_principal_point[id] for id in src_ids])
        ref_w2cs = torch.stack([ref_w2cs[id] for id in src_ids])

        rgb_files = [target_rgb_file] + ref_rgb_files
        focal_lengths = torch.cat([target_focal_length[None], ref_focal_lengths], axis=0)
        principals = torch.cat([target_principal[None], ref_principals], axis=0)
        w2cs = torch.cat([target_w2c[None], ref_w2cs], axis=0)
        
        image_rgb = [self.PIL2Tensor((Image.open(rgb_file))) for rgb_file in rgb_files]
        image_rgb = torch.stack(image_rgb, dim=0)

        near_depth = 2.
        far_depth = 6.

        depth_range = torch.tensor([near_depth, far_depth])

        return {
                "image_rgb": image_rgb,
                "image_path": rgb_files,
                "focal_length": focal_lengths,
                "principal_point": principals,
                "w2c": w2cs,
                "depth_range": depth_range,
                }