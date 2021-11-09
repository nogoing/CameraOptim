import torch
from .co3d_utils import *
from torch.utils.data import Dataset
import imageio


class CO3Ddataset(Dataset):
    def __init__(self, root_dir, mode, N_src, categories=[], bounded_crop=False, pose_normalize=False, **kwargs):
        self.dataset_path = os.path.join(root_dir, 'CO3D/')
        self.bounded_crop = bounded_crop

        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val

        self.num_source_views = N_src

        total_category = os.listdir(self.dataset_path)

        if len(categories) > 0:
            if isinstance(categories, str):
                categories = [categories]
        else:
            categories = total_category

        print("loading {} for {}".format(categories, mode))
        
        self.rgb_paths = []
        self.masks = []
        self.poses = []
        self.intrinsics = []

        for category in categories:
            self.category_path = os.path.join(self.dataset_path, category)     # ".../CO3D/teddybear"

            cg_rgb_paths, cg_mask_paths, cg_c2ws, cg_intrinsics  = read_category_data(self.category_path, pose_normalize)
            
            if self.mode != 'train':
                cg_rgb_paths = cg_rgb_paths[::self.testskip]
                cg_mask_paths = cg_mask_paths[::self.testskip]
                cg_c2ws = cg_c2ws[::self.testskip]
                cg_intrinsics = cg_intrinsics[::self.testskip]

            self.rgb_paths.extend(cg_rgb_paths)
            self.masks.extend(cg_mask_paths)
            self.poses.extend(cg_c2ws)
            self.intrinsics.extend(cg_intrinsics)


    def __len__(self):
        return len(self.rgb_paths)


    def __getitem__(self, idx):
        # 해당 idx는 타겟뷰가 된다.
        tgt_rgb_path = os.path.join(self.dataset_path, self.rgb_paths[idx])
        tgt_mask_path = os.path.join(self.dataset_path, self.masks[idx])
        tgt_pose = self.poses[idx]
        tgt_intrinsic = self.intrinsics[idx]

        print(tgt_rgb_path)
        
        # 타겟뷰가 속한 카테고리-시퀀스 경로를 가져온다.
        seq_path = '/'.join(tgt_rgb_path.split('/')[:-2])   # ".../CO3D/teddybear/"38_1655_5016"
        
        # 해당 sequence(=오브젝트)의 모든 img, c2w, intrinsic 정보를 읽어온다. --> 이 중에서 소스뷰를 선별
        seq_rgb_paths, seq_mask_paths, seq_c2ws, seq_intrinsics = read_seq_data(seq_path)

        if self.mode == 'train':
            id_render = int(os.path.basename(tgt_rgb_path)[5:-4]) - 1
            # 비슷한 소스뷰를 N_src보다 더 많이 선택하게 하는 인자값
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2]) 
        else:
            id_render = -1
            subsample_factor = 1

        rgb = imageio.imread(tgt_rgb_path).astype(np.float32).transpose(2, 0, 1) / 255.
        mask = imageio.imread(tgt_mask_path).astype(np.float32) / 255.
        
        img_size = rgb.shape[-2:]

        if self.bounded_crop:
            bbox_xywh = torch.tensor(get_bbox_from_mask(mask, thr=0.4), dtype=torch.float32)
            rgb = crop_around_box(torch.tensor(rgb, dtype=torch.float32), bbox_xywh, tgt_rgb_path)
            mask = crop_around_box(torch.tensor(mask, dtype=torch.float32), bbox_xywh, tgt_rgb_path)

        # 타겟뷰의 카메라 정보를 일렬로 이어붙임. (이미지 사이즈 2 + intrinsic 16 + extrinsic 16)
        camera = np.concatenate((list(img_size), tgt_intrinsic.flatten(),
                                 tgt_pose.flatten())).astype(np.float32)
        camera = torch.tensor(camera, dtype=torch.float32)

        tgt_camera_center = tgt_pose[:3, 3]
        tgt_camera_norm = np.linalg.norm(tgt_camera_center) - 8.

        near_depth = 0.0 if tgt_camera_norm < 0 else tgt_camera_norm
        far_depth = np.linalg.norm(tgt_camera_center) + 8.

        depth_range = torch.tensor([near_depth, far_depth])

        # 시퀀스 내의 모든 카메라에서 타겟뷰와 가장 가까운 N_src개의 소스뷰를 고른다.
        nearest_src_ids = get_nearest_src(tgt_pose, np.stack(seq_c2ws, axis=0),
                                            int(self.num_source_views*subsample_factor),
                                            tgt_id=id_render,
                                            angular_dist_method='vector')
        # subsample_factor 때문에 N_src보다 더 많이 추출된 소스뷰들 중에서 랜덤으로 N_src개를 선택한다.
        nearest_src_ids = np.random.choice(nearest_src_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_src_ids
        
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_src_ids[np.random.choice(len(nearest_src_ids))] = id_render

        src_rgb_paths = []
        src_rgbs = []
        src_masks = []
        src_cameras = []
        for id in nearest_src_ids:
            src_rgb_path = os.path.join(self.dataset_path, seq_rgb_paths[id])
            src_rgb = imageio.imread(src_rgb_path).astype(np.float32).transpose(2, 0, 1) / 255.

            src_mask_path = os.path.join(self.dataset_path, seq_mask_paths[id])
            src_mask = imageio.imread(src_mask_path).astype(np.float32) / 255.

            src_pose = seq_c2ws[id]
            src_intrinsic = seq_intrinsics[id]

            img_size = src_rgb.shape[-2:]

            if self.bounded_crop:
                bbox_xywh = torch.tensor(get_bbox_from_mask(src_mask, thr=0.4), dtype=torch.float32)
                src_rgb = crop_around_box(torch.tensor(src_rgb, dtype=torch.float32), bbox_xywh, src_rgb_path)
                src_mask = crop_around_box(torch.tensor(src_mask, dtype=torch.float32), bbox_xywh, src_mask_path)

            src_rgb_paths.append(seq_rgb_paths[id])
            src_rgbs.append(src_rgb)
            src_masks.append(src_mask)
            
            # 소스뷰의 카메라 정보를 일렬로 이어붙임. (이미지 사이즈 2 + intrinsic 16 + extrinsic 16)
            src_camera = np.concatenate((list(img_size), src_intrinsic.flatten(),
                                              src_pose.flatten())).astype(np.float32)
            src_cameras.append(torch.tensor(src_camera, dtype=torch.float32))

        return {'rgb': rgb, 
                'mask': mask,
                'camera': camera,
                'rgb_path': tgt_rgb_path,

                'src_rgbs': src_rgbs,
                'src_masks': src_masks,
                'src_cameras': src_cameras,
                'src_rgb_paths': src_rgb_paths,

                'depth_range': depth_range
                }