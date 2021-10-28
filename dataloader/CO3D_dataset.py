import torch
from co3d_utils import *
from torch.utils.data import Dataset
import imageio


class CO3Ddataset(Dataset):
    def __init__(self, args, mode, categories=[], **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'CO3D/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val w

        self.num_source_views = args.num_source_views

        total_category = os.listdir(self.folder_path)

        if len(categories) > 0:
            if isinstance(categories, str):
                categories = [categories]
        else:
            categories = total_category

        print("loading {} for {}".format(categories, mode))
        
        self.tgt_imgs = []
        self.tgt_poses = []
        self.tgt_intrinsics = []

        for category in categories:
            self.category_path = os.path.join(self.folder_path, category)     # ".../CO3D/teddybear"

            rgb_files, c2ws, intrinsics  = read_category_data(self.category_path)
            
            if self.mode != 'train':
                rgb_files = rgb_files[::self.testskip]
                intrinsics = intrinsics[::self.testskip]
                c2ws = c2ws[::self.testskip]
            self.tgt_imgs.extend(rgb_files)
            self.tgt_poses.extend(c2ws)
            self.tgt_intrinsics.extend(intrinsics)


    def __len__(self):
        return len(self.tgt_imgs)


    def __getitem__(self, idx):
        tgt_img = self.tgt_imgs[idx]
        tgt_pose = self.tgt_poses[idx]
        tgt_intrinsic = self.tgt_intrinsics[idx]
        
        # 선택한 데이터가 속한 category, sequence 이름을 가져온다.
        category, seq_name = tgt_img.split('/')[:-2]   # "teddybear", "38_1655_5016"
        seq_file_path = os.path.join(self.folder_path, category, seq_name, "frame_annotations_file.json")
        # 해당 sequence (=오브젝트)의 모든 img, c2w, intrinsic 정보를 읽어온다. --> 소스뷰로 이용
        src_imgs, src_c2ws, src_intrinsics = read_seq_data(seq_file_path)

        if self.mode == 'train':
            id_render = int(os.path.basename(tgt_img)[:-4].split('_')[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = imageio.imread(tgt_img).astype(np.float32) / 255.
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), tgt_intrinsic.flatten(),
                                 tgt_pose.flatten())).astype(np.float32)

        # 시퀀스 내의 모든 카메라에서 타겟뷰와 가장 가까운 n개의 소스뷰를 고른다.
        nearest_src_ids = get_nearest_src(tgt_pose,
                                                src_c2ws,
                                                int(self.num_source_views*subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_src_ids = np.random.choice(nearest_src_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_src_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_src_ids[np.random.choice(len(nearest_src_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_src_ids:
            src_rgb = imageio.imread(src_imgs[id]).astype(np.float32) / 255.
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = src_c2ws[id]
            src_intrinsics_ = src_intrinsics[id]

            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, tgt_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), src_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        near_depth = 2.
        far_depth = 6.

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': tgt_img,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }