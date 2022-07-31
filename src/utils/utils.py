import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
import numpy as np

from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .rendering import render_image, feature_sampling
from src.utils.proj_ray_distance import proj_ray_dist_loss_single


# (N_src + N_extra)개의 랜덤 뷰 이미지 중에서
# 타겟과 가장 가까운 소스 뷰 이미지 N_src개를 선택하여 리턴한다.
def data_to_frame(data, N_src, dataset="co3d", rot_noise_factor=0.025, trans_noise_factor=0.05):
    if dataset == "co3d":
        imgs = data.image_rgb
        img_paths = data.image_path
        masks = data.fg_probability
        camera = data.camera
        in_ndc = True
    elif dataset ==  "nerf_synthetic":
        imgs = data["image_rgb"].squeeze(0)[:, :3, :, :]
        img_paths = data["image_path"]
        masks = data["image_rgb"].squeeze(0)[:, -1, :, :].unsqueeze(1)   # mask = alpha channel
        camera = PerspectiveCameras(focal_length=data["focal_length"].squeeze(0),
                                    principal_point=data["principal_point"].squeeze(0),
                                    R=data["w2c"][0, :, :3, :3],
                                    T=data["w2c"][0, :, 3, :3],
                                    device=imgs.device)
        in_ndc = True
        depth_range = data["depth_range"][0]

    c2ws = camera.get_world_to_view_transform().inverse().get_matrix()
    tgt_idx = 0
    tgt_c2w = c2ws[tgt_idx]
    src_c2ws = c2ws[1:]
    src_idxs = get_nearest_src(tgt_c2w, src_c2ws, N_src) + 1

    target = {}
    target["rgb"] = imgs[tgt_idx]
    target["rgb_path"] = img_paths[tgt_idx]
    tgt_principal_point = camera.principal_point[tgt_idx]
    tgt_focal_length = camera.focal_length[tgt_idx]
    tgt_R = camera.R[tgt_idx]
    tgt_T = camera.T[tgt_idx]
    tgt_cam = PerspectiveCameras(
                                focal_length=tgt_focal_length[None],
                                principal_point=tgt_principal_point[None],
                                R=tgt_R[None],
                                T=tgt_T[None],
                                in_ndc=in_ndc,
                            ).to(camera.device)
    target["camera"] = tgt_cam
    # 현재 타겟 카메라의 위치를 기준으로
    # 샘플링할 ray들의 depth range 계산
    if dataset == "co3d":
        target_camera_position = camera.get_camera_center()[tgt_idx]
        near = torch.max(torch.tensor([0.1], device=target_camera_position.device), torch.norm(target_camera_position) - 8)
        far = torch.norm(target_camera_position) + 8
        depth_range =  torch.tensor([near, far], dtype=torch.float32, device=imgs.device)
        target["depth_range"] = depth_range
    elif dataset == "nerf_synthetic":
        target["depth_range"] = depth_range
    
    srcs = {}
    srcs["rgb"] = imgs[src_idxs]
    srcs["rgb_path"] = [img_paths[i] for i in src_idxs]
    src_principal_points = camera.principal_point[src_idxs]
    src_focal_lengths = camera.focal_length[src_idxs]
    src_Rs = camera.R[src_idxs]
    src_Ts = camera.T[src_idxs]
    srcs["camera"] = PerspectiveCameras(
                                        focal_length=src_focal_lengths,
                                        principal_point=src_principal_points,
                                        R=src_Rs,
                                        T=src_Ts,
                                        in_ndc=in_ndc,
                                    ).to(camera.device)
    srcs["noise_camera"] = make_noise_camera(srcs["camera"], rot_noise_factor, trans_noise_factor)
    
    if dataset == "co3d":
        target["mask"] = masks[tgt_idx]
        srcs["mask"] = masks[src_idxs]
    elif dataset == "nerf_synthetic":
        target["mask"] = masks[tgt_idx]
        srcs["mask"] = masks[src_idxs]
        
    return target, srcs


def make_noise_camera(camera, rot_noise_factor, trans_noise_factor):
    c2w = camera.get_world_to_view_transform().inverse()
    c2w_mat = c2w.get_matrix()
    
    cam_num = c2w_mat.shape[0]
    
    R = c2w_mat[..., :3, :3]
    roll_pitch_yaw_noise = (torch.randn((cam_num, 3), device=c2w_mat.device))*rot_noise_factor        # degree (-5 ~ +5)
    rotation_noise_mat = get_rotation_matrix_from_RPY(roll_pitch_yaw_noise[..., 0], roll_pitch_yaw_noise[..., 1], roll_pitch_yaw_noise[..., 2])
    R_noise = torch.matmul(R, rotation_noise_mat)       # row major

    T = c2w_mat[..., 3, :3]
    translation_noise = (torch.randn((cam_num, 3), device=c2w_mat.device))*trans_noise_factor
    T_noise = T + translation_noise

    c2w_noise = torch.eye(4).repeat(cam_num, 1, 1)
    c2w_noise[..., :3, :3] = R_noise
    c2w_noise[..., 3, :3] = T_noise

    w2c_noise = torch.inverse(c2w_noise)
    noise_camera = PerspectiveCameras(
                                            focal_length=camera.focal_length,
                                            principal_point=camera.principal_point,
                                            R=w2c_noise[..., :3, :3],      # row major
                                            T=w2c_noise[..., 3, :3],      # row major
                                        ).to(camera.device)
    
    return noise_camera


def get_rotation_matrix_from_RPY(rolls, pitchs, yaws):
    # roll, pitch, yaw : radian
    src_num = rolls.shape[0]
    tensor_0 = torch.zeros(1, device=rolls.device)
    tensor_1 = torch.ones(1, device=rolls.device)

    rotations = []
    for cam_idx in range(0, src_num):
        roll = rolls[cam_idx].unsqueeze(0)
        pitch = pitchs[cam_idx].unsqueeze(0)
        yaw = yaws[cam_idx].unsqueeze(0)
        
        R_x = torch.stack([
                        torch.stack([tensor_1, tensor_0, tensor_0]),
                        torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                        torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

        R_y = torch.stack([
                        torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                        torch.stack([tensor_0, tensor_1, tensor_0]),
                        torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)
        
        R_z = torch.stack([
                        torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                        torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                        torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

        rotation = torch.matmul(torch.matmul(R_z, R_y), R_x)    # roll * pitch * yaw
        rotation = torch.transpose(rotation, dim0=0, dim1=1)    # row major로 변환
        rotations.append(rotation)
        
    rotation = torch.stack(rotations)
    
    return rotation


def get_relative_pose(tgt_cam, src_cams):
    tgt_w2c = tgt_cam.get_world_to_view_transform().get_matrix()
    srcs_w2c = src_cams.get_world_to_view_transform().get_matrix()
    
    srcs_c2w = torch.inverse(srcs_w2c)

    # tgt과 srcs 사이의 relative pose 구하기
    # srcs >>> tgt
    relative_pose = torch.matmul(srcs_c2w, tgt_w2c)
    
    return relative_pose


def camera_update_and_feature_resampling(feature_maps, pts, pe_dim, tgt_cam, src_cams, delta_rotation, delta_translation):
    relative_pose = get_relative_pose(tgt_cam, src_cams)
    
    tgt_c2w = tgt_cam.get_world_to_view_transform().inverse().get_matrix()
    
    # update relative pose
    relative_pose[:, :3, :3] = torch.matmul(relative_pose[:, :3, :3], delta_rotation)
    relative_pose[:, 3, :3] = relative_pose[:, 3, :3] + delta_translation
    
    # update srcs_c2w
    new_srcs_c2w = torch.matmul(relative_pose, tgt_c2w)

    optimized_srcs_w2c = torch.inverse(new_srcs_c2w)
    optimized_srcs_R = optimized_srcs_w2c[:, :3, :3]      # row major
    optimized_srcs_T = optimized_srcs_w2c[:, 3, :3]       # row major
        
    # camera는 return 하지 않아도 객체 값 자체가 업데이트 됨.
    src_cams.R = optimized_srcs_R
    src_cams.T = optimized_srcs_T
    
    # image feature resampling with optimized camera
    resampling_feature = feature_sampling(feature_maps, src_cams, pts, pe_dim, resampling=True)
    
    return resampling_feature


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_camera(fig, cam_c2w, image_bound_rays, c, color, face_color, marker, s, draw_corner=True):
    # translation
    T = cam_c2w[3, :3].detach().cpu().numpy()
                
    # camera center
    x, y, z = T
    fig.scatter3D(x, y, z, c=c, marker=marker, s=s)

    for _, ray_d in enumerate(image_bound_rays):
        xs = [x, ray_d[0].item()]
        ys = [y, ray_d[1].item()]
        zs = [z, ray_d[2].item()]

        a = Arrow3D(xs, ys, zs, mutation_scale=15, 
                            lw=1., arrowstyle="-", color=color)
        fig.add_artist(a)
        
    rays = image_bound_rays.detach().cpu()
    
    left_top = rays[0].numpy()
    right_top = rays[1].numpy()
    left_bottom = rays[2].numpy()
    right_bottom = rays[3].numpy()

    # left
    fig.plot(np.linspace(left_top[0], left_bottom[0]), np.linspace(left_top[1], left_bottom[1]), np.linspace(left_top[2], left_bottom[2]), color)
    # top
    fig.plot(np.linspace(left_top[0], right_top[0]), np.linspace(left_top[1], right_top[1]), np.linspace(left_top[2], right_top[2]), color)
    # right
    fig.plot(np.linspace(right_top[0], right_bottom[0]), np.linspace(right_top[1], right_bottom[1]), np.linspace(right_top[2], right_bottom[2]), color)
    # bottom
    fig.plot(np.linspace(right_bottom[0], left_bottom[0]), np.linspace(right_bottom[1], left_bottom[1]), np.linspace(right_bottom[2], left_bottom[2]), color)

    verts = [left_top, right_bottom, right_top, left_bottom, left_top, right_bottom]
    fig.add_collection3d(Poly3DCollection(verts, facecolors=face_color, alpha=.30))
    
    if draw_corner:
        fig.scatter3D(left_top[0], left_top[1], left_top[2], c='red', s=1)                  # left-top (= image의 (0,0))
        fig.scatter3D(right_bottom[0], right_bottom[1], right_bottom[2], c='blue', s=1)     # right-bottom (= image의 (H-1,W-1))
        fig.scatter3D(right_top[0], right_top[1], right_top[2], c='black', s=1)
        fig.scatter3D(left_bottom[0], left_bottom[1], left_bottom[2], c='black', s=1)
    

def get_noise_camera_figure(orig_cams, noise_cams, draw_in_one_figure=False):
    # orig_cams : GT src views
    # noise_cams : [5 * src views]  >> Noise Cam or Optimized Cam from Noise
    
    # left-top, right-top, left-bottom, right-bottom
    ndc_bound_ray = torch.tensor([[1., 1., 1.], [-1., 1., 1.], [1., -1., 1.], [-1., -1., 1.]], device=orig_cams.device)

    gt_K_inv = orig_cams.get_projection_transform().inverse()       # (N_src, 4, 4)
    gt_c2ws = orig_cams.get_world_to_view_transform().inverse()      # (N_src, 4, 4)
    gt_c2w_mats = gt_c2ws.get_matrix()

    gt_cam_bound_rays = gt_K_inv.transform_points(ndc_bound_ray)
    gt_world_bound_rays = gt_c2ws.transform_points(gt_cam_bound_rays)
    
    if noise_cams == None:
        for cam_idx in range(0, gt_c2w_mats.shape[0]):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlabel("X_world")
            ax.set_ylabel("Y_world")
            ax.set_zlabel("Z_world")
            ax.view_init(elev=30, azim=120)
            
            gt_c2w_mat = gt_c2w_mats[[cam_idx]]   # (1, 4, 4)
            gt_bound_ray = gt_world_bound_rays[[cam_idx]]     # (1, 4, 3)
            noise_c2w_mat = noise_c2w_mats[cam_idx]      # (optim_step, 4, 4)
            noise_bound_ray = noise_world_bound_rays[cam_idx]   # (optim_step, 4, 3)
            
            for step, (c2w, bound_ray) in enumerate(zip(c2w_mats, bound_rays)):
                if step == 0:       # gt
                    line_color = "black"
                    c = 'black'
                    face_color = "red"
                    marker = 'd'
                    s = 50
            
                # 인자로 받은 figure에 카메라 하나(c2w, bound_ray)를 그린다.
                draw_camera(ax, c2w, bound_ray, c, line_color, face_color, marker, s)
        
        return fig
        
    noise_c2w_mats = []
    noise_world_bound_rays = []
    for noise_cam in noise_cams:
        K_inv = noise_cam.get_projection_transform().inverse()       # (N_src, 4, 4)
        c2w = noise_cam.get_world_to_view_transform().inverse()      # (N_src, 4, 4)
        noise_c2w_mats.append(c2w.get_matrix())

        cam_bound_ray = K_inv.transform_points(ndc_bound_ray)
        world_bound_rays = c2w.transform_points(cam_bound_ray)
        noise_world_bound_rays.append(world_bound_rays)
    noise_c2w_mats = torch.stack(noise_c2w_mats, dim=1)                       # (N_src, optim_step, 4, 4)
    noise_world_bound_rays = torch.stack(noise_world_bound_rays, dim=1)     # (N_src, optim_step, 4, 3)
    
    colors = ["lightgray", "silver", "gray", "dimgray", "black"]
    # 소스 카메라 1개당 figure 1개
    if not(draw_in_one_figure):
        figs = []
        for cam_idx in range(0, gt_c2w_mats.shape[0]):
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlabel("X_world")
            ax.set_ylabel("Y_world")
            ax.set_zlabel("Z_world")
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=30, azim=120)
            
            gt_c2w_mat = gt_c2w_mats[[cam_idx]]   # (1, 4, 4)
            gt_bound_ray = gt_world_bound_rays[[cam_idx]]     # (1, 4, 3)
            noise_c2w_mat = noise_c2w_mats[cam_idx]      # (optim_step, 4, 4)
            noise_bound_ray = noise_world_bound_rays[cam_idx]   # (optim_step, 4, 3)

            c2w_mats = torch.cat((gt_c2w_mat, noise_c2w_mat), dim=0)        # (1+optim_step, 4, 4)
            bound_rays = torch.cat((gt_bound_ray, noise_bound_ray), dim=0)  # (1+optim_step, 4, 3)
                        
            for step, (c2w, bound_ray) in enumerate(zip(c2w_mats, bound_rays)):
                if step == 0:       # gt
                    line_color = "red"
                    c = 'red'
                    face_color = "red"
                    marker = 'd'
                    s = 50
                else:
                    line_color = colors[step-1]
                    c = colors[step-1]
                    face_color = "gray"
                    marker = 'o'
                    s = 30
                
                # 인자로 받은 figure에 카메라 하나(c2w, bound_ray)를 그린다.
                draw_camera(ax, c2w, bound_ray, c, line_color, face_color, marker, s)
                
            figs.append(fig)
                
        return figs
    # figure 1개에 모든 소스 카메라를 함께 그리기
    else:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_xlabel("X_world")
        ax.set_ylabel("Y_world")
        ax.set_zlabel("Z_world")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=30, azim=120)

        for cam_idx in range(0, gt_c2w_mats.shape[0]):
            gt_c2w_mat = gt_c2w_mats[[cam_idx]]   # (1, 4, 4)
            gt_bound_ray = gt_world_bound_rays[[cam_idx]]     # (1, 4, 3)
            noise_c2w_mat = noise_c2w_mats[cam_idx]      # (optim_step, 4, 4)
            noise_bound_ray = noise_world_bound_rays[cam_idx]   # (optim_step, 4, 3)

            c2w_mats = torch.cat((gt_c2w_mat, noise_c2w_mat), dim=0)        # (1+optim_step, 4, 4)
            bound_rays = torch.cat((gt_bound_ray, noise_bound_ray), dim=0)  # (1+optim_step, 4, 3)
            
            
            for step, (c2w, bound_ray) in enumerate(zip(c2w_mats, bound_rays)):
                if step == 0:       # gt
                    line_color = "red"
                    c = 'red'
                    face_color = "red"
                    marker = 'd'
                    s = 1
                    before_T = c2w[3, :3]
                else:
                    line_color = "gray"
                    c = "gray"
                    face_color = "gray"
                    marker = 'o'
                    s = 1
                    noise_T = c2w[3, :3]
                    x = before_T[0].item()
                    y = before_T[1].item()
                    z = before_T[2].item()
                    # orig 카메라와 noise 카메라 사이의 translation error를 그린다.
                    xs = [x, noise_T[0].item()]
                    ys = [y, noise_T[1].item()]
                    zs = [z, noise_T[2].item()]
                    
                    before_T = noise_T

                    a = Arrow3D(xs, ys, zs, mutation_scale=15, 
                                        lw=2., arrowstyle="-", color='red')
                    ax.add_artist(a)
                
                # 인자로 받은 figure에 카메라 하나(c2w, bound_ray)를 그린다.
                draw_camera(ax, c2w, bound_ray, c, line_color, face_color, marker, s)
                
        return fig
    
    
def get_correspondences_figure(imgs, noise_camera, noise_pair_rays, optimized_camera, optim_pair_rays, pair, result, with_colmap):
    # result: correspondencs 값 (2, corr, 2)
    # pair_rays: result로 향하는 rays {"rays_d":(2, corr, 2), "rays_o":(2, corr, 2)}
    imgs = imgs.detach().cpu().numpy()
    
    if with_colmap:
        initial_camera_type = "COLMAP GT"
    else:
        initial_camera_type = "(COLAMAP GT + Noise)"
        
    keypoint_idx = 10
    i, j = pair     # (camera_i, camera_j)
    noise_proejction_point_i, noise_proejction_point_j = proj_ray_dist_loss_single(result[0, [keypoint_idx]], result[1, [keypoint_idx]], 
                                                      noise_camera, (i, j), noise_pair_rays, "test", 800, 800, only_projection_points=True)   
    optim_proejction_point_i, optim_proejction_point_j = proj_ray_dist_loss_single(result[0, [keypoint_idx]], result[1, [keypoint_idx]], 
                                                      optimized_camera, (i, j), optim_pair_rays, "test", 800, 800, only_projection_points=True)
    
    fig = plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0, hspace=0.1, left=0.1, right=0.9, bottom=0, top=1)
    
    i_x, i_y = result[0, keypoint_idx]
    j_x, j_y = result[1, keypoint_idx]
    
    # 최적화 전 카메라의 투영 결과 시각화
    ax = fig.add_subplot(2, 2, 1)
    ax.axis("off")
    ax.set_title("%s Camera - Image %d"%(initial_camera_type, i))
    ax.imshow(imgs[i])
    ax.scatter(i_x.item(), i_y.item(), c="yellow", s=50)
    # print(noise_proejction_point_i)
    if len(noise_proejction_point_i) != 0:
        ax.scatter(noise_proejction_point_i[0, 0].item(), noise_proejction_point_i[0, 1].item(),  c="magenta", marker="*", s=40)

    ax = fig.add_subplot(2, 2, 2)
    ax.axis("off")
    ax.set_title("%s Camera - Image %d"%(initial_camera_type, j))
    ax.imshow(imgs[j])
    ax.scatter(j_x.item(), j_y.item(), c="yellow", s=50)
    if len(noise_proejction_point_j) != 0:
        ax.scatter(noise_proejction_point_j[0, 0].item(), noise_proejction_point_j[0, 1].item(),  c="magenta", marker="*", s=40)
    
    # 최적화 후 카메라의 투영 결과 시각화
    ax = fig.add_subplot(2, 2, 3)
    ax.axis("off")
    ax.set_title("Final Optimized Camera - Image %d"%i)
    ax.imshow(imgs[i])
    ax.scatter(i_x.item(), i_y.item(), c="yellow", s=50)
    if len(optim_proejction_point_i) != 0:
        ax.scatter(optim_proejction_point_i[0, 0].item(), optim_proejction_point_i[0, 1].item(),  c="magenta", marker="*", s=40)

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    ax.set_title("Final Optimized Camera - Image %d"%j)
    ax.imshow(imgs[j])
    ax.scatter(j_x.item(), j_y.item(), c="yellow", s=50)
    if len(optim_proejction_point_j) != 0:
        ax.scatter(optim_proejction_point_j[0, 0].item(), optim_proejction_point_j[0, 1].item(),  c="magenta", marker="*", s=40)
    
    return fig


########################################### target과 가까운 source view 계산하기 ###########################################
TINY_NUMBER = 1e-6
# Roation matrix 사이의 angular distance를 측정
# TODO: np 쓰는 부분 torch로 바꿔야 함. 에러난다.
def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3

    return torch.arccos(torch.clip((torch.diagonal(torch.matmul(R2.permute(0, 2, 1), R1), dim1=-2, dim2=-1).sum(-1) - 1) / 2.,
                             min=-1 + TINY_NUMBER, max=1 - TINY_NUMBER))


# 두 벡터(두 카메라의 위치) 사이의 거리를 측정
def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (torch.norm(vec1, dim=1, keepdim=True) + TINY_NUMBER)
    vec2_unit = vec2 / (torch.norm(vec2, dim=1, keepdim=True) + TINY_NUMBER)
    angular_dists = torch.arccos(torch.clamp(torch.sum(vec1_unit*vec2_unit, dim=-1), min=-1.0, max=1.0))
    
    return angular_dists


def get_nearest_src(tgt_pose, src_poses, num_select, tgt_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tgt_pose: target pose [3, 3] ?? [3, 4]여야 하는거 아닌가
        src_poses: reference poses [N, 3, 3] ?? 이것도 [N, 3, 4]...
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(src_poses)
    num_select = min(num_select, num_cams)
    batched_tgt_pose = tgt_pose[None, ...].repeat(num_cams, 1, 1)
    

    # 타겟뷰와 소스뷰의 유사성 판단 메소드 : matrix / vector / dist
    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tgt_pose[:, :3, :3], src_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        # 타겟뷰와 소스뷰의 translation 값을 가져옴
        tgt_cam_locs = batched_tgt_pose[:, 3, :3]
        ref_cam_locs = src_poses[:, 3, :3]
        # 씬의 중앙으로부터 각 뷰가 떨어진 거리를 측정
        scene_center = torch.tensor(scene_center, device=tgt_cam_locs.device)[None, ...]
        tar_vectors = tgt_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        # 이 거리들 사이의 차이를 측정
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    # 단순히 타겟뷰와 소스뷰의 translation 차이를 측정
    elif angular_dist_method == 'dist':
        tgt_cam_locs = batched_tgt_pose[:, 3, :3]
        ref_cam_locs = src_poses[:, 3, :3]
        dists = torch.norm(tgt_cam_locs - ref_cam_locs, dim=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tgt_id >= 0:
        assert tgt_id < num_cams
        dists[tgt_id] = 1e3  # 타겟뷰는 선택되지 않도록 방지한다.

    # dists 값이 작은 순으로 "인덱스"를 정렬
    sorted_ids = torch.argsort(dists)
    # 정렬된 인덱스를 소스뷰 개수만큼 선택
    selected_ids = sorted_ids[:num_select]

    return selected_ids


########################################### 텐서보드 시각화에 사용되는 함수들 ###########################################
HUGE_NUMBER = 1e10

def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new


def colorize(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x


def log_view_to_tensorboard_pl(coarse_net, fine_net, ray_sampler, feature_maps, srcs, pe_dim, gts,
                                chunk_size, render_stride, N_samples, N_importance, inv_uniform, det, 
                                model_type, with_colmap, training_phase, masking):
    coarse_net.eval()
    fine_net.eval()

    with torch.no_grad():
        output = render_image(ray_sampler,
                            coarse_net, fine_net, feature_maps, pe_dim,
                            chunk_size, render_stride,
                            N_samples, N_importance, inv_uniform, det, model_type, with_colmap, training_phase,
                            masking)

    srcs = srcs["rgb"].detach().cpu()
    N_src = srcs.shape[0]

    if render_stride != 1:
        rgb_gt = gts["img"][::render_stride, ::render_stride]
        srcs = srcs[:, :, ::render_stride, ::render_stride]

    img_HWC2CHW = lambda x: x.permute(2, 0, 1)
    
    rgb_gt = img_HWC2CHW(gts["img"])
    rgb_pred = img_HWC2CHW(output['outputs_coarse']['rgb'].detach().cpu())

    h = rgb_gt.shape[-2]
    w = rgb_gt.shape[-1]

    srcs_im = torch.zeros(3, h, N_src*w)
    for i in range(0, N_src):
        srcs_im[:, :h, i*w:(i+1)*w] = srcs[i]

    rgb_im = torch.zeros(3, h, 3*w)
    rgb_im[:, :rgb_gt.shape[-2], :rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], w:w+rgb_pred.shape[-1]] = rgb_pred

    depth_im = output['outputs_coarse']['depth'].detach().cpu()

    # acc_map = torch.sum(output['outputs_coarse']['weights'], dim=-1).detach().cpu()
    mask_gt = img_HWC2CHW(gts["mask"]).detach().cpu()
    mask_pred = img_HWC2CHW(output["outputs_coarse"]["mask"].detach().cpu().unsqueeze(-1))

    if output['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        depth_im = img_HWC2CHW(depth_im)
        # acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
        mask_im = torch.cat((mask_gt, mask_pred), dim=-1)
    else:
        rgb_fine = img_HWC2CHW(output['outputs_fine']['rgb'].detach().cpu())
        rgb_im[:, :rgb_fine.shape[-2], 2*w:2*w+rgb_fine.shape[-1]] = rgb_fine
        depth_im = torch.cat((depth_im, output['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))

        # acc_map = torch.cat((acc_map, torch.sum(output['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        # acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
        mask_fine = img_HWC2CHW(output["outputs_fine"]["mask"].detach().cpu().unsqueeze(-1))
        mask_im = torch.cat((mask_gt, mask_pred, mask_fine), dim=-1)

    tb_imgs = {
                "src": srcs_im,
                "rgb":rgb_im,
                "depth":depth_im,
                "mask":mask_im,
    }
    # write scalar
    # pred_rgb = output['outputs_fine']['rgb'] if output['outputs_fine'] is not None else output['outputs_coarse']['rgb']
    # psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    # writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    coarse_net.train()
    fine_net.train()

    return tb_imgs

############################################################################################################