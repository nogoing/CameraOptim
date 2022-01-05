import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
import numpy as np

from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl

from rendering import render_image


def data_to_frame(co3d_data, N_src):
    imgs = co3d_data.image_rgb
    img_paths = co3d_data.image_path
    masks = co3d_data.fg_probability
    camera = co3d_data.camera

    c2ws = camera.get_world_to_view_transform().inverse().get_matrix()
    tgt_idx = 0
    tgt_c2w = c2ws[tgt_idx]
    src_c2ws = c2ws[1:]
    src_idxs = get_nearest_src(tgt_c2w, src_c2ws, N_src) + 1

    target = {}
    target["rgb"] = imgs[tgt_idx]
    target["rgb_path"] = img_paths[tgt_idx]
    target["mask"] = masks[tgt_idx]
    tgt_principal_point = camera.principal_point[tgt_idx]
    tgt_focal_length = camera.focal_length[tgt_idx]
    tgt_R = camera.R[tgt_idx]
    tgt_T = camera.T[tgt_idx]
    target["camera"] = PerspectiveCameras(
                                            focal_length=tgt_focal_length[None],
                                            principal_point=tgt_principal_point[None],
                                            R=tgt_R[None],
                                            T=tgt_T[None],
                                        ).to(camera.device)
    # 현재 타겟 카메라의 위치를 기준으로
    # 샘플링할 ray들의 depth range 계산
    target_camera_position = camera.get_camera_center()[tgt_idx]
    near = torch.norm(target_camera_position) - 8
    far = torch.norm(target_camera_position) + 8
    depth_range =  torch.tensor([near, far], dtype=torch.float32, device=imgs.device)
    target["depth_range"] = depth_range

    srcs = {}
    # srcs["rgb"] = imgs[src_idxs].to(device)
    # srcs["mask"] = masks[src_idxs].to(device)
    srcs["rgb"] = imgs[src_idxs]
    srcs["rgb_path"] = [img_paths[i] for i in src_idxs]
    srcs["mask"] = masks[src_idxs]
    src_principal_points = camera.principal_point[src_idxs]
    src_focal_lengths = camera.focal_length[src_idxs]
    src_Rs = camera.R[src_idxs]
    src_Ts = camera.T[src_idxs]
    srcs["camera"] = PerspectiveCameras(
                                        focal_length=src_focal_lengths,
                                        principal_point=src_principal_points,
                                        R=src_Rs,
                                        T=src_Ts,
                                    ).to(camera.device)
    
    return target, srcs


########################################### target과 가까운 source view 선택하기 ###########################################
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

    return torch.arccos(torch.clip((torch.diagonal(torch.matmul(R2_tensor.permute(0, 2, 1), R1_tensor), dim1=-2, dim2=-1).sum(-1) - 1) / 2.,
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


def log_view_to_tensorboard(args, writer, global_step, coarse_net, fine_net, ray_sampler, feature_maps, srcs, PE, gts, prefix=''):
    coarse_net.eval()
    fine_net.eval()

    with torch.no_grad():
        # 이미지의 모든 픽셀에 대해 정의된 ray batch 생성.
        ray_batch = ray_sampler.get_all()

        output = render_image(ray_sampler=ray_sampler,
                            ray_batch=ray_batch,
                            coarse_model=coarse_net,
                            fine_model=fine_net,
                            feature_maps=feature_maps,
                            PE=PE,
                            args=args,
                            )

    srcs = srcs["rgb"].detach().cpu()
    N_src = srcs.shape[0]

    if args.render_stride != 1:
        gt_img = gt_img[::args.render_stride, ::args.render_stride]
        srcs = srcs[:, :, ::args.render_stride, ::args.render_stride]

    img_HWC2CHW = lambda x: x.permute(2, 0, 1)
    gt_img = gts["img"]
    gt_mask = gts["mask"]

    rgb_gt = img_HWC2CHW(gt_img)
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
    mask_gt = img_HWC2CHW(gt_mask).detach().cpu()
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

    # 위에서 만든 시각화 이미지를 텐서보드에 기록
    writer.add_image(prefix + 'rgb_sources', srcs_im, global_step)
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    # writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)
    writer.add_image(prefix + 'mask_gt-coarse-fine', mask_im, global_step)

    # write scalar
    # pred_rgb = output['outputs_fine']['rgb'] if output['outputs_fine'] is not None else output['outputs_coarse']['rgb']
    # psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    # writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    coarse_net.train()
    fine_net.train()


def log_view_to_tensorboard_pl(args, coarse_net, fine_net, ray_sampler, feature_maps, srcs, PE, gts):
    coarse_net.eval()
    fine_net.eval()

    with torch.no_grad():
        # 이미지의 모든 픽셀에 대해 정의된 ray batch 생성.
        ray_batch = ray_sampler.get_all()

        output = render_image(ray_sampler=ray_sampler,
                            ray_batch=ray_batch,
                            coarse_model=coarse_net,
                            fine_model=fine_net,
                            feature_maps=feature_maps,
                            PE=PE,
                            args=args,
                            )

    srcs = srcs["rgb"].detach().cpu()
    N_src = srcs.shape[0]

    if args.render_stride != 1:
        rgb_gt = gts["img"][::args.render_stride, ::args.render_stride]
        srcs = srcs[:, :, ::args.render_stride, ::args.render_stride]

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