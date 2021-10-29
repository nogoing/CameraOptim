import torch
import os
import numpy as np
import cv2
import json, gzip



# 특정 카테고리 안의 n개 오브젝트에 대한 프레임 시퀀스가 전부 하나의 파일에 저장되어 있음.
# ex) teddybear 카테고리 하나 당 annotation file 한 개.

# 이 annotation 파일을 읽어와 각 오브젝트 별로 구분하여 json 파일을 생성한다.
def generate_co3d_json_file(root_path):
    categories = os.listdir(root_path)

    for category in categories:
        category_path =  os.path.join(root_path, category)
        
        frame_annotations_file_path = os.path.join(category_path, "frame_annotations.jgz")
        frame_zipfile = gzip.open(frame_annotations_file_path, "rt", encoding="utf8")
        frame_dicts = json.load(frame_zipfile)

        seq_name_prev = frame_dicts[0]["sequence_name"]

        json_data = []
        for frame_dict in (frame_dicts):
            seq_name = frame_dict["sequence_name"]
            
            if seq_name != seq_name_prev:
                if seq_name_prev != "":
                    with open(os.path.join(category_path, seq_name_prev, "frame_annotations_file.json"), 'w', encoding="utf-8") as f:
                        json.dump(json_data, f, ensure_ascii=False, indent="\t")
                
                json_data = []
                seq_name_prev = seq_name

            json_data.append({
                "frame_number" : frame_dict["frame_number"],
                "frame_timestamp" : frame_dict["frame_timestamp"],
                "image" : frame_dict["image"],
                "depth" : frame_dict["depth"],
                "mask" : frame_dict["mask"],
                "viewpoint" : frame_dict["viewpoint"]
            })


def get_c2w_intrinsic(img_size, viewpoint):
    rotation = torch.tensor(viewpoint['R'])
    translation =  torch.tensor(viewpoint['T'])
    
    extrinsic = torch.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    # extrinsic[1:3, :] *= -1.
    w2c = extrinsic
    c2w = torch.inverse(w2c)
    
    focal_length =  torch.tensor(viewpoint["focal_length"])
    principal_point =  torch.tensor(viewpoint["principal_point"])

    # principal point and focal length in pixels
    half_image_size_wh_orig = torch.tensor([x/2 for x in img_size])
    principal_point_px = -1.0 * (principal_point - 1.0) * half_image_size_wh_orig
    focal_length_px = focal_length * half_image_size_wh_orig
    # if self.box_crop:
    #     assert clamp_bbox_xyxy is not None
    #     principal_point_px -= clamp_bbox_xyxy[:2]

    # # rescaled principal point and focal length in ndc
    # principal_point = 1 - principal_point_px * scale / half_image_size_wh_output
    # focal_length = focal_length_px * scale / half_image_size_wh_output

    intrinsic = torch.eye(4)
    intrinsic[0][2], intrinsic[1][2] = principal_point_px
    intrinsic[0][0], intrinsic[1][1] = focal_length_px

    return c2w, intrinsic


# 하나의 sequence (= 오브젝트 한 개) data를 읽어오는 함수
def read_seq_data(seq_path, normalization=True):
    frame_annotations_file = os.path.join(seq_path, "frame_annotations_file.json")
    with open(frame_annotations_file, 'r') as f:
        frames = json.load(f)

    seq_imgs = []
    seq_masks = []
    seq_c2w_mats = []        # extrinsics
    seq_intrinsic_mats = []

    xs = []
    ys = []
    zs = []

    for frame in frames:
        seq_imgs.append(frame["image"]["path"])
        seq_masks.append(frame["mask"]["path"])
        
        c2w, intrinsic = get_c2w_intrinsic(frame["image"]["size"], frame["viewpoint"])

        # max_ = c2w[:3, 3].max()
        # if max_ > max_bd:
        #     max_bd = max_
        # min_ = c2w[:3, 3].min()
        # if min_ < min_bd:
        #     min_bd = min_
        if (normalization):
            xs.append(c2w[0, 3].item())
            ys.append(c2w[1, 3].item())
            zs.append(c2w[2, 3].item())
        
        seq_c2w_mats.append(c2w)
        
        seq_intrinsic_mats.append(intrinsic)
    
    if (normalization):
        x_bd = [min(xs), max(xs)]
        y_bd = [min(ys), max(ys)]
        z_bd = [min(zs), max(zs)]
        for c2w in seq_c2w_mats:
            c2w[0, 3] = 2*((c2w[0, 3] - x_bd[0])/(x_bd[1] - x_bd[0])) - 1
            c2w[1, 3] = 2*((c2w[1, 3] - y_bd[0])/(y_bd[1] - y_bd[0])) - 1
            c2w[2, 3] = 2*((c2w[2, 3] - z_bd[0])/(z_bd[1] - z_bd[0])) - 1
        # for c2w in seq_c2w_mats:
        #     c2w[:3, 3] = 2*((c2w[:3, 3] - min_bd)/(max_bd - min_bd)) - 1

    return seq_imgs, seq_masks, seq_c2w_mats, seq_intrinsic_mats


# 하나의 category (= 해당 카테고리 내 모든 오브젝트) data를 읽어오는 함수
def read_category_data(category_path, normalization=True):
    category_imgs = []
    category_masks = []
    category_c2w_mats = []
    category_intrinsic_mats = []

    # 해당 카테고리 내의 모든 sequence 목록을 가져온다.
    seq_file_path = os.path.join(category_path, "sequence_annotations.jgz")
    seq_zipfile = gzip.open(seq_file_path, "rt", encoding="utf8")
    seq_dicts = json.load(seq_zipfile)

    # viewpoint_qualiry_score가 높은 순으로 sequences를 정렬한 뒤 상위 50퍼만 사용.
    sorted_seq_dicts = sorted(seq_dicts, key=lambda x : x["viewpoint_quality_score"], reverse=True)
    sorted_seq_dicts = sorted_seq_dicts[:int(len(sorted_seq_dicts)*0.5)]
    choosen_seqs = [seq_dic["sequence_name"] for seq_dic in sorted_seq_dicts]

    for seq_name in choosen_seqs:
        seq_path = os.path.join(category_path, seq_name)
        seq_imgs, seq_masks, seq_c2w_mats, seq_intrinsic_mats = read_seq_data(seq_path, normalization)

        category_imgs.extend(seq_imgs)
        category_masks.extend(seq_masks)
        category_c2w_mats.extend(seq_c2w_mats)
        category_intrinsic_mats.extend(seq_intrinsic_mats)

    return category_imgs, category_masks, category_c2w_mats, category_intrinsic_mats


# Roation matrix 사이의 angular distance를 측정
TINY_NUMBER = 1e-6
def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


# 두 벡터(두 카메라의 위치) 사이의 거리를 측정
def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    
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
    num_select = min(num_select, num_cams-1)
    batched_tgt_pose = tgt_pose[None, ...].repeat(num_cams, 0)

    # 타겟뷰와 소스뷰의 유사성 판단 메소드 : matrix / vector / dist
    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tgt_pose[:, :3, :3], src_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        # 타겟뷰와 소스뷰의 translation 값을 가져옴
        tgt_cam_locs = batched_tgt_pose[:, :3, 3]
        ref_cam_locs = src_poses[:, :3, 3]
        # 씬의 중앙으로부터 각 뷰가 떨어진 거리를 측정
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tgt_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        # 이 거리들 사이의 차이를 측정
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    # 단순히 타겟뷰와 소스뷰의 translation 차이를 측정
    elif angular_dist_method == 'dist':
        tgt_cam_locs = batched_tgt_pose[:, :3, 3]
        ref_cam_locs = src_poses[:, :3, 3]
        dists = np.linalg.norm(tgt_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tgt_id >= 0:
        assert tgt_id < num_cams
        dists[tgt_id] = 1e3  # 타겟뷰는 선택되지 않도록 방지한다.

    # dists 값이 작은 순으로 "인덱스"를 정렬
    sorted_ids = np.argsort(dists)
    # 정렬된 인덱스를 소스뷰 개수만큼 선택
    selected_ids = sorted_ids[:num_select]

    return selected_ids


def rectify_inplane_rotation(src_pose, tar_pose, src_img, th=40):
    relative = np.linalg.inv(tar_pose).dot(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler('zxy', degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler('z', -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.) / 2., (h - 1.) / 2.)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255*src_img).astype(np.uint8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    rotated = rotated.astype(np.float32) / 255.
    return out_pose, rotated