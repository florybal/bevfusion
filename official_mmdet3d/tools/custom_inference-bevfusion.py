import os, argparse, numpy as np, torch
np.Inf = np.inf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from mmengine import Config
from mmdet3d.apis import init_model
from mmdet3d.structures import LiDARInstance3DBoxes, Det3DDataSample
from nuscenes.nuscenes import NuScenes

CONFIG_FILE = "projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CHECKPOINT = "/workspace/results/training/epoch_6.pth"
DATA_ROOT = "data/nuscenes"

CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

BEV_X_RANGE = [-54, 54]
BEV_Y_RANGE = [-54, 54]

def get_sample_inputs(token, nusc):
    sample = nusc.get('sample', token)
    cam_names = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                 'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    target_size = (256, 704)

    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pcd_path = os.path.join(DATA_ROOT, lidar_data['filename'])
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)

    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3,1,1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3,1,1)

    imgs = []
    cam2imgs = []
    lidar2cams = []
    cam2lidars = []
    lidar2imgs = []
    img_aug_matrices = []

    for cam in cam_names:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        img_path = os.path.join(DATA_ROOT, cam_data['filename'])
        img = plt.imread(img_path).copy()
        img = torch.from_numpy(img).float().permute(2, 0, 1)
        orig_h, orig_w = img.shape[1], img.shape[2]

        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(cam_calib['camera_intrinsic'])[:3, :3]
        scale_w = target_size[1] / orig_w
        scale_h = target_size[0] / orig_h
        intrinsic[0, 0] *= scale_w
        intrinsic[0, 2] *= scale_w
        intrinsic[1, 1] *= scale_h
        intrinsic[1, 2] *= scale_h
        cam2imgs.append(torch.from_numpy(intrinsic).float())

        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(
            nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])['rotation']
        ).rotation_matrix
        lidar2ego[:3,3] = np.array(
            nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])['translation']
        )
        cam2ego = np.eye(4)
        cam2ego[:3,:3] = Quaternion(cam_calib['rotation']).rotation_matrix
        cam2ego[:3,3] = np.array(cam_calib['translation'])
        lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
        cam2lidar = np.linalg.inv(lidar2cam)
        lidar2img = intrinsic @ lidar2cam[:3, :4]

        lidar2cams.append(torch.from_numpy(lidar2cam).float())
        cam2lidars.append(torch.from_numpy(cam2lidar).float())
        lidar2imgs.append(torch.from_numpy(lidar2img).float())

        I_3x4 = torch.eye(4)[:3]
        img_aug_matrices.append(I_3x4)

        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0)
        img = (img * 255.0 - mean) / std
        imgs.append(img)

    imgs = torch.stack(imgs)
    meta = {
        'cam2img': torch.stack(cam2imgs),
        'lidar2cam': torch.stack(lidar2cams),
        'cam2lidar': torch.stack(cam2lidars),
        'lidar2img': torch.stack(lidar2imgs),
        'img_aug_matrix': torch.stack(img_aug_matrices),
        'ori_cam2img': torch.stack(cam2imgs),
        'ori_lidar2img': torch.stack(lidar2imgs),
        'img_shape': target_size,
        'box_type_3d': LiDARInstance3DBoxes,
        'token': token,
        'sample_idx': 0,
    }
    return points, imgs, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='fd8420396768425eabec9bdddf7e64b6')
    parser.add_argument('--score-thr', type=float, default=0.3)
    args = parser.parse_args()

    cfg = Config.fromfile(CONFIG_FILE)
    model = init_model(cfg, CHECKPOINT, device='cuda:0')
    model.eval()

    nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=False)
    points, imgs, meta = get_sample_inputs(args.token, nusc)

    batch_points = torch.from_numpy(points).float().unsqueeze(0).cuda()
    batch_imgs = imgs.unsqueeze(0).cuda()
    dummy_sample = Det3DDataSample()
    dummy_sample.set_metainfo(meta)
    batch_data_samples = [dummy_sample]

    with torch.no_grad():
        results = model.predict(
            batch_inputs_dict={'points': batch_points, 'imgs': batch_imgs},
            batch_data_samples=batch_data_samples
        )
    result = results[0]

    # Caminho da imagem da câmera frontal
    cam_front_data = nusc.get('sample_data',
                              nusc.get('sample', args.token)['data']['CAM_FRONT'])
    img_path = os.path.join(DATA_ROOT, cam_front_data['filename'])
    front_img = plt.imread(img_path)

    # =====================================================================
    # Cria figura combinada: front + BEV
    # =====================================================================
    fig, (ax_front, ax_bev) = plt.subplots(1, 2, figsize=(20, 8))

    # ---- FRONT VIEW ----
    ax_front.imshow(front_img)
    ax_front.set_xlim(0, front_img.shape[1])
    ax_front.set_ylim(front_img.shape[0], 0)
    ax_front.set_title('Camera (Front)')

    # ---- BEV VIEW ----
    ax_bev.set_xlim(BEV_X_RANGE)
    ax_bev.set_ylim(BEV_Y_RANGE)
    ax_bev.set_aspect('equal')
    ax_bev.set_xlabel('X (m)')
    ax_bev.set_ylabel('Y (m)')
    ax_bev.grid(True, linestyle='--', alpha=0.7)
    ax_bev.plot(0, 0, 's', color='black', markersize=6)   # ego vehicle
    ax_bev.set_title('BEV')

    if hasattr(result, 'pred_instances_3d'):
        pred = result.pred_instances_3d
        keep = pred.scores_3d > args.score_thr
        boxes = pred.bboxes_3d[keep].tensor.cpu().numpy()
        labels = pred.labels_3d[keep].cpu().numpy()
        scores = pred.scores_3d[keep].cpu().numpy()

        # Parâmetros para projeção na câmera frontal
        cam_calib = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
        intrinsic_orig = np.array(cam_calib['camera_intrinsic'])[:3, :3]
        lidar2cam = meta['lidar2cam'][0].numpy()

        # Cores para as classes
        color_map = {
            'car': 'blue',
            'truck': 'green',
            'construction_vehicle': 'orange',
            'bus': 'purple',
            'trailer': 'brown',
            'barrier': 'red',
            'motorcycle': 'cyan',
            'bicycle': 'magenta',
            'pedestrian': 'yellow',
            'traffic_cone': 'gray'
        }

        for i, box in enumerate(boxes):
            if len(box) == 9:
                x, y, z, l, w, h, yaw, vx, vy = box
            else:
                x, y, z, l, w, h, yaw = box

            # ----- Projeção na câmera (FRONT) -----
            corners_3d = np.array([
                [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2], [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
                [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2], [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2]
            ])
            rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw),  np.cos(yaw), 0],
                            [0, 0, 1]])
            corners_3d = (rot @ corners_3d.T).T + np.array([x, y, z])
            pts_4d = np.hstack([corners_3d, np.ones((8, 1))])
            pts_cam = (lidar2cam @ pts_4d.T).T[:, :3]
            pts_2d = intrinsic_orig @ pts_cam.T
            pts_2d = pts_2d[:2] / pts_2d[2, :]
            pts_2d = pts_2d.T

            # Desenha no front se algum canto estiver visível
            if np.any((pts_2d[:, 0] > 0) & (pts_2d[:, 0] < front_img.shape[1]) &
                      (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < front_img.shape[0])):
                ax_front.plot(pts_2d[[0,1,2,3,0,4,5,1,5,6,2,6,7,3,7,4], 0],
                              pts_2d[[0,1,2,3,0,4,5,1,5,6,2,6,7,3,7,4], 1], 'r-', linewidth=1.5)
                ax_front.fill(pts_2d[[0,1,2,3], 0], pts_2d[[0,1,2,3], 1], alpha=0.2, color='red')
                ax_front.text(pts_2d[0, 0], pts_2d[0, 1] - 8,
                              f'{CLASSES[labels[i]]} {scores[i]:.2f}',
                              color='red', fontsize=6, bbox=dict(facecolor='white', alpha=0.7))

            # ----- Desenho no BEV (top‑down) -----
            class_name = CLASSES[labels[i]]
            color = color_map.get(class_name, 'blue')
            corners_bev = np.array([
                [ l/2,  w/2],
                [ l/2, -w/2],
                [-l/2, -w/2],
                [-l/2,  w/2]
            ])
            rot_bev = np.array([[np.cos(yaw), -np.sin(yaw)],
                                [np.sin(yaw),  np.cos(yaw)]])
            corners_bev = (rot_bev @ corners_bev.T).T + np.array([x, y])
            ax_bev.add_patch(patches.Polygon(corners_bev, closed=True, fill=True,
                                             alpha=0.3, color=color))
            ax_bev.plot(corners_bev[[0,1,2,3,0], 0], corners_bev[[0,1,2,3,0], 1],
                        color=color, linewidth=1)
            ax_bev.text(x, y, f'{class_name} {scores[i]:.2f}',
                        fontsize=5, ha='center', va='bottom', color=color)

    plt.tight_layout()
    output_file = 'bevfusion_inference.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"view saved to {output_file}")

if __name__ == '__main__':
    main()