#!/usr/bin/env python3
"""
Gera PNGs para câmera, BEV com pred+GT e BEV só GT.
Uso: python tools/video_inference/make_pngsGT.py
"""

import os
import sys
import numpy as np
np.Inf = np.inf
np.NaN = np.nan

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
import cv2
import mmengine

from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.apis import MultiModalityDet3DInferencer

# ===== CONFIGURAÇÕES =====
CONFIG_FILE = "projects/BEVFusion/configs/bevfusion_plastipak.py"
CHECKPOINT = "/workspace/results/training/plastipak/30epoch/epoch_20.pth"  
DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
PKL_FILE = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_train.pkl"
OUTPUT_DIR = "./output/qualitative_vis/bevfusion/frames"
SCORE_THR = 0.01
MAX_FRAMES = 0  # 0 = todos os frames

CLASSES = ('obstrucao', 'empilhadeira', 'carga', 'maquina',
           'humano', 'navegavel', 'estrutura', 'portapalete')
COLOR_MAP = {
    'obstrucao': 'red', 'empilhadeira': 'orange', 'carga': 'magenta',
    'maquina': 'blue', 'humano': 'cyan', 'navegavel': 'green',
    'estrutura': 'yellow', 'portapalete': 'brown',
}
BOX_EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

# ===== Funções auxiliares =====
def box_utils_boxes_to_corners_3d(boxes):
    N = boxes.shape[0]
    corners = np.zeros((N, 8, 3), dtype=np.float32)
    for i in range(N):
        x, y, z, w, l, h, yaw = boxes[i]
        corners_local = np.array([
            [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2],
            [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
            [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2],
            [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2]
        ])
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw),  np.cos(yaw), 0],
                        [0, 0, 1]])
        corners[i] = (rot @ corners_local.T).T + np.array([x, y, z])
    return corners

def _box_corners_xy(boxes):
    return box_utils_boxes_to_corners_3d(boxes[:, :7])[:, :4, :2]

def _project_corners(boxes, l2i, aug):
    corners = box_utils_boxes_to_corners_3d(boxes[:, :7]).astype(np.float32)
    ch = np.concatenate([corners, np.ones((*corners.shape[:2], 1), np.float32)], -1)
    proj = ch @ l2i.T
    d = proj[..., 2]
    valid = d > 1e-4
    xy = np.zeros((*corners.shape[:2], 2), np.float32)
    xy[...,0] = proj[...,0] / np.clip(d, 1e-4, None)
    xy[...,1] = proj[...,1] / np.clip(d, 1e-4, None)
    xyh = np.concatenate([xy, np.ones((*xy.shape[:2], 2), np.float32)], -1)
    return (xyh @ aug.T)[..., :2], valid

# ===== Desenho =====
def draw_camera(ax, img_rgb, boxes, scores, labels, class_names, l2i, aug, score_thresh):
    h, w = img_rgb.shape[:2]
    ax.imshow(img_rgb); ax.axis("off"); ax.set_title("Câmera ZED-left")
    if boxes is None or boxes.size == 0: return
    c2d, vm = _project_corners(boxes, l2i, aug)
    for i in range(boxes.shape[0]):
        if scores[i] < score_thresh: continue
        color = COLOR_MAP.get(class_names[int(labels[i])], "#ffcc00")
        for s,e in BOX_EDGES:
            if not (vm[i,s] and vm[i,e]): continue
            p0, p1 = c2d[i,s], c2d[i,e]
            ok0 = 0 <= p0[0] < w and 0 <= p0[1] < h
            ok1 = 0 <= p1[0] < w and 0 <= p1[1] < h
            if not (ok0 or ok1): continue
            ax.plot([np.clip(p0[0],0,w), np.clip(p1[0],0,w)],
                    [np.clip(p0[1],0,h), np.clip(p1[1],0,h)],
                    color=color, linewidth=1.4)

def draw_bev(ax, points, boxes, scores, labels, class_names, pc_range, score_thresh,
             gt_boxes=None, gt_labels=None, title="BEV"):
    xlim = (pc_range[0], pc_range[3]); ylim = (pc_range[1], pc_range[4])
    mask = ((points[:,0]>=xlim[0])&(points[:,0]<=xlim[1])&
            (points[:,1]>=ylim[0])&(points[:,1]<=ylim[1]))
    if mask.any():
        ax.scatter(points[mask,0], points[mask,1], s=0.3, c="dimgray", alpha=0.4, zorder=1)
    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        bottoms = _box_corners_xy(gt_boxes)
        for i in range(gt_boxes.shape[0]):
            cls = class_names[int(gt_labels[i])]
            color = COLOR_MAP.get(cls, "#aaaaaa")
            ax.add_patch(Polygon(bottoms[i], closed=True, fill=False,
                                 edgecolor=color, lw=1.5, ls="--", alpha=0.7))
    if boxes is not None and boxes.size > 0:
        bottoms = _box_corners_xy(boxes)
        for i in range(boxes.shape[0]):
            if scores[i] < score_thresh: continue
            cls = class_names[int(labels[i])]
            color = COLOR_MAP.get(cls, "#ffcc00")
            ax.add_patch(Polygon(bottoms[i], closed=True, fill=False, edgecolor=color, lw=1.8))
            cx, cy = boxes[i,0], boxes[i,1]
            ax.text(cx, cy, f"{cls[:3]}\n{scores[i]:.2f}", color=color, fontsize=6,
                    ha="center", va="center")
    handles = [Patch(facecolor=c, edgecolor="none", label=n) for n,c in COLOR_MAP.items()]
    handles.append(Line2D([0],[0], color="white", lw=1.3, ls="--", label="GT", alpha=0.75))
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal","box")
    ax.grid(alpha=0.15); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(title)

# ===== MAIN =====
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_list = mmengine.load(PKL_FILE)
    total = len(data_list)
    if MAX_FRAMES > 0:
        total = min(MAX_FRAMES, total)

    inferencer = MultiModalityDet3DInferencer(
        model=CONFIG_FILE,
        weights=CHECKPOINT,
        device="cuda:0"
    )

    # ===== BYPASSES =====
    inferencer._inputs_to_list = lambda inputs, **kwargs: inputs
    inferencer.postprocess = lambda preds, *args, **kwargs: {"predictions": preds, "visualization": None}

    print(f"Processando {total} frames...")
    for idx in range(total):
        sample = data_list[idx]
        ts = str(sample.get("timestamp", f"{idx:06d}"))

        # Ajustar caminhos
        if not os.path.isabs(sample["lidar_path"]):
            sample["lidar_path"] = os.path.join(DATA_ROOT, sample["lidar_path"])
        for cam in sample["images"]:
            if not os.path.isabs(sample["images"][cam]["img_path"]):
                sample["images"][cam]["img_path"] = os.path.join(DATA_ROOT, sample["images"][cam]["img_path"])

        # ===== CORREÇÃO: GARANTIR lidar2cam E cam2img EM TODAS AS CÂMERAS =====
        for cam_name, cam_info in sample["images"].items():
            if 'lidar2cam' not in cam_info:
                # Tenta obter de uma lista global se existir
                if 'lidar2cam' in sample and isinstance(sample['lidar2cam'], list):
                    cam_list = list(sample["images"].keys())
                    idx_cam = cam_list.index(cam_name)
                    if idx_cam < len(sample['lidar2cam']):
                        cam_info['lidar2cam'] = sample['lidar2cam'][idx_cam]
                    else:
                        cam_info['lidar2cam'] = np.eye(4, dtype=np.float32)
                else:
                    cam_info['lidar2cam'] = np.eye(4, dtype=np.float32)
            if 'cam2img' not in cam_info:
                if 'cam2img' in sample and isinstance(sample['cam2img'], list):
                    cam_list = list(sample["images"].keys())
                    idx_cam = cam_list.index(cam_name)
                    if idx_cam < len(sample['cam2img']):
                        cam_info['cam2img'] = sample['cam2img'][idx_cam]
                    else:
                        cam_info['cam2img'] = np.eye(3, dtype=np.float32)
                else:
                    cam_info['cam2img'] = np.eye(3, dtype=np.float32)

        sample["box_type_3d"] = LiDARInstance3DBoxes
        sample["box_mode_3d"] = "LIDAR"

        # Inferência
        try:
            result = inferencer([sample], return_vis=False, show=False, draw_pred=False, pred_score_thr=SCORE_THR)
            preds = result["predictions"][0]
            pred_inst = preds.pred_instances_3d
            boxes = pred_inst.bboxes_3d.tensor.cpu().numpy()
            scores = pred_inst.scores_3d.cpu().numpy()
            labels = pred_inst.labels_3d.cpu().numpy().astype(int)
        except Exception as e:
            print(f"  [{idx+1}/{total}] {ts} - inferência falhou ({e}), usando apenas GT")
            boxes = np.zeros((0,7), np.float32)
            scores = np.zeros(0, np.float32)
            labels = np.zeros(0, np.int32)

        # GT
        gt_boxes = np.zeros((0,7), np.float32)
        gt_labels = np.zeros(0, np.int32)
        if 'annos' in sample:
            if 'bboxes_3d' in sample['annos']:
                gt_boxes = sample['annos']['bboxes_3d']
                gt_labels = sample['annos']['labels_3d']

        # Dados para desenho
        # Pega lidar2cam da primeira câmera (todas devem ser iguais para projeção)
        first_cam = list(sample["images"].keys())[0]
        l2i = np.asarray(sample["images"][first_cam].get('lidar2cam', np.eye(4)), np.float32)
        aug = np.asarray(sample.get("img_aug_matrix", np.eye(4))[0] if isinstance(sample.get("img_aug_matrix"), list) else np.eye(4), np.float32)
        points = np.fromfile(sample['lidar_path'], dtype=np.float32).reshape(-1, 4)
        pc_range = [-54, -54, -5, 54, 54, 3]

        # 1. Câmera
        img_path = sample["images"][first_cam]["img_path"]
        front_img = cv2.imread(img_path)
        if front_img is None:
            print(f"Erro ao ler imagem: {img_path}")
            continue
        front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1,1, figsize=(10,4))
        draw_camera(ax, front_img, boxes, scores, labels, CLASSES, l2i, aug, SCORE_THR)
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ts}_camera.png"), bbox_inches="tight")
        plt.close(fig)

        # 2. BEV + Pred + GT
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        draw_bev(ax, points, boxes, scores, labels, CLASSES, pc_range, SCORE_THR,
                 gt_boxes=gt_boxes, gt_labels=gt_labels, title="BEV: Pred (sólido) + GT (tracejado)")
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ts}_det.png"), bbox_inches="tight")
        plt.close(fig)

        # 3. BEV apenas GT
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        draw_bev(ax, points, None, None, None, CLASSES, pc_range, SCORE_THR,
                 gt_boxes=gt_boxes, gt_labels=gt_labels, title="BEV: apenas GT")
        fig.savefig(os.path.join(OUTPUT_DIR, f"{ts}_det_gt.png"), bbox_inches="tight")
        plt.close(fig)

        print(f"  [{idx+1}/{total}] {ts} salvo")

    print(f"\nPNGs salvos em: {OUTPUT_DIR}")
    print("Para visualizar:  python tools/video_inference/view_frames.py --frames-dir " + OUTPUT_DIR)
    print("Para vídeo:       python tools/video_inference/make_eval_video.py --frames-dir " + OUTPUT_DIR + " --output ./video.mp4")

if __name__ == "__main__":
    main()