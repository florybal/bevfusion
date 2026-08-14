#!/usr/bin/env python3
"""Visualização interativa de um frame: entradas do modelo + (opcionalmente) predições salvas.

Mostra 5 painéis com matplotlib:
  1. Câmera ZED-left
  2. Saída 2 — BEV nuvem de pontos + GT (tracejado) + predições (sólido)
  3. Saída 2 — BEV apenas GT
  4. Saída 1 — Mapa estático + predições projetadas
  5. Saída 1 — Mapa estático + GT projetado

Uso — apenas entradas do dataset (sem GPU, sem checkpoint):
    python tools/visualize_frame.py --frame-idx 0

Uso — com predições pré-geradas por infer_frames.py:
    python tools/visualize_frame.py --frame-idx 0 --frames-dir output/qualitative_vis/fix_v1/frames

Uso — com predições de checkpoints (requer GPU):
    python tools/visualize_frame.py --frame-idx 0 \\
      --det-ckpt output/.../checkpoint_epoch_60.pth \\
      --map-ckpt output/.../checkpoint_epoch_200.pth

Dica: use --save para salvar PNG em vez de abrir janela interativa.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import _init_path  # noqa: F401

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import __all__ as DATASET_REGISTRY
from pcdet.utils import box_utils
from easydict import EasyDict

CLASS_COLORS = {
    "carga": "#ff8c42",
    "estrutura": "#c44dff",
    "humano": "#35c759",
    "maquina": "#ff5a5f",
    "navegavel": "#00bcd4",
    "obstrucao": "#8d6e63",
}


def parse_args():
    p = argparse.ArgumentParser(description="Visualização qualitativa interativa")
    p.add_argument("--det-cfg",  default=str(REPO_ROOT / "tools/cfgs/custom_models/bev_record_unitr_v2.yaml"))
    p.add_argument("--map-cfg",  default=str(REPO_ROOT / "tools/cfgs/custom_models/bev_record_unitr_map_v2.yaml"))
    p.add_argument("--det-ckpt", default="", help="Checkpoint detecção (opcional; requer GPU)")
    p.add_argument("--map-ckpt", default="", help="Checkpoint mapa (opcional; requer GPU)")
    p.add_argument("--split",    default="val", choices=["train", "val", "test"])
    p.add_argument("--frame-idx",type=int, default=0, help="Índice do frame no dataset")
    p.add_argument("--frames-dir", default="", help="Dir com PNGs pré-gerados por infer_frames.py")
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--seg-thresh",   type=float, default=0.5)
    p.add_argument("--map-crop-m",   type=float, default=30.0)
    p.add_argument("--save",    default="", help="Caminho PNG para salvar ao invés de exibir")
    p.add_argument("--dpi",     type=int,   default=120)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de visualização (copiados de infer_frames.py para autonomia)
# ──────────────────────────────────────────────────────────────────────────────

BOX_EDGES = (
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
)


def _box_corners_xy(boxes):
    return box_utils.boxes_to_corners_3d(boxes[:,:7])[:,:4,:2]


def _unnormalize(t, cfg):
    import torch
    for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc.NAME == "image_normalize":
            mean = np.array(proc.mean, np.float32).reshape(3,1,1)
            std  = np.array(proc.std,  np.float32).reshape(3,1,1)
            img  = t.detach().cpu().numpy().astype(np.float32) * std + mean
            return np.clip(np.transpose(img,(1,2,0)), 0, 1)
    return np.transpose(t.detach().cpu().numpy(), (1,2,0))


def _project_corners(boxes, l2i, aug):
    corners = box_utils.boxes_to_corners_3d(boxes[:,:7]).astype(np.float32)
    ch = np.concatenate([corners, np.ones((*corners.shape[:2],1),np.float32)],-1)
    proj = ch @ l2i.T
    d  = proj[...,2]
    xy = np.zeros((*corners.shape[:2],2),np.float32)
    xy[...,0] = proj[...,0] / np.clip(d,1e-4,None)
    xy[...,1] = proj[...,1] / np.clip(d,1e-4,None)
    xyh = np.concatenate([xy, np.ones((*xy.shape[:2],2),np.float32)],-1)
    return (xyh @ aug.T)[...,:2], d > 1e-4


def _load_static_map(cfg):
    for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc.NAME == "load_bev_segmentation_raster":
            map_dir = (REPO_ROOT / proc.map_dir).resolve()
            yaml_path = map_dir / "static_map.yaml"
            if not yaml_path.exists():
                return None, None, None, None
            meta = yaml.safe_load(yaml_path.read_text())
            from PIL import Image as PILImage
            img = np.asarray(PILImage.open((map_dir / meta["image"]).resolve()).convert("L"))
            map_rgb = np.stack([img,img,img],-1)
            origin = tuple(float(v) for v in meta.get("origin",[0,0,0])[:2])
            res    = float(meta.get("resolution", 0.05))
            return map_rgb, origin, res, proc
    return None, None, None, None


def _make_bev_seg_overlay(mask_arr, class_names, seg_thresh, alpha_val=180):
    """Converte (num_classes, H, W) → RGBA uint8 para imshow em coordenadas de metro."""
    arr = np.asarray(mask_arr)
    H = arr.shape[1] if arr.ndim == 3 else arr.shape[0]
    W = arr.shape[2] if arr.ndim == 3 else arr.shape[1]
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    n_ch = arr.shape[0] if arr.ndim == 3 else 1
    for ci in range(min(n_ch, len(class_names))):
        ch = arr[ci] if arr.ndim == 3 else arr
        active = ch >= seg_thresh
        if not active.any():
            continue
        hex_c = CLASS_COLORS.get(class_names[ci], "#888888")
        r, g, b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
        rgba[active, 0] = r
        rgba[active, 1] = g
        rgba[active, 2] = b
        rgba[active, 3] = alpha_val
    return rgba


def _draw_bev(ax, points, pred_boxes, pred_scores, pred_labels,
              gt_boxes, gt_labels, class_names, pc_range, score_thresh,
              pred_mask=None, gt_masks=None, seg_thresh=0.5, raster_cfg=None,
              title="BEV"):
    from matplotlib.patches import Polygon, Patch
    from matplotlib.lines import Line2D

    xlim = (pc_range[0], pc_range[3])
    ylim = (pc_range[1], pc_range[4])

    # BEV seg overlay (abaixo do ponto cloud)
    def _seg_imshow(mask_arr, a_val):
        if mask_arr is None or raster_cfg is None:
            return
        rgba = _make_bev_seg_overlay(mask_arr, class_names, seg_thresh, a_val)
        extent = [raster_cfg.xbound[0], raster_cfg.xbound[1],
                  raster_cfg.ybound[0], raster_cfg.ybound[1]]
        ax.imshow(np.flipud(rgba), extent=extent, origin="lower",
                  alpha=0.55, zorder=0, aspect="auto", interpolation="nearest")

    _seg_imshow(pred_mask, 180)
    _seg_imshow(gt_masks,  140)

    mask = ((points[:,0]>=xlim[0])&(points[:,0]<=xlim[1])&
            (points[:,1]>=ylim[0])&(points[:,1]<=ylim[1]))
    if mask.any():
        ax.scatter(points[mask,0], points[mask,1], s=0.3, c="dimgray", alpha=0.4, zorder=1)

    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        bottoms = _box_corners_xy(gt_boxes)
        for i in range(gt_boxes.shape[0]):
            cls = class_names[int(gt_labels[i])-1] if gt_labels is not None else "?"
            col = CLASS_COLORS.get(cls,"#aaaaaa")
            ax.add_patch(Polygon(bottoms[i], closed=True, fill=False,
                                 edgecolor=col, lw=1.5, ls="--", alpha=0.7))
            cx, cy, yaw = gt_boxes[i,0], gt_boxes[i,1], gt_boxes[i,6]
            ax.annotate("", xy=(cx+gt_boxes[i,3]/2*np.cos(yaw), cy+gt_boxes[i,3]/2*np.sin(yaw)),
                        xytext=(cx,cy), arrowprops={"arrowstyle":"->","color":col,"lw":0.9,"alpha":0.6})

    if pred_boxes is not None and pred_boxes.size > 0:
        bottoms = _box_corners_xy(pred_boxes)
        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < score_thresh: continue
            cls = class_names[int(pred_labels[i])-1]
            col = CLASS_COLORS.get(cls,"#ffcc00")
            ax.add_patch(Polygon(bottoms[i], closed=True, fill=False, edgecolor=col, lw=1.8))
            cx, cy, yaw = pred_boxes[i,0], pred_boxes[i,1], pred_boxes[i,6]
            ax.annotate("", xy=(cx+pred_boxes[i,3]/2*np.cos(yaw), cy+pred_boxes[i,3]/2*np.sin(yaw)),
                        xytext=(cx,cy), arrowprops={"arrowstyle":"->","color":col,"lw":1.2})
            ax.text(cx, cy, f"{cls[:3]}\n{pred_scores[i]:.2f}", color=col, fontsize=6,
                    ha="center", va="center")

    handles = [Patch(facecolor=c, edgecolor="none", label=n) for n,c in CLASS_COLORS.items()]
    handles.append(Line2D([0],[0], color="white", lw=1.3, ls="--", label="GT", alpha=0.75))
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal","box")
    ax.grid(alpha=0.15); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_title(title)


def _draw_map(ax, map_rgb, pred_mask, pred_boxes, pred_scores, pred_labels,
              gt_masks, gt_boxes, gt_labels, class_names,
              raster_cfg, car_from_global, lidar_aug_matrix,
              map_origin, map_res, map_crop_m, seg_thresh, score_thresh, title):
    from matplotlib.patches import Polygon, Patch
    from matplotlib.lines import Line2D

    overlay = map_rgb.copy().astype(np.float32)
    try:
        # correct l2g: stored pts/boxes are in X-forward car frame
        l2g = np.linalg.inv(car_from_global) @ np.linalg.inv(lidar_aug_matrix)

        # segmentation overlay — each BEV channel → its class color
        def _proj_seg_multiclass(mask_arr, thresh):
            if mask_arr is None or raster_cfg is None:
                return
            arr = np.asarray(mask_arr)
            n_ch = arr.shape[0] if arr.ndim == 3 else 1
            for ci in range(min(n_ch, len(class_names))):
                ch = arr[ci] if arr.ndim == 3 else arr
                rows, cols = np.where(ch >= thresh)
                if rows.size == 0:
                    continue
                hex_c = CLASS_COLORS.get(class_names[ci], "#888888")
                col_rgb = [int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)]
                x = raster_cfg.xbound[0] + (cols+0.5)*raster_cfg.xbound[2]
                y = raster_cfg.ybound[1] - (rows+0.5)*raster_cfg.ybound[2]
                lp = np.stack([x,y,np.zeros_like(x),np.ones_like(x)],-1)
                gp = (l2g @ lp.T).T
                px = np.round((gp[:,0]-map_origin[0])/map_res).astype(int)
                py = np.round(map_rgb.shape[0]-1-(gp[:,1]-map_origin[1])/map_res).astype(int)
                vld = (px>=0)&(px<map_rgb.shape[1])&(py>=0)&(py<map_rgb.shape[0])
                overlay[py[vld],px[vld]] = col_rgb

        _proj_seg_multiclass(pred_mask, seg_thresh)
        _proj_seg_multiclass(gt_masks,  0.5)

        # crop around ego
        ego_g = l2g @ np.array([0,0,0,1])
        cpx = int(round((ego_g[0]-map_origin[0])/map_res))
        cpy = int(round(map_rgb.shape[0]-1-(ego_g[1]-map_origin[1])/map_res))
        r   = max(50, int(map_crop_m/map_res))
        h,w = map_rgb.shape[:2]
        x0 = max(0,cpx-r); x1 = min(w,cpx+r)
        y0 = max(0,cpy-r); y1 = min(h,cpy+r)
        ax.imshow(overlay[y0:y1,x0:x1].astype(np.uint8))

        def _draw_boxes_on_map(boxes, scores, labels, solid, lw, alpha=1.0):
            if boxes is None or boxes.size==0: return
            all_c = box_utils.boxes_to_corners_3d(boxes[:,:7])  # (N,8,3)
            for i in range(boxes.shape[0]):
                if scores is not None and scores[i] < score_thresh: continue
                bot = all_c[i,:4,:].astype(np.float32)
                ch  = np.concatenate([bot,np.ones((4,1),np.float32)],-1)
                gc  = (l2g @ ch.T).T
                bpx = (gc[:,0]-map_origin[0])/map_res - x0
                bpy = map_rgb.shape[0]-1-(gc[:,1]-map_origin[1])/map_res - y0
                pts = np.stack([bpx,bpy],-1)
                cls = class_names[int(labels[i])-1] if labels is not None else "?"
                col = CLASS_COLORS.get(cls,"#ffcc00")
                ls  = "-" if solid else "--"
                ax.add_patch(Polygon(pts, closed=True, fill=False,
                                     edgecolor=col, lw=lw, ls=ls, alpha=alpha))

        _draw_boxes_on_map(pred_boxes, pred_scores, pred_labels, solid=True, lw=1.6)
        _draw_boxes_on_map(gt_boxes,   None,        gt_labels,   solid=False, lw=1.2, alpha=0.7)

        ax.scatter([cpx-x0],[cpy-y0], s=50, c="yellow", marker="x", zorder=5)

        handles = [Patch(facecolor=c, edgecolor="none", label=n) for n,c in CLASS_COLORS.items()]
        handles.append(Line2D([0],[0], color="white", lw=1.6, ls="-",  label="Pred box"))
        handles.append(Line2D([0],[0], color="white", lw=1.2, ls="--", label="GT box", alpha=0.75))
        ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.7)

    except Exception as e:
        ax.imshow(map_rgb); ax.text(5,5,str(e),color="red",fontsize=6)

    ax.set_title(title); ax.axis("off")


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # matplotlib backend: interactive if no --save
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── load dataset ──
    class _L:
        def info(self,m,*a): print(m%a if a else m)
    logger = _L()

    base_cfg_path = args.map_cfg if Path(args.map_cfg).exists() else args.det_cfg
    cfg = EasyDict(); cfg.ROOT_DIR = REPO_ROOT; cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(base_cfg_path, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT.test = args.split
    if args.split == "train":
        cfg.DATA_CONFIG.INFO_PATH.test = list(cfg.DATA_CONFIG.INFO_PATH.train)
    cls = DATASET_REGISTRY[cfg.DATA_CONFIG.DATASET]
    root = (REPO_ROOT / cfg.DATA_CONFIG.DATA_PATH).resolve()
    ds = cls(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
             training=False, root_path=root, logger=logger)

    n = len(ds)
    idx = min(max(0, args.frame_idx), n-1)
    print(f"Dataset: {n} frames  →  mostrando índice {idx}")
    sample = ds[idx]
    ts = sample.get("frame_id", f"{idx:06d}")
    class_names = cfg.CLASS_NAMES
    pc_range    = list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

    # ── GT ──
    pts        = np.asarray(sample["points"],  np.float32)
    gt_raw     = np.asarray(sample.get("gt_boxes", np.zeros((0,8))), np.float32)
    valid_gt   = gt_raw[:,3] > 0
    gt_raw     = gt_raw[valid_gt]
    gt_boxes   = gt_raw[:,:7]  if gt_raw.shape[0] > 0 else np.zeros((0,7),np.float32)
    gt_labels  = (gt_raw[:,7].astype(np.int32)
                  if gt_raw.shape[0] > 0 and gt_raw.shape[1] > 7
                  else np.ones(len(gt_raw),np.int32))
    gt_masks   = sample.get("gt_masks_bev")

    # ── prediction: try frames-dir first, then GPU ──
    pred_boxes  = np.zeros((0,7), np.float32)
    pred_scores = np.zeros(0,  np.float32)
    pred_labels = np.zeros(0,  np.int32)
    pred_mask   = None
    img_rgb     = _unnormalize(sample["camera_imgs"][0], cfg)

    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    if frames_dir and frames_dir.exists():
        import cv2
        print(f"Usando PNGs pré-gerados de {frames_dir}")
        # just show the saved panels side by side
        panel_names = ["camera","det","det_gt","map","map_gt"]
        panels = []
        for pn in panel_names:
            pp = frames_dir / f"{ts}_{pn}.png"
            if pp.exists():
                img = cv2.cvtColor(cv2.imread(str(pp)), cv2.COLOR_BGR2RGB)
                panels.append((pn, img))
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels), 5))
            if len(panels) == 1:
                axes = [axes]
            for ax, (name, img) in zip(axes, panels):
                ax.imshow(img); ax.axis("off"); ax.set_title(name)
            fig.suptitle(f"Frame {ts}  (PNGs pré-gerados)", fontsize=10)
            plt.tight_layout()
            if args.save:
                plt.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
                print(f"Salvo: {args.save}")
            else:
                plt.show()
            return

    has_det = bool(args.det_ckpt) and Path(args.det_ckpt).exists()
    has_map = bool(args.map_ckpt) and Path(args.map_ckpt).exists()
    if has_det or has_map:
        import torch
        from pcdet.models import build_network, load_data_to_gpu
        batch = ds.collate_batch([copy.deepcopy(sample)])
        load_data_to_gpu(batch)
        if has_det:
            det_cfg = EasyDict(); det_cfg.ROOT_DIR = REPO_ROOT; det_cfg.LOCAL_RANK = 0
            cfg_from_yaml_file(args.det_cfg, det_cfg)
            det_cfg.DATA_CONFIG.DATA_SPLIT.test = args.split
            det_model = build_network(model_cfg=copy.deepcopy(det_cfg.MODEL),
                                      num_class=len(det_cfg.CLASS_NAMES), dataset=ds)
            det_model.load_params_from_file(filename=args.det_ckpt, logger=logger, to_cpu=False)
            det_model.cuda().eval()
            with torch.no_grad():
                preds, _ = det_model(batch)
            p = preds[0]
            pred_boxes  = p["pred_boxes"].cpu().numpy()
            pred_scores = p["pred_scores"].cpu().numpy()
            pred_labels = p["pred_labels"].cpu().numpy()
            class_names = det_cfg.CLASS_NAMES
            del det_model; torch.cuda.empty_cache()
        if has_map:
            map_cfg = EasyDict(); map_cfg.ROOT_DIR = REPO_ROOT; map_cfg.LOCAL_RANK = 0
            cfg_from_yaml_file(args.map_cfg, map_cfg)
            map_cfg.DATA_CONFIG.DATA_SPLIT.test = args.split
            map_model = build_network(model_cfg=copy.deepcopy(map_cfg.MODEL),
                                      num_class=len(map_cfg.CLASS_NAMES), dataset=ds)
            map_model.load_params_from_file(filename=args.map_ckpt, logger=logger, to_cpu=False)
            map_model.cuda().eval()
            with torch.no_grad():
                map_preds = map_model(batch)
            pred_mask = map_preds[0]["masks_bev"].cpu().numpy()[0]
            del map_model; torch.cuda.empty_cache()

    # ── layout: 5 panels ──
    map_rgb, map_origin, map_res, raster_cfg = _load_static_map(cfg)
    ref_from_car    = np.asarray(sample.get("ref_from_car",  np.eye(4)), np.float32)
    car_from_global = np.asarray(sample.get("car_from_global", np.eye(4)), np.float32)
    lidar_aug       = np.asarray(sample.get("lidar_aug_matrix", np.eye(4)), np.float32)
    l2i = np.asarray(sample["lidar2image"][0],  np.float32)
    aug = np.asarray(sample["img_aug_matrix"][0], np.float32)

    fig = plt.figure(figsize=(25, 10))
    fig.suptitle(f"Frame {ts} | idx={idx} | split={args.split}", fontsize=11)

    # 1. Camera
    ax1 = fig.add_subplot(2,3,1)
    h_img, w_img = img_rgb.shape[:2]
    ax1.imshow(img_rgb); ax1.axis("off"); ax1.set_title("Câmera ZED-left")
    if pred_boxes.size > 0:
        from matplotlib.patches import Polygon
        c2d, vm = _project_corners(pred_boxes, l2i, aug)
        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < args.score_thresh: continue
            col = CLASS_COLORS.get(class_names[int(pred_labels[i])-1],"#ffcc00")
            for s,e in BOX_EDGES:
                if not (vm[i,s] and vm[i,e]): continue
                p0,p1 = c2d[i,s], c2d[i,e]
                if not (0<=p0[0]<w_img and 0<=p0[1]<h_img or 0<=p1[0]<w_img and 0<=p1[1]<h_img): continue
                ax1.plot([np.clip(p0[0],0,w_img), np.clip(p1[0],0,w_img)],
                         [np.clip(p0[1],0,h_img), np.clip(p1[1],0,h_img)],
                         color=col, lw=1.4)

    # 2. BEV Saída 2 — pred seg overlay + pred boxes + GT boxes
    ax2 = fig.add_subplot(2,3,2)
    _draw_bev(ax2, pts, pred_boxes, pred_scores, pred_labels,
              gt_boxes, gt_labels, class_names, pc_range, args.score_thresh,
              pred_mask=pred_mask, seg_thresh=args.seg_thresh, raster_cfg=raster_cfg,
              title="Saída 2 — BEV: Pred seg + boxes (sólido) + GT (tracejado)")

    # 3. BEV Saída 2 — GT seg overlay + GT boxes
    ax3 = fig.add_subplot(2,3,3)
    _draw_bev(ax3, pts, None, None, None,
              gt_boxes, gt_labels, class_names, pc_range, args.score_thresh,
              gt_masks=gt_masks, seg_thresh=args.seg_thresh, raster_cfg=raster_cfg,
              title="Saída 2 — BEV: GT segmentação + boxes")

    # 4. Saída 1 — Map + pred
    ax4 = fig.add_subplot(2,3,4)
    if map_rgb is not None and "car_from_global" in sample:
        _draw_map(ax4, map_rgb, pred_mask, pred_boxes, pred_scores, pred_labels,
                  None, gt_boxes, gt_labels, class_names,
                  raster_cfg, car_from_global, lidar_aug,
                  map_origin, map_res, args.map_crop_m,
                  args.seg_thresh, args.score_thresh,
                  title="Saída 1 — Mapa + Pred projetada (sólido) + GT (tracejado)")
    else:
        ax4.text(0.5,0.5,"mapa não disponível", ha="center", transform=ax4.transAxes)
        ax4.axis("off")

    # 5. Saída 1 — Map + GT
    ax5 = fig.add_subplot(2,3,5)
    if map_rgb is not None and "car_from_global" in sample:
        _draw_map(ax5, map_rgb, None, None, None, None,
                  gt_masks, gt_boxes, gt_labels, class_names,
                  raster_cfg, car_from_global, lidar_aug,
                  map_origin, map_res, args.map_crop_m,
                  args.seg_thresh, args.score_thresh,
                  title="Saída 1 — Mapa + GT segmentação (verde) + GT boxes")
    else:
        ax5.axis("off")

    # 6. Info panel
    ax6 = fig.add_subplot(2,3,6)
    info_lines = [
        f"frame_id: {ts}",
        f"idx: {idx}",
        f"split: {args.split}",
        f"pontos: {pts.shape[0]}",
        f"GT boxes: {gt_boxes.shape[0]}",
        f"pred boxes: {pred_boxes.shape[0]}",
        "",
        "GT classes:",
    ]
    for i in range(gt_boxes.shape[0]):
        cls = class_names[int(gt_labels[i])-1] if len(gt_labels)>i else "?"
        info_lines.append(f"  [{i}] {cls} cx={gt_boxes[i,0]:.1f} cy={gt_boxes[i,1]:.1f}")
    ax6.text(0.05, 0.95, "\n".join(info_lines), transform=ax6.transAxes,
             fontsize=8, va="top", fontfamily="monospace")
    ax6.axis("off"); ax6.set_title("Informações do frame")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Salvo em: {args.save}")
    else:
        print("Abrindo janela interativa... (feche para sair)")
        plt.show()


if __name__ == "__main__":
    main()
