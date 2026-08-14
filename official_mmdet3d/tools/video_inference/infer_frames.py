#!/usr/bin/env python3
"""Per-frame inference engine: saves separate PNGs for camera, detection BEV and map BEV.

Produces three image types per frame in output/qualitative_vis/<tag>/frames/:
  {ts}_camera.png  — camera ZED-left with 3D boxes projected
  {ts}_det.png     — Bird's-Eye-View with point cloud + detection boxes + legend
  {ts}_map.png     — static map with BEV segmentation overlay + detection boxes

Usage:
    python tools/infer_frames.py \\
      --det-cfg  tools/cfgs/custom_models/bev_record_unitr.yaml \\
      --det-ckpt output/custom_models/bev_record_unitr/meu_exp/ckpt/checkpoint_epoch_80.pth \\
      --map-cfg  tools/cfgs/custom_models/bev_record_unitr_map.yaml \\
      --map-ckpt output/custom_models/bev_record_unitr_map/map_v1/ckpt/checkpoint_epoch_80.pth \\
      --split val --score-thresh 0.25 --seg-thresh 0.5 --tag eval_v1
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict
from matplotlib.patches import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
import _init_path  # noqa: F401

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import __all__ as DATASET_REGISTRY
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import box_utils

CLASS_COLORS = {
    "carga": "#ff8c42",
    "estrutura": "#c44dff",
    "humano": "#35c759",
    "maquina": "#ff5a5f",
    "navegavel": "#00bcd4",
    "obstrucao": "#8d6e63",
}

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


class _Logger:
    def info(self, msg, *a):
        print(msg % a if a else msg)


def parse_args():
    p = argparse.ArgumentParser(description="Per-frame inference → separate PNGs")
    p.add_argument("--det-cfg", default=str(REPO_ROOT / "tools/cfgs/custom_models/bev_record_unitr.yaml"))
    p.add_argument("--det-ckpt", default="")
    p.add_argument("--map-cfg", default=str(REPO_ROOT / "tools/cfgs/custom_models/bev_record_unitr_map.yaml"))
    p.add_argument("--map-ckpt", default="")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--score-thresh", type=float, default=0.25)
    p.add_argument("--seg-thresh", type=float, default=0.5)
    p.add_argument("--map-crop-m", type=float, default=30.0)
    p.add_argument("--tag", default="default")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--preview-frame", type=int, default=-1,
                   help="Se >= 0, renderiza apenas este índice e salva composite PNG para inspeção rápida")
    p.add_argument("--output-root", default=str(REPO_ROOT / "output/qualitative_vis"))
    p.add_argument("--dpi", type=int, default=120)
    # Ground truth overlay
    gt_grp = p.add_mutually_exclusive_group()
    gt_grp.add_argument("--show-gt", dest="show_gt", action="store_true", default=True,
                         help="Desenhar GT boxes nos painéis BEV (padrão: ligado)")
    gt_grp.add_argument("--hide-gt", dest="show_gt", action="store_false")
    # Temporal yaw smoothing
    sy_grp = p.add_mutually_exclusive_group()
    sy_grp.add_argument("--smooth-yaw", dest="smooth_yaw", action="store_true", default=True,
                         help="Suavizar orientação das boxes entre frames (padrão: ligado)")
    sy_grp.add_argument("--no-smooth-yaw", dest="smooth_yaw", action="store_false")
    p.add_argument("--smooth-yaw-alpha", type=float, default=0.35,
                   help="Peso EMA para suavização angular (0=estático, 1=sem suavização)")
    p.add_argument("--smooth-yaw-dist", type=float, default=3.0,
                   help="Distância máxima (m) para associar boxes entre frames")
    return p.parse_args()


def _load_cfg(path: str, split: str) -> EasyDict:
    cfg = EasyDict()
    cfg.ROOT_DIR = REPO_ROOT
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(path, cfg)
    cfg.DATA_CONFIG.DATA_SPLIT.test = split
    if split == "train":
        cfg.DATA_CONFIG.INFO_PATH.test = list(cfg.DATA_CONFIG.INFO_PATH.train)
    return cfg


def _build_dataset(cfg: EasyDict, logger):
    cls = DATASET_REGISTRY[cfg.DATA_CONFIG.DATASET]
    root = (cfg.ROOT_DIR / cfg.DATA_CONFIG.DATA_PATH).resolve()
    return cls(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
               training=False, root_path=root, logger=logger)


def _build_model(cfg: EasyDict, dataset, ckpt: str, logger):
    model = build_network(model_cfg=copy.deepcopy(cfg.MODEL),
                          num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda().eval()
    return model


def _unnormalize(t: torch.Tensor, cfg: EasyDict) -> np.ndarray:
    for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc.NAME == "image_normalize":
            mean = np.array(proc.mean, np.float32).reshape(3, 1, 1)
            std = np.array(proc.std, np.float32).reshape(3, 1, 1)
            img = t.detach().cpu().numpy().astype(np.float32) * std + mean
            return np.clip(np.transpose(img, (1, 2, 0)), 0, 1)
    return np.transpose(t.detach().cpu().numpy(), (1, 2, 0))


def _box_corners_xy(boxes: np.ndarray) -> np.ndarray:
    return box_utils.boxes_to_corners_3d(boxes[:, :7])[:, :4, :2]


def _project_corners(boxes: np.ndarray, l2i: np.ndarray, aug: np.ndarray):
    corners = box_utils.boxes_to_corners_3d(boxes[:, :7]).astype(np.float32)
    ch = np.concatenate([corners, np.ones((*corners.shape[:2], 1), np.float32)], -1)
    proj = ch @ l2i.T
    d = proj[..., 2]
    valid = d > 1e-4
    xy = np.zeros((*corners.shape[:2], 2), np.float32)
    xy[..., 0] = proj[..., 0] / np.clip(d, 1e-4, None)
    xy[..., 1] = proj[..., 1] / np.clip(d, 1e-4, None)
    xyh = np.concatenate([xy, np.ones((*xy.shape[:2], 2), np.float32)], -1)
    return (xyh @ aug.T)[..., :2], valid


def _load_static_map(cfg: EasyDict):
    for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc.NAME == "load_bev_segmentation_raster":
            map_dir = (REPO_ROOT / proc.map_dir).resolve()
            yaml_path = map_dir / "static_map.yaml"
            if not yaml_path.exists():
                return None, None, None, None
            meta = yaml.safe_load(yaml_path.read_text())
            from PIL import Image as PILImage
            img = np.asarray(PILImage.open((map_dir / meta["image"]).resolve()).convert("L"))
            map_rgb = np.stack([img, img, img], -1)
            origin = tuple(float(v) for v in meta.get("origin", [0, 0, 0])[:2])
            res = float(meta.get("resolution", 0.05))
            return map_rgb, origin, res, proc
    return None, None, None, None


def _save_camera(ax, image_rgb, boxes, scores, labels, class_names, l2i, aug, score_thresh):
    h, w = image_rgb.shape[:2]
    ax.imshow(image_rgb)
    ax.axis("off")
    ax.set_title("Câmera ZED-left + detecções projetadas")
    if boxes.size == 0:
        return
    c2d, vm = _project_corners(boxes, l2i, aug)
    for i in range(boxes.shape[0]):
        if scores[i] < score_thresh:
            continue
        color = CLASS_COLORS.get(class_names[int(labels[i]) - 1], "#ffcc00")
        drawn = False
        for s, e in BOX_EDGES:
            if not (vm[i, s] and vm[i, e]):
                continue
            p0, p1 = c2d[i, s], c2d[i, e]
            ok0 = 0 <= p0[0] < w and 0 <= p0[1] < h
            ok1 = 0 <= p1[0] < w and 0 <= p1[1] < h
            if not (ok0 or ok1):
                continue
            ax.plot([np.clip(p0[0], 0, w), np.clip(p1[0], 0, w)],
                    [np.clip(p0[1], 0, h), np.clip(p1[1], 0, h)],
                    color=color, linewidth=1.4)
            if not drawn and ok0:
                cls = class_names[int(labels[i]) - 1]
                ax.text(p0[0], p0[1], f"{cls} {scores[i]:.2f}", color="white", fontsize=7,
                        bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1.2})
                drawn = True


def _make_bev_seg_overlay(mask_arr, class_names, seg_thresh, alpha_val=180):
    """Converte (num_classes, H, W) → RGBA uint8 para imshow em coordenadas de metro.

    A orientação do raster é: row 0 = y_max, row N = y_min.
    Use np.flipud() no resultado antes de passar para imshow com origin='lower'.
    """
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
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        rgba[active, 0] = r
        rgba[active, 1] = g
        rgba[active, 2] = b
        rgba[active, 3] = alpha_val
    return rgba


def _save_det_bev(ax, points, boxes, scores, labels, class_names, pc_range, score_thresh,
                   gt_boxes=None, gt_labels=None,
                   pred_mask=None, seg_thresh=0.5, raster_cfg=None):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    xlim = (pc_range[0], pc_range[3])
    ylim = (pc_range[1], pc_range[4])

    # BEV segmentation overlay — abaixo do point cloud (zorder=0)
    if pred_mask is not None and raster_cfg is not None:
        rgba = _make_bev_seg_overlay(pred_mask, class_names, seg_thresh)
        extent = [raster_cfg.xbound[0], raster_cfg.xbound[1],
                  raster_cfg.ybound[0], raster_cfg.ybound[1]]
        ax.imshow(np.flipud(rgba), extent=extent, origin="lower",
                  alpha=0.55, zorder=0, aspect="auto", interpolation="nearest")

    mask = ((points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
            (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]))
    if mask.any():
        ax.scatter(points[mask, 0], points[mask, 1], s=0.3, c="dimgray", alpha=0.4, zorder=1)

    # GT boxes — dashed outline, sem texto de score
    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        gt_bottoms = _box_corners_xy(gt_boxes)
        for i in range(gt_boxes.shape[0]):
            cls_idx = int(gt_labels[i]) - 1 if gt_labels is not None else 0
            cls = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else "?"
            color = CLASS_COLORS.get(cls, "#aaaaaa")
            poly = Polygon(gt_bottoms[i], closed=True, fill=False,
                           edgecolor=color, linewidth=1.5, linestyle="--", alpha=0.65)
            ax.add_patch(poly)
            cx, cy, yaw = gt_boxes[i, 0], gt_boxes[i, 1], gt_boxes[i, 6]
            half_l = gt_boxes[i, 3] / 2
            ax.annotate("", xy=(cx + half_l * np.cos(yaw), cy + half_l * np.sin(yaw)),
                        xytext=(cx, cy),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9, "alpha": 0.6})

    # Predições — contorno sólido
    if boxes.size > 0:
        bottoms = _box_corners_xy(boxes)
        for i in range(boxes.shape[0]):
            if scores[i] < score_thresh:
                continue
            cls = class_names[int(labels[i]) - 1]
            color = CLASS_COLORS.get(cls, "#ffcc00")
            poly = Polygon(bottoms[i], closed=True, fill=False, edgecolor=color, linewidth=1.8)
            ax.add_patch(poly)
            cx, cy, yaw = boxes[i, 0], boxes[i, 1], boxes[i, 6]
            half_l = boxes[i, 3] / 2
            ax.annotate("", xy=(cx + half_l * np.cos(yaw), cy + half_l * np.sin(yaw)),
                        xytext=(cx, cy),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
            ax.text(cx, cy, f"{cls[:3]}\n{scores[i]:.2f}", color=color, fontsize=6,
                    ha="center", va="center")

    # legend
    handles = []
    for cls, col in CLASS_COLORS.items():
        handles.append(Patch(facecolor=col, edgecolor="none", label=cls))
    handles.append(Line2D([0], [0], color="white", linewidth=1.3, linestyle="--",
                           label="GT", alpha=0.75))
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Saída 2 — BEV · Pred (sólido) + GT (tracejado)")


def _save_map_bev(ax, map_rgb, pred_mask, boxes, scores, labels, class_names,
                  score_thresh, seg_thresh, raster_cfg,
                  ref_from_car, car_from_global, lidar_aug_matrix,
                  map_origin, map_res, map_crop_m,
                  gt_boxes=None, gt_labels=None, gt_class_names=None):
    overlay = map_rgb.copy().astype(np.float32)

    # ref_from_car is NOT used: boxes/pts are in X-forward ego frame.
    # Correct l2g = inv(car_from_global) @ inv(lidar_aug_matrix).
    try:
        l2g = np.linalg.inv(car_from_global) @ np.linalg.inv(lidar_aug_matrix)

        # Crop bounds computed first so inverse mapping covers only the visible region.
        e2g_mat = np.linalg.inv(car_from_global)
        cg = e2g_mat[:2, 3]
        h, w = map_rgb.shape[:2]
        cpx = int(round((cg[0] - map_origin[0]) / map_res))
        cpy = int(round(h - 1 - (cg[1] - map_origin[1]) / map_res))
        radius_px = max(50, int(map_crop_m / map_res))
        x0 = max(0, cpx - radius_px); x1 = min(w, cpx + radius_px)
        y0 = max(0, cpy - radius_px); y1 = min(h, cpy + radius_px)

        if pred_mask is not None and raster_cfg is not None:
            mask_arr = np.asarray(pred_mask)
            n_ch = mask_arr.shape[0] if mask_arr.ndim == 3 else 1
            bev_h = mask_arr.shape[1] if mask_arr.ndim == 3 else mask_arr.shape[0]
            bev_w = mask_arr.shape[2] if mask_arr.ndim == 3 else mask_arr.shape[1]

            # Inverse mapping: for every map pixel in the crop, find its BEV cell.
            # This fills solid regions instead of sparse center-point dots.
            g2l = np.linalg.inv(l2g)
            crop_h, crop_w = y1 - y0, x1 - x0
            gx = map_origin[0] + np.arange(x0, x1) * map_res          # global X per col
            gy = map_origin[1] + (h - 1 - np.arange(y0, y1)) * map_res  # global Y per row
            GX, GY = np.meshgrid(gx, gy)                               # (crop_h, crop_w)
            GP = np.stack([GX, GY, np.zeros_like(GX), np.ones_like(GX)], axis=-1)
            LP = (g2l @ GP.reshape(-1, 4).T).T.reshape(crop_h, crop_w, 4)
            bev_col = np.floor((LP[..., 0] - raster_cfg.xbound[0]) / raster_cfg.xbound[2]).astype(int)
            bev_row = np.floor((raster_cfg.ybound[1] - LP[..., 1]) / raster_cfg.ybound[2]).astype(int)
            in_bev = (bev_col >= 0) & (bev_col < bev_w) & (bev_row >= 0) & (bev_row < bev_h)

            for ci in range(min(n_ch, len(class_names))):
                ch = mask_arr[ci] if mask_arr.ndim == 3 else mask_arr
                active = np.zeros((crop_h, crop_w), dtype=bool)
                active[in_bev] = ch[bev_row[in_bev], bev_col[in_bev]] >= seg_thresh
                if not active.any():
                    continue
                hex_c = CLASS_COLORS.get(class_names[ci], "#888888")
                col_rgb = np.array([int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)],
                                   dtype=np.float32)
                oc = overlay[y0:y1, x0:x1]
                oc[active] = 0.45 * oc[active] + 0.55 * col_rgb
                overlay[y0:y1, x0:x1] = oc

        crop = overlay[y0:y1, x0:x1].astype(np.uint8)
        ax.imshow(crop)

        # draw detection boxes on map — use full 3D corners so l2g keeps correct shape
        if boxes.size > 0:
            all_corners_3d = box_utils.boxes_to_corners_3d(boxes[:, :7])  # (N,8,3)
            for i in range(boxes.shape[0]):
                if scores[i] < score_thresh:
                    continue
                bot = all_corners_3d[i, :4, :].astype(np.float32)  # (4,3) bottom with real z
                corners_h = np.concatenate([bot, np.ones((4, 1), np.float32)], -1)
                gc = (l2g @ corners_h.T).T
                bpx = (gc[:, 0] - map_origin[0]) / map_res - x0
                bpy = map_rgb.shape[0] - 1 - (gc[:, 1] - map_origin[1]) / map_res - y0
                pts = np.stack([bpx, bpy], -1)
                color = CLASS_COLORS.get(class_names[int(labels[i]) - 1], "#ffcc00")
                poly = Polygon(pts, closed=True, fill=False, edgecolor=color, linewidth=1.6)
                ax.add_patch(poly)

        # GT boxes on map (dashed) — same fix
        if gt_boxes is not None and gt_boxes.shape[0] > 0:
            gt_names = gt_class_names or class_names
            all_gt_corners_3d = box_utils.boxes_to_corners_3d(gt_boxes[:, :7])  # (N,8,3)
            for i in range(gt_boxes.shape[0]):
                bot = all_gt_corners_3d[i, :4, :].astype(np.float32)
                corners_h = np.concatenate([bot, np.ones((4, 1), np.float32)], -1)
                gc = (l2g @ corners_h.T).T
                bpx = (gc[:, 0] - map_origin[0]) / map_res - x0
                bpy = map_rgb.shape[0] - 1 - (gc[:, 1] - map_origin[1]) / map_res - y0
                pts = np.stack([bpx, bpy], -1)
                cls_idx = int(gt_labels[i]) - 1 if gt_labels is not None else 0
                cls = gt_names[cls_idx] if 0 <= cls_idx < len(gt_names) else "?"
                color = CLASS_COLORS.get(cls, "#aaaaaa")
                poly = Polygon(pts, closed=True, fill=False, edgecolor=color,
                               linewidth=1.2, linestyle="--", alpha=0.6)
                ax.add_patch(poly)

        ax.scatter([cpx - x0], [cpy - y0], s=50, c="yellow", marker="x", zorder=5)

        from matplotlib.patches import Patch as _Patch
        from matplotlib.lines import Line2D as _Line2D
        _handles = [_Patch(facecolor=col, edgecolor="none", label=cls)
                    for cls, col in CLASS_COLORS.items()]
        _handles.append(_Line2D([0], [0], color="white", lw=1.6, ls="-",  label="Pred box"))
        _handles.append(_Line2D([0], [0], color="white", lw=1.2, ls="--", label="GT box", alpha=0.75))
        ax.legend(handles=_handles, fontsize=6, loc="upper right", framealpha=0.7)
    except Exception:
        ax.imshow(map_rgb)

    ax.set_title("Saída 1 — Mapa + Detecções projetadas (sólido) · GT (tracejado)")
    ax.axis("off")


def _run_split(dataset, model, indices: List[int]) -> Dict[int, dict]:
    results = {}
    for idx in indices:
        sample = dataset[idx]
        batch = dataset.collate_batch([copy.deepcopy(sample)])
        load_data_to_gpu(batch)
        with torch.no_grad():
            preds, _ = model(batch)
        results[idx] = {"pred": preds[0], "sample": sample}
        torch.cuda.empty_cache()
    return results


def _run_split_map(dataset, model, indices: List[int]) -> Dict[int, dict]:
    results = {}
    for idx in indices:
        sample = dataset[idx]
        batch = dataset.collate_batch([copy.deepcopy(sample)])
        load_data_to_gpu(batch)
        with torch.no_grad():
            preds = model(batch)
        results[idx] = {"pred": preds[0], "sample": sample}
        torch.cuda.empty_cache()
    return results


def _save_det_gt_bev(ax, points, gt_boxes, gt_labels, class_names, pc_range,
                     gt_masks=None, seg_thresh=0.5, raster_cfg=None):
    """Painel BEV só com Ground-Truth (sólido, sem predições)."""
    from matplotlib.patches import Patch

    xlim = (pc_range[0], pc_range[3])
    ylim = (pc_range[1], pc_range[4])

    # GT segmentation overlay
    if gt_masks is not None and raster_cfg is not None:
        gt_arr = np.asarray(gt_masks)
        rgba = _make_bev_seg_overlay(gt_arr, class_names, 0.5, alpha_val=160)
        extent = [raster_cfg.xbound[0], raster_cfg.xbound[1],
                  raster_cfg.ybound[0], raster_cfg.ybound[1]]
        ax.imshow(np.flipud(rgba), extent=extent, origin="lower",
                  alpha=0.55, zorder=0, aspect="auto", interpolation="nearest")

    mask = ((points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1]) &
            (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]))
    if mask.any():
        ax.scatter(points[mask, 0], points[mask, 1], s=0.3, c="dimgray", alpha=0.4, zorder=1)

    if gt_boxes is not None and gt_boxes.shape[0] > 0:
        from matplotlib.patches import Polygon as MplPolygon
        bottoms = _box_corners_xy(gt_boxes)
        for i in range(gt_boxes.shape[0]):
            cls_idx = int(gt_labels[i]) - 1 if gt_labels is not None else 0
            cls = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else "?"
            color = CLASS_COLORS.get(cls, "#aaaaaa")
            poly = MplPolygon(bottoms[i], closed=True, fill=False,
                              edgecolor=color, linewidth=2.0)
            ax.add_patch(poly)
            cx, cy, yaw = gt_boxes[i, 0], gt_boxes[i, 1], gt_boxes[i, 6]
            half_l = gt_boxes[i, 3] / 2
            ax.annotate("", xy=(cx + half_l * np.cos(yaw), cy + half_l * np.sin(yaw)),
                        xytext=(cx, cy),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4})
            ax.text(cx, cy, cls[:3], color=color, fontsize=6, ha="center", va="center")

    handles = [Patch(facecolor=col, edgecolor="none", label=cls)
               for cls, col in CLASS_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", "box")
    ax.grid(alpha=0.15)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Saída 2 — BEV · Ground Truth")


def _save_map_gt_bev(ax, map_rgb, gt_masks_bev, gt_boxes, gt_labels, class_names,
                     raster_cfg, ref_from_car, car_from_global, lidar_aug_matrix,
                     map_origin, map_res, map_crop_m):
    """Painel mapa com segmentação GT projetada sobre fundo estático."""
    from matplotlib.patches import Polygon as MplPolygon
    overlay = map_rgb.copy().astype(np.float32)

    try:
        l2g = np.linalg.inv(car_from_global) @ np.linalg.inv(lidar_aug_matrix)
        # (ref_from_car is NOT used: boxes/pts are in X-forward frame, not physical sensor frame)

        # GT segmentation raster
        if gt_masks_bev is not None and raster_cfg is not None:
            gt_arr = np.asarray(gt_masks_bev)
            n_ch = gt_arr.shape[0] if gt_arr.ndim == 3 else 1
            for ci in range(min(n_ch, len(class_names))):
                ch = gt_arr[ci] if gt_arr.ndim == 3 else gt_arr
                rows, cols = np.where(ch > 0.5)
                if rows.size == 0:
                    continue
                hex_c = CLASS_COLORS.get(class_names[ci], "#888888")
                col_rgb = [int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)]
                x = raster_cfg.xbound[0] + (cols + 0.5) * raster_cfg.xbound[2]
                y = raster_cfg.ybound[1] - (rows + 0.5) * raster_cfg.ybound[2]
                lp = np.stack([x, y, np.zeros_like(x), np.ones_like(x)], -1)
                gp = (l2g @ lp.T).T
                px = np.round((gp[:, 0] - map_origin[0]) / map_res).astype(int)
                py = np.round(map_rgb.shape[0] - 1 - (gp[:, 1] - map_origin[1]) / map_res).astype(int)
                valid = (px >= 0) & (px < map_rgb.shape[1]) & (py >= 0) & (py < map_rgb.shape[0])
                overlay[py[valid], px[valid]] = col_rgb

        # crop around ego
        e2g_mat = np.linalg.inv(car_from_global)
        cg = e2g_mat[:2, 3]
        cpx = int(round((cg[0] - map_origin[0]) / map_res))
        cpy = int(round(map_rgb.shape[0] - 1 - (cg[1] - map_origin[1]) / map_res))
        radius_px = max(50, int(map_crop_m / map_res))
        h, w = map_rgb.shape[:2]
        x0 = max(0, cpx - radius_px); x1 = min(w, cpx + radius_px)
        y0 = max(0, cpy - radius_px); y1 = min(h, cpy + radius_px)
        crop = overlay[y0:y1, x0:x1].astype(np.uint8)
        ax.imshow(crop)

        # GT boxes on map — full 3D corners so l2g preserves correct box shape
        if gt_boxes is not None and gt_boxes.shape[0] > 0:
            all_gt_corners_3d = box_utils.boxes_to_corners_3d(gt_boxes[:, :7])  # (N,8,3)
            for i in range(gt_boxes.shape[0]):
                bot = all_gt_corners_3d[i, :4, :].astype(np.float32)  # (4,3)
                corners_h = np.concatenate([bot, np.ones((4, 1), np.float32)], -1)
                gc = (l2g @ corners_h.T).T
                bpx = (gc[:, 0] - map_origin[0]) / map_res - x0
                bpy = map_rgb.shape[0] - 1 - (gc[:, 1] - map_origin[1]) / map_res - y0
                pts = np.stack([bpx, bpy], -1)
                cls_idx = int(gt_labels[i]) - 1 if gt_labels is not None else 0
                cls = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else "?"
                color = CLASS_COLORS.get(cls, "#aaaaaa")
                poly = MplPolygon(pts, closed=True, fill=False,
                                  edgecolor=color, linewidth=2.0)
                ax.add_patch(poly)

        ax.scatter([cpx - x0], [cpy - y0], s=50, c="yellow", marker="x", zorder=5)

        from matplotlib.patches import Patch as _Patch
        _handles = [_Patch(facecolor=col, edgecolor="none", label=cls)
                    for cls, col in CLASS_COLORS.items()]
        ax.legend(handles=_handles, fontsize=6, loc="upper right", framealpha=0.7)
    except Exception:
        ax.imshow(map_rgb)

    ax.set_title("Saída 1 — Mapa + GT segmentação (verde) + GT boxes")
    ax.axis("off")


def _cdiff(a: float, b: float) -> float:
    """Diferença circular a − b em (−π, π]."""
    return float(((a - b + np.pi) % (2 * np.pi)) - np.pi)


def _smooth_yaw_temporal(det_results: Dict[int, dict], sorted_indices: List[int],
                          alpha: float = 0.35, max_dist_m: float = 3.0) -> None:
    """Suavização in-place do ângulo de orientação via rastreamento por proximidade.

    Problema: o modelo prediz yaw por frame de forma independente; para objetos
    simétricos (caixas, pallets) ele pode alternar entre θ e θ+π entre frames,
    causando 'flip' visual de 180°.

    Solução: para cada box no frame atual, encontra a box mais próxima no frame
    anterior (dentro de max_dist_m). Compara a distância angular de yaw com a
    do yaw+π. Se o yaw invertido é mais próximo do anterior, usa essa versão.
    Aplica EMA circular: yaw_suav = yaw_ant + alpha * diff_angular.
    """
    prev_boxes: Optional[np.ndarray] = None
    prev_yaws: Optional[np.ndarray] = None

    for idx in sorted_indices:
        if idx not in det_results:
            prev_boxes, prev_yaws = None, None
            continue

        p = det_results[idx]["pred"]
        curr = p["pred_boxes"].cpu().numpy().copy()

        if curr.shape[0] == 0:
            prev_boxes = curr.copy()
            prev_yaws = np.array([], dtype=np.float32)
            continue

        if prev_boxes is None or prev_boxes.shape[0] == 0:
            prev_boxes = curr.copy()
            prev_yaws = curr[:, 6].copy()
            continue

        curr_xy = curr[:, :2]           # (N_curr, 2)
        prev_xy = prev_boxes[:, :2]     # (N_prev, 2)
        # distâncias par-a-par (N_curr × N_prev)
        dists = np.linalg.norm(curr_xy[:, None, :] - prev_xy[None, :, :], axis=-1)

        new_yaws = curr[:, 6].copy()
        for ci in range(len(curr)):
            pi = int(np.argmin(dists[ci]))
            if dists[ci, pi] > max_dist_m:
                continue
            y_c = curr[ci, 6]
            y_p = prev_yaws[pi]
            diff_same = _cdiff(y_c, y_p)
            diff_flip = _cdiff(y_c + np.pi, y_p)
            # prefere a interpretação que exige menor rotação
            if abs(diff_flip) < abs(diff_same):
                new_yaws[ci] = y_p + alpha * diff_flip
            else:
                new_yaws[ci] = y_p + alpha * diff_same

        curr[:, 6] = new_yaws
        p["pred_boxes"] = torch.from_numpy(curr).to(p["pred_boxes"].device)
        prev_boxes = curr.copy()
        prev_yaws = new_yaws.copy()


def main():
    args = parse_args()
    logger = _Logger()

    out_dir = Path(args.output_root) / args.tag / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    has_det = bool(args.det_ckpt) and Path(args.det_ckpt).exists()
    has_map = bool(args.map_ckpt) and Path(args.map_ckpt).exists()

    if not has_det and not has_map:
        print("Forneça ao menos --det-ckpt ou --map-ckpt existente.")
        return

    # build dataset from map config (superset of det config)
    base_cfg_path = args.map_cfg if has_map else args.det_cfg
    cfg = _load_cfg(base_cfg_path, args.split)
    dataset = _build_dataset(cfg, logger)

    all_indices = list(range(len(dataset)))
    if args.max_frames > 0:
        all_indices = all_indices[:args.max_frames]
    if args.preview_frame >= 0:
        idx_pv = min(args.preview_frame, len(all_indices) - 1)
        all_indices = [all_indices[idx_pv]]
        print(f"Modo preview: apenas frame índice {all_indices[0]}")

    print(f"Split={args.split}  frames={len(all_indices)}  out={out_dir}")

    map_rgb, map_origin, map_res, raster_cfg = _load_static_map(cfg)

    # --- DETECTION PASS ---
    det_results: Dict[int, dict] = {}
    if has_det:
        print(f"\n[1/2] Detection model: {args.det_ckpt}")
        det_cfg = _load_cfg(args.det_cfg, args.split)
        det_model = _build_model(det_cfg, dataset, args.det_ckpt, logger)
        for idx in all_indices:
            sample = dataset[idx]
            batch = dataset.collate_batch([copy.deepcopy(sample)])
            load_data_to_gpu(batch)
            with torch.no_grad():
                preds, _ = det_model(batch)
            det_results[idx] = {"pred": preds[0], "sample": sample,
                                 "cfg": det_cfg}
            torch.cuda.empty_cache()
        del det_model
        torch.cuda.empty_cache()

    # Suavização temporal do yaw (resolve flip 180° entre frames)
    if args.smooth_yaw and det_results:
        print("Suavizando orientações das detecções entre frames...")
        _smooth_yaw_temporal(det_results, all_indices,
                              alpha=args.smooth_yaw_alpha,
                              max_dist_m=args.smooth_yaw_dist)

    # --- MAP PASS ---
    map_results: Dict[int, dict] = {}
    if has_map:
        print(f"\n[2/2] Map model: {args.map_ckpt}")
        map_cfg = _load_cfg(args.map_cfg, args.split)
        map_model = _build_model(map_cfg, dataset, args.map_ckpt, logger)
        for idx in all_indices:
            sample = dataset[idx]
            batch = dataset.collate_batch([copy.deepcopy(sample)])
            load_data_to_gpu(batch)
            with torch.no_grad():
                preds = map_model(batch)
            map_results[idx] = {"pred": preds[0], "sample": sample,
                                 "cfg": map_cfg}
            torch.cuda.empty_cache()
        del map_model
        torch.cuda.empty_cache()

    # --- RENDER ---
    print(f"\nRenderizando frames em {out_dir}")
    cam_order = list(cfg.DATA_CONFIG.CAMERA_CONFIG.CAMERA_ORDER)

    for idx in all_indices:
        # choose a result dict to pull sample from
        rd = det_results.get(idx) or map_results.get(idx)
        sample = rd["sample"]
        ts = sample["frame_id"] if "frame_id" in sample else f"{idx:06d}"
        print(f"  [{idx+1}/{len(all_indices)}] frame={ts}")

        # Ground Truth do frame (col 7 = label 1-indexed)
        gt_boxes_np = np.zeros((0, 7), np.float32)
        gt_labels_np = np.zeros(0, np.int32)
        if args.show_gt:
            raw_gt = sample.get("gt_boxes")
            if raw_gt is not None:
                raw_gt = np.asarray(raw_gt, np.float32)
                valid_gt = raw_gt[:, 3] > 0   # filtra padding (l > 0)
                raw_gt = raw_gt[valid_gt]
                if raw_gt.shape[0] > 0:
                    gt_boxes_np = raw_gt[:, :7]
                    gt_labels_np = (raw_gt[:, 7].astype(np.int32)
                                    if raw_gt.shape[1] > 7
                                    else np.ones(len(raw_gt), np.int32))

        # which result has det boxes?
        boxes = np.zeros((0, 7), np.float32)
        scores = np.zeros(0, np.float32)
        labels = np.zeros(0, np.int32)
        class_names_det = []
        if idx in det_results:
            p = det_results[idx]["pred"]
            det_cfg_ = det_results[idx]["cfg"]
            boxes = p["pred_boxes"].cpu().numpy()
            scores = p["pred_scores"].cpu().numpy()
            labels = p["pred_labels"].cpu().numpy()
            class_names_det = det_cfg_.CLASS_NAMES
        elif idx in map_results:
            # use map model det output if available
            p = map_results[idx]["pred"]
            map_cfg_ = map_results[idx]["cfg"]
            if "pred_boxes" in p:
                boxes = p["pred_boxes"].cpu().numpy()
                scores = p["pred_scores"].cpu().numpy()
                labels = p["pred_labels"].cpu().numpy()
                class_names_det = map_cfg_.CLASS_NAMES

        # CAMERA PNG
        l2i = np.asarray(sample["lidar2image"][0], np.float32)
        aug_m = np.asarray(sample["img_aug_matrix"][0], np.float32)
        d_cfg = (det_results[idx]["cfg"] if idx in det_results
                 else map_results[idx]["cfg"])
        img_rgb = _unnormalize(sample["camera_imgs"][0], d_cfg)

        fig_cam, ax_cam = plt.subplots(1, 1, figsize=(10, 4))
        _save_camera(ax_cam, img_rgb, boxes, scores, labels, class_names_det,
                     l2i, aug_m, args.score_thresh)
        fig_cam.tight_layout()
        fig_cam.savefig(out_dir / f"{ts}_camera.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_cam)

        # Extrai máscaras BEV antes dos renders (usadas em painel BEV e mapa)
        pred_mask = None
        gt_masks = sample.get("gt_masks_bev")
        if idx in map_results:
            mp = map_results[idx]["pred"]
            if "masks_bev" in mp:
                pred_mask = mp["masks_bev"].cpu().numpy()[0]

        # DET BEV PNG (seg overlay + pred sólido + GT tracejado)
        pc_range = list(d_cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        pts = np.asarray(sample["points"], np.float32)
        fig_det, ax_det = plt.subplots(1, 1, figsize=(7, 7))
        _save_det_bev(ax_det, pts, boxes, scores, labels, class_names_det,
                      pc_range, args.score_thresh,
                      gt_boxes=gt_boxes_np, gt_labels=gt_labels_np,
                      pred_mask=pred_mask, seg_thresh=args.seg_thresh, raster_cfg=raster_cfg)
        fig_det.tight_layout()
        fig_det.savefig(out_dir / f"{ts}_det.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_det)

        # DET GT PNG (gt seg overlay + GT boxes sólido)
        fig_dgt, ax_dgt = plt.subplots(1, 1, figsize=(7, 7))
        _save_det_gt_bev(ax_dgt, pts, gt_boxes_np, gt_labels_np,
                         class_names_det, pc_range,
                         gt_masks=gt_masks, seg_thresh=args.seg_thresh, raster_cfg=raster_cfg)
        fig_dgt.tight_layout()
        fig_dgt.savefig(out_dir / f"{ts}_det_gt.png", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig_dgt)

        # MAP PNG — predição sobre mapa estático
        if map_rgb is not None and "ref_from_car" in sample:
            ref_from_car = np.asarray(sample["ref_from_car"], np.float32)
            car_from_global = np.asarray(sample["car_from_global"], np.float32)
            lidar_aug = np.asarray(sample["lidar_aug_matrix"], np.float32)

            fig_map, ax_map = plt.subplots(1, 1, figsize=(7, 7))
            _save_map_bev(ax_map, map_rgb, pred_mask, boxes, scores, labels,
                          class_names_det, args.score_thresh, args.seg_thresh,
                          raster_cfg, ref_from_car, car_from_global, lidar_aug,
                          map_origin, map_res, args.map_crop_m,
                          gt_boxes=gt_boxes_np, gt_labels=gt_labels_np,
                          gt_class_names=class_names_det)
            fig_map.tight_layout()
            fig_map.savefig(out_dir / f"{ts}_map.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_map)

            fig_mgt, ax_mgt = plt.subplots(1, 1, figsize=(7, 7))
            _save_map_gt_bev(ax_mgt, map_rgb, gt_masks, gt_boxes_np, gt_labels_np,
                             class_names_det, raster_cfg, ref_from_car,
                             car_from_global, lidar_aug, map_origin, map_res,
                             args.map_crop_m)
            fig_mgt.tight_layout()
            fig_mgt.savefig(out_dir / f"{ts}_map_gt.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig_mgt)

    print(f"\nPronto. Frames salvos em: {out_dir}")

    if args.preview_frame >= 0:
        try:
            import cv2 as _cv2
            last_ts = ts  # noqa: F821 — defined in render loop above
            panel_order = ["camera", "det", "det_gt", "map", "map_gt"]
            strips = []
            for pn in panel_order:
                pp = out_dir / f"{last_ts}_{pn}.png"
                if pp.exists():
                    img = _cv2.imread(str(pp))
                    strips.append(_cv2.resize(img, (480, 480)))
            if strips:
                composite = np.hstack(strips)
                comp_path = out_dir.parent / "preview_composite.png"
                _cv2.imwrite(str(comp_path), composite)
                print(f"Preview composite: {comp_path}")
        except Exception as _e:
            print(f"Preview composite falhou: {_e}")

    print("Para gerar o vídeo:")
    print(f"  # Saída 1 — mapa com detecções projetadas:")
    print(f"  python tools/make_eval_video.py --frames-dir {out_dir} "
          f"--output {out_dir.parent}/video_saida1_mapa.mp4 --include camera map map_gt")
    print(f"  # Saída 2 — BEV overhead com caixas:")
    print(f"  python tools/make_eval_video.py --frames-dir {out_dir} "
          f"--output {out_dir.parent}/video_saida2_bev.mp4 --include camera det det_gt")


if __name__ == "__main__":
    main()
