import os
import numpy as np

# --- Monkey Patch ---
np.Inf = np.inf 
# ---------------------------

import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Patch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.dataset import pseudo_collate
from mmdet3d.registry import MODELS, DATASETS
from mmdet3d.utils import register_all_modules

# ===== CONFIGURAÇÕES =====
CONFIG_FILE = "projects/BEVFusion/configs/bevfusion_plastipak.py"
CHECKPOINT = "/workspace/results/training/plastipak/30epoch/epoch_20.pth"  
DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
OUTPUT_IMAGE = "./prediction_visualization_final.png"
SCORE_THR = 0.01
SAMPLE_INDEX = 0

CLASSES = (
    "obstrucao", "empilhadeira", "carga", "maquina",
    "humano", "navegavel", "estrutura", "portapalete"
)
COLOR_MAP = {
    "obstrucao": "red", "empilhadeira": "orange", "carga": "magenta",
    "maquina": "blue", "humano": "cyan", "navegavel": "green",
    "estrutura": "yellow", "portapalete": "brown"
}

register_all_modules(init_default_scope='mmdet3d')

# 1. Carregar Configuração
cfg = Config.fromfile(CONFIG_FILE)
cfg.data_root = DATA_ROOT

# Garantir que a chave 'img' esteja no Pack3DDetInputs do test_pipeline
for transform in cfg.test_pipeline:
    if transform.get('type') == 'Pack3DDetInputs':
        if 'keys' in transform and 'img' not in transform['keys']:
            transform['keys'] = list(transform['keys']) + ['img']

# Reconstruir pipeline no dataset
dataset_cfg = cfg.test_dataloader.dataset.copy()
dataset_cfg['pipeline'] = cfg.test_pipeline
dataset = DATASETS.build(dataset_cfg)

# 2. Carregar Amostra e Criar Batch
sample = dataset[SAMPLE_INDEX]
batch = pseudo_collate([sample])

# 3. Construir e Carregar Modelo
model = MODELS.build(cfg.model)
load_checkpoint(model, CHECKPOINT, map_location='cpu')
model.cuda()
model.eval()

# 4. Inferência Oficial (Passando pelo DataPreprocessor)
with torch.no_grad():
    # O test_step é vital! Ele chama o model.data_preprocessor internamente,
    # move para GPU, aplica padding, agrupa tensores e converte 'img' para 'imgs'.
    predictions = model.test_step(batch)
    preds = predictions[0]

# 5. Extrair Predições
pred_instances = preds.pred_instances_3d
boxes = pred_instances.bboxes_3d.tensor.cpu().numpy()
labels = pred_instances.labels_3d.cpu().numpy()
scores = pred_instances.scores_3d.cpu().numpy()

print("\n" + "=" * 70)
print("DIAGNÓSTICO DAS PREDIÇÕES")
print("=" * 70)
 
print("Número de predições:", len(scores))
print(model.bbox_head.test_cfg) 

if len(scores) > 0:
 
    print("\nScores:")
    print("  min :", scores.min())
    print("  max :", scores.max())
    print("  mean:", scores.mean())
 
    print("\nQuantidade por threshold:")
    for thr in [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
        print(
            f"  score >= {thr:.3f}: "
            f"{np.sum(scores >= thr)}"
        )
 
    print("\nDistribuição das classes:")
    for c in range(len(CLASSES)):
        n = np.sum(labels == c)
        if n > 0:
            print(
                f"  {c}: {CLASSES[c]:15s} "
                f"{n:4d} preds"
            )
 
    print("\nPrimeiras caixas:")
    for i in range(min(20, len(boxes))):
        print(
            f"[{i:02d}] "
            f"class={labels[i]} ({CLASSES[labels[i]]}) "
            f"score={scores[i]:.4f} "
            f"box={boxes[i]}"
        )
 
    print("\nEstatísticas XYZ:")
 
    print("X:")
    print("  min =", boxes[:, 0].min())
    print("  max =", boxes[:, 0].max())
 
    print("Y:")
    print("  min =", boxes[:, 1].min())
    print("  max =", boxes[:, 1].max())
 
    print("Z:")
    print("  min =", boxes[:, 2].min())
    print("  max =", boxes[:, 2].max())
 
    print("\nDimensões:")
 
    print("W:")
    print("  min =", boxes[:, 3].min())
    print("  max =", boxes[:, 3].max())
 
    print("L:")
    print("  min =", boxes[:, 4].min())
    print("  max =", boxes[:, 4].max())
 
    print("H:")
    print("  min =", boxes[:, 5].min())
    print("  max =", boxes[:, 5].max())
 
else:
    print("!!! MODELO NÃO PRODUZIU NENHUMA PREDIÇÃO !!!")
 
print("=" * 70 + "\n")

# 6. Imagem para Exibição
inputs = batch['inputs']
data_samples = batch['data_samples']

img_path = None
if hasattr(data_samples[0], 'img_path'):
    img_path = data_samples[0].img_path
    if isinstance(img_path, list):
        img_path = img_path[0]

if img_path and os.path.exists(img_path):
    front_img = cv2.imread(img_path)
elif 'img' in inputs and len(inputs['img']) > 0:
    raw_img = inputs['img'][0]
    if raw_img.ndim == 4:  # (N_views, C, H, W)
        raw_img = raw_img[0]
    raw_img = raw_img.cpu().numpy().transpose(1, 2, 0)
    if raw_img.max() <= 1.0:
        raw_img = (raw_img * 255).astype(np.uint8)
    front_img = cv2.cvtColor(raw_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
else:
    front_img = np.zeros((720, 1280, 3), dtype=np.uint8)

# 7. Renderização do Gráfico
fig, (ax_front, ax_bev) = plt.subplots(1, 2, figsize=(24, 10))
ax_front.imshow(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
ax_front.set_title("Camera (front)", fontsize=14)
ax_front.axis("off")

BEV_X_RANGE = [-54, 54]
BEV_Y_RANGE = [-54, 54]
ax_bev.set_xlim(BEV_X_RANGE)
ax_bev.set_ylim(BEV_Y_RANGE)
ax_bev.set_aspect("equal")
ax_bev.grid(True, linestyle="--", alpha=0.7)
ax_bev.plot(0, 0, "s", color="black", markersize=8)
ax_bev.set_title("BEV Predictions", fontsize=14)

for box, label, score in zip(boxes, labels, scores):
    if score < SCORE_THR:
        continue
    x, y, z, w, l, h, yaw = box[:7]
    class_name = CLASSES[label]
    color = COLOR_MAP.get(class_name, "blue")

    corners_bev = np.array(
        [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]
    )
    rot_bev = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners_bev = (rot_bev @ corners_bev.T).T + np.array([x, y])
    ax_bev.add_patch(
        patches.Polygon(corners_bev, closed=True, fill=True, alpha=0.4, color=color)
    )
    ax_bev.plot(
        corners_bev[[0, 1, 2, 3, 0], 0],
        corners_bev[[0, 1, 2, 3, 0], 1],
        color=color,
        linewidth=2,
    )

legend_elements = [
    Patch(facecolor=color, label=name, alpha=0.7, edgecolor="black")
    for name, color in COLOR_MAP.items()
]
ax_bev.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=14,
    title="Classes",
    title_fontsize=14,
    framealpha=0.95,
    edgecolor="black",
)

plt.subplots_adjust(left=0.05, right=0.75, top=0.95, bottom=0.08)
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight", pad_inches=0.2)
print(f"Visualização salva com sucesso em: {OUTPUT_IMAGE}")