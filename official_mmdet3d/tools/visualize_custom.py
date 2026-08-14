# =================================================================
# BEVFusion - Novo Inference Script and Visualization (Front + BEV) 
# =================================================================

import os
import numpy as np

# Compatibilidade NumPy 2.x com versões antigas do Matplotlib
np.Inf = np.inf
np.NaN = np.nan
np.Infinity = np.inf

import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from matplotlib.patches import Patch
import mmengine

from mmdet3d.structures import (
    LiDARInstance3DBoxes,
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
)

from mmdet3d.apis import MultiModalityDet3DInferencer

# ===== CONFIGURAÇÕES =====
CONFIG_FILE = "projects/BEVFusion/configs/bevfusion_plastipak.py"
CHECKPOINT = "/workspace/results/training/plastipak/epoch_20.pth"
DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
PKL_FILE = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_train.pkl"
OUTPUT_IMAGE = "./prediction_visualization_final.png"
SCORE_THR = 0.001
SAMPLE_INDEX = 0

CLASSES = (
    "obstrucao",
    "empilhadeira",
    "carga",
    "maquina",
    "humano",
    "navegavel",
    "estrutura",
    "portapalete",
)
COLOR_MAP = {
    "obstrucao": "red",
    "empilhadeira": "orange",
    "carga": "magenta",
    "maquina": "blue",
    "humano": "cyan",
    "navegavel": "green",
    "estrutura": "yellow",
    "portapalete": "brown",
}

# 1. Carregar sample do .pkl
data_list = mmengine.load(PKL_FILE)
sample = data_list[SAMPLE_INDEX]

# Apenas garantimos que os caminhos são absolutos para a pipeline conseguir ler do disco
if not os.path.isabs(sample["lidar_path"]):
    sample["lidar_path"] = os.path.join(DATA_ROOT, sample["lidar_path"])

for cam_key in list(sample["images"].keys()):
    if not os.path.isabs(sample["images"][cam_key]["img_path"]):
        sample["images"][cam_key]["img_path"] = os.path.join(
            DATA_ROOT, sample["images"][cam_key]["img_path"]
        )

# Substitui a string pela classe chamável (callable) esperada pelo modelo
if "box_type_3d" in sample and isinstance(sample["box_type_3d"], str):
    if "Camera" in sample["box_type_3d"]:
        sample["box_type_3d"] = CameraInstance3DBoxes
    elif "Depth" in sample["box_type_3d"]:
        sample["box_type_3d"] = DepthInstance3DBoxes
    else:
        sample["box_type_3d"] = LiDARInstance3DBoxes
elif "box_type_3d" not in sample:
    # Caso a chave não tenha sido salva no pkl, força o padrão do BEVFusion
    sample["box_type_3d"] = LiDARInstance3DBoxes

# 2. Criar inferenciador
inferencer = MultiModalityDet3DInferencer(
    model=CONFIG_FILE,
    weights=CHECKPOINT,
    device="cuda:0"
)

# ============================================================
# BYPASS 1
# Não deixar o inferencer tentar interpretar nosso sample
# ============================================================

inferencer._inputs_to_list = lambda inputs, **kwargs: inputs

# ============================================================
# BYPASS 2
# Manter o Det3DDataSample produzido pelo modelo
# mas devolver no formato esperado pelo Base3DInferencer
# ============================================================

def custom_postprocess(preds, *args, **kwargs):

    print("\n========== POSTPROCESS ==========")
    print("tipo preds:", type(preds))

    if isinstance(preds, list):
        print("len preds:", len(preds))

        if len(preds) > 0:
            print("tipo preds[0]:", type(preds[0]))

    print("=================================\n")

    return {
        "predictions": preds,
        "visualization": None,
    }

inferencer.postprocess = custom_postprocess

# 3. Fazer inferência
result = inferencer(
    [sample],
    return_vis=False,  # <--- Desliga o visualizador interno problemático
    show=False,
    draw_pred=False,  # <--- Diz para a biblioteca não tentar desenhar as caixas
    pred_score_thr=SCORE_THR,
)

print("\n========== RESULTADO ==========")
print("tipo result:", type(result))
print("keys:", result.keys())

predictions = result["predictions"]

print("tipo predictions:", type(predictions))
print("len predictions:", len(predictions))

prediction = predictions[0]

print("tipo prediction:", type(prediction))
print("metainfo:", prediction.metainfo)

print(
    "tem pred_instances_3d:",
    hasattr(prediction, "pred_instances_3d")
)

pred_instances = prediction.pred_instances_3d

print("Número de caixas:", len(pred_instances))

print(
    "scores:",
    pred_instances.scores_3d.detach().cpu().numpy()
)

print(
    "labels:",
    pred_instances.labels_3d.detach().cpu().numpy()
)

print(
    "boxes:",
    pred_instances.bboxes_3d.tensor.detach().cpu().numpy()
)

print("===============================\n")

# 4. Extrair predições (O resto do seu código segue normalmente)
# Como desliguei o return_vis, o resultado virá um pouco diferente:
# 'result' terá apenas um dicionário contendo as 'predictions'.
predictions = result["predictions"][0]
pred_instances = predictions.pred_instances_3d

print("BOX TYPE:", type(pred_instances.bboxes_3d))
print("BOX SHAPE:", pred_instances.bboxes_3d.tensor.shape)
print("LABELS:", pred_instances.labels_3d)
print("SCORES:", pred_instances.scores_3d)

print("\n" + "=" * 70)
print("DIAGNÓSTICO DAS PREDIÇÕES")
print("=" * 70)

scores = pred_instances.scores_3d.detach().cpu().numpy()
labels = pred_instances.labels_3d.detach().cpu().numpy()
boxes = pred_instances.bboxes_3d.tensor.detach().cpu().numpy()

print("Número de predições:", len(scores))

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

print("=" * 70)

boxes = pred_instances.bboxes_3d.tensor.cpu().numpy()
labels = pred_instances.labels_3d.cpu().numpy()
scores = pred_instances.scores_3d.cpu().numpy()

# 5. Visualizar
first_cam = list(sample["images"].keys())[0]

img_path = sample["images"][first_cam]["img_path"]

print("Imagem usada:", img_path)

front_img = cv2.imread(img_path)

if front_img is None:
    raise RuntimeError(
        f"Não foi possível carregar a imagem:\n{img_path}"
    )

front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)

fig, (ax_front, ax_bev) = plt.subplots(
    1, 2,
    figsize=(24, 10)
)

ax_front.imshow(front_img)
ax_front.set_title(
    f"Camera ({first_cam})",
    fontsize=14
)
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
print(f"Visualização salva em {OUTPUT_IMAGE}")

