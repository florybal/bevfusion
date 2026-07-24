import os
import sys
import numpy as np
# Monkey patch para compatibilidade com NumPy 2.0
if not hasattr(np, 'Inf'):
    np.Inf = np.inf

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mmengine import Config
from mmdet3d.apis import init_model
from mmdet3d.structures import Det3DDataSample
import mmengine
import cv2

# ===== CONFIGURAÇÕES =====
CONFIG_FILE = "projects/BEVFusion/configs/bevfusion_plastipak.py"
CHECKPOINT = "/workspace/official_mmdet3d/work_dirs/bevfusion_plastipak/epoch_6.pth"
DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
PKL_FILE = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_train.pkl"
OUTPUT_IMAGE = "./prediction_visualization.png"
SCORE_THR = 0.1
SAMPLE_INDEX = 0

# Classes e cores
CLASSES = ('obstrucao', 'empilhadeira', 'carga', 'maquina',
           'humano', 'navegavel', 'estrutura', 'portapalete')
COLOR_MAP = {
    'obstrucao': 'red', 'empilhadeira': 'orange', 'carga': 'magenta',
    'maquina': 'blue', 'humano': 'cyan', 'navegavel': 'green',
    'estrutura': 'yellow', 'portapalete': 'brown'
}

# 1. Carregar modelo
cfg = Config.fromfile(CONFIG_FILE)
model = init_model(cfg, CHECKPOINT, device='cuda:0')
model.eval()

# 2. Carregar sample do .pkl
data_list = mmengine.load(PKL_FILE)
sample = data_list[SAMPLE_INDEX]

# 3. Carregar nuvem de pontos (manual)
lidar_path = sample['lidar_path']
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
points_tensor = torch.from_numpy(points).float().unsqueeze(0).cuda()

# 4. Carregar e pré-processar imagens (manual)
cam_names = list(sample['images'].keys())
imgs_list = []
img_paths = []
for cam_name in cam_names:
    cam_info = sample['images'][cam_name]
    img_path = cam_info['img_path']
    img_paths.append(img_path)
    # Ler imagem
    img = plt.imread(img_path)
    # Redimensionar para 256x704
    img_resized = cv2.resize(img, (704, 256), interpolation=cv2.INTER_LINEAR)
    # Converter para tensor (CHW)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1)
    # Normalizar (ImageNet)
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)
    img_tensor = (img_tensor * 255.0 - mean) / std
    imgs_list.append(img_tensor)

imgs_tensor = torch.stack(imgs_list).unsqueeze(0).cuda()  # (1, N, 3, H, W)

# 5. Extrair calibrações (CPU)
cam2imgs = []
lidar2cams = []
lidar2imgs = []
for cam_name in cam_names:
    cam_info = sample['images'][cam_name]
    K = cam_info['cam2img']
    T_lidar2cam = cam_info['lidar2cam']
    cam2imgs.append(torch.from_numpy(K).float())
    lidar2cams.append(torch.from_numpy(T_lidar2cam).float())
    lidar2img = K @ T_lidar2cam[:3, :]
    lidar2imgs.append(torch.from_numpy(lidar2img).float())

cam2img = torch.stack(cam2imgs)          # (N, 3, 3) CPU
lidar2cam = torch.stack(lidar2cams)      # (N, 4, 4) CPU
lidar2img = torch.stack(lidar2imgs)      # (N, 3, 4) CPU
cam2lidar = torch.linalg.inv(lidar2cam)  # (N, 4, 4) CPU
img_aug_matrix = torch.eye(4).unsqueeze(0).repeat(len(cam_names), 1, 1)  # CPU

# 6. Batch inputs (GPU)
batch_inputs = {
    'points': points_tensor,
    'imgs': imgs_tensor,
}

# 7. Metadados (CPU) - corrigindo box_type_3d
# O campo 'box_type_3d' deve ser uma classe que pode ser instanciada, não uma string.
# Vamos usar LiDARInstance3DBoxes ou deixar como string e depois converter.
# Como o erro anterior era 'str' object is not callable, vamos passar uma string e o modelo
# deveria lidar com isso, mas parece que não. Vamos importar e usar a classe.
from mmdet3d.structures import LiDARInstance3DBoxes
box_type_cls = LiDARInstance3DBoxes

batch_input_metas = [{
    'cam2img': cam2img,
    'lidar2cam': lidar2cam,
    'lidar2img': lidar2img,
    'cam2lidar': cam2lidar,
    'img_aug_matrix': img_aug_matrix,
    'box_type_3d': box_type_cls,  # <-- usar a classe, não a string
    'box_mode_3d': 'LIDAR',
    'sample_idx': SAMPLE_INDEX,
    'lidar_path': lidar_path,
    'img_path': img_paths,
    'num_pts_feats': 4,
}]

data_sample = Det3DDataSample()
data_sample.set_metainfo(batch_input_metas[0])

# 8. Inferência
with torch.no_grad():
    result = model.predict(batch_inputs, [data_sample])[0]
# ... após a inferência ...

# Criar figura maior
fig, (ax_front, ax_bev) = plt.subplots(1, 2, figsize=(24, 10))

# --- Câmera frontal (primeira câmera) ---
first_cam = cam_names[0]
front_img = plt.imread(sample['images'][first_cam]['img_path'])
ax_front.imshow(front_img)
ax_front.set_title(f'Camera ({first_cam})', fontsize=14)
ax_front.axis('off')  # remove eixos para ficar mais limpo

# --- BEV ---
BEV_X_RANGE = [-54, 54]
BEV_Y_RANGE = [-54, 54]
ax_bev.set_xlim(BEV_X_RANGE)
ax_bev.set_ylim(BEV_Y_RANGE)
ax_bev.set_aspect('equal')
ax_bev.set_xlabel('X (m)', fontsize=12)
ax_bev.set_ylabel('Y (m)', fontsize=12)
ax_bev.grid(True, linestyle='--', alpha=0.7)
ax_bev.plot(0, 0, 's', color='black', markersize=8)
ax_bev.set_title('BEV Predictions', fontsize=14)

if hasattr(result, 'pred_instances_3d'):
    pred = result.pred_instances_3d
    keep = pred.scores_3d > SCORE_THR
    boxes = pred.bboxes_3d[keep].tensor.cpu().numpy()
    labels = pred.labels_3d[keep].cpu().numpy()
    scores = pred.scores_3d[keep].cpu().numpy()

    for i, box in enumerate(boxes):
        x, y, z, w, l, h, yaw = box[:7]
        class_name = CLASSES[labels[i]]
        color = COLOR_MAP.get(class_name, 'blue')

        # Desenhar polígono (sem texto)
        corners_bev = np.array([
            [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
        ])
        rot_bev = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw), np.cos(yaw)]])
        corners_bev = (rot_bev @ corners_bev.T).T + np.array([x, y])
        ax_bev.add_patch(patches.Polygon(corners_bev, closed=True, fill=True,
                                         alpha=0.4, color=color))
        ax_bev.plot(corners_bev[[0,1,2,3,0], 0], corners_bev[[0,1,2,3,0], 1],
                    color=color, linewidth=2)

# ===== LEGENDA (maior e mais visível) =====
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=name, alpha=0.7, edgecolor='black')
                   for name, color in COLOR_MAP.items()]

# Colocar a legenda à direita, fora do gráfico
ax_bev.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1.02, 0.5),   # fora à direita
              fontsize=14,                  # fonte maior
              title='Classes',
              title_fontsize=14,
              framealpha=0.95,
              edgecolor='black')

# Ajustar layout para dar espaço à legenda
plt.subplots_adjust(left=0.05, right=0.75, top=0.95, bottom=0.08)  # right=0.75 deixa espaço para legenda

# Salvar com alta resolução
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"Visualização salva em {OUTPUT_IMAGE}")