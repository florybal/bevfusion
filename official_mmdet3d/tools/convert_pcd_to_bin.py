import os
import numpy as np
import mmengine
import open3d as o3d
import random
from pathlib import Path

# ===== CONFIGURAÇÕES =====
DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
MANIFEST_PATH = os.path.join(DATA_ROOT, "manifests/manifest.json")
CALIB_PATH = os.path.join(DATA_ROOT, "matrizes_finais_bev.json")
OUTPUT_DIR = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/"
BIN_DIR = os.path.join(OUTPUT_DIR, "bin")
os.makedirs(BIN_DIR, exist_ok=True)

# ===== 1. Carregar manifest =====
manifest = mmengine.load(MANIFEST_PATH)
print(f"Total frames no manifesto: {len(manifest)}")

# ===== 2. Converter .pcd para .bin =====
data_list = []
for item in manifest:
    timestamp = float(item['timestamp'])
    lidar_rel = item['files']['lidar']  # ex: "record_.../annotations/pointclouds/velodyne_points/1760...pcd"
    pcd_path = os.path.join(DATA_ROOT, lidar_rel)
    if not os.path.exists(pcd_path):
        print(f"Arquivo .pcd não encontrado: {pcd_path}")
        continue
    # Caminho .bin
    basename = os.path.basename(pcd_path).replace('.pcd', '.bin')
    bin_path = os.path.join(BIN_DIR, basename)
    # Converter
    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)
            intensity = (colors[:, 0] + colors[:, 1] + colors[:, 2]) / 3.0
        else:
            intensity = np.zeros(points.shape[0], dtype=np.float32)
        data_4 = np.hstack([points, intensity.reshape(-1, 1)]).astype(np.float32)
        with open(bin_path, 'wb') as f:
            f.write(data_4.tobytes())
        # Verifica tamanho
        size = os.path.getsize(bin_path)
        if size % 16 != 0:
            print(f"AVISO: {bin_path} tem tamanho {size} (não múltiplo de 16)")
        # Monta sample
        images = {}
        for cam in ['fisheye_left', 'fisheye_right', 'zed_left', 'zed_right']:
            img_rel = item['files'][cam]
            img_path = os.path.join(DATA_ROOT, img_rel)
            # Nota: Você precisará das intrínsecas e extrínsecas – isso é apenas um placeholder
            # Recomendo usar a calibração estática média ou os dados do matrizes_finais_bev.json
            images[cam] = {'img_path': img_path}
        sample = {
            'timestamp': timestamp,
            'lidar_path': bin_path,
            'images': images,
            'box_type_3d': 'LiDAR',
            'box_mode_3d': 'LIDAR',
            # As anotações serão adicionadas depois
        }
        data_list.append(sample)
        print(f"Convertido: {basename}")
    except Exception as e:
        print(f"Erro ao converter {pcd_path}: {e}")

print(f"Total de samples convertidos: {len(data_list)}")

# ===== 3. Adicionar anotações (se disponíveis) =====
# Aqui você deve integrar a lógica do add_annotations_from_xml.py
# Por simplicidade, vamos pular por enquanto

# ===== 4. Salvar dataset completo =====
full_pkl = os.path.join(OUTPUT_DIR, "bevfusion_dataset_full.pkl")
mmengine.dump(data_list, full_pkl)
print(f"Dataset completo salvo em {full_pkl}")

# ===== 5. Criar splits =====
random.seed(42)
random.shuffle(data_list)
n = len(data_list)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

mmengine.dump(data_list[:train_end], os.path.join(OUTPUT_DIR, "bevfusion_dataset_train.pkl"))
mmengine.dump(data_list[train_end:val_end], os.path.join(OUTPUT_DIR, "bevfusion_dataset_val.pkl"))
mmengine.dump(data_list[val_end:], os.path.join(OUTPUT_DIR, "bevfusion_dataset_test.pkl"))
print(f"Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")