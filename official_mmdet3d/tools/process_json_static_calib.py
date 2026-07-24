import json
import os
import numpy as np
import mmengine
from pathlib import Path

DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
CALIB_JSON = os.path.join(DATA_ROOT, "matrizes_finais_bev.json")
MANIFEST_JSON = os.path.join(DATA_ROOT, "manifests/manifest.json")
OUTPUT_PKL = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_all.pkl"

LIDAR_SENSOR = "lidar_front_link"

CAMERA_NAMES = {
    "camera_left_calib_link": "fisheye_left",
    "camera_right_calib_link": "fisheye_right",
    "camera_zed_left_calib_link": "zed_left",
    "camera_zed_right_calib_link": "zed_right",
}

# Intrínsecas (P) - já extraídas
P_MATRICES = {
    "camera_left_calib_link": np.array([[500.391, 0, 343.318, 0],
                                        [0, 545.179, 280.577, 0],
                                        [0, 0, 1, 0]], dtype=np.float32),
    "camera_right_calib_link": np.array([[492.441, 0, 645.295, 0],
                                         [0, 491.652, 348.315, 0],
                                         [0, 0, 1, 0]], dtype=np.float32),
    "camera_zed_left_calib_link": np.array([[534.755, 0, 644.505, 0],
                                            [0, 534.79, 347.236, 0],
                                            [0, 0, 1, 0]], dtype=np.float32),
    "camera_zed_right_calib_link": np.array([[534.25, 0, 638.765, 0],
                                             [0, 534.285, 338.202, 0],
                                             [0, 0, 1, 0]], dtype=np.float32),
}
INTRINSICS = {cam: P[:, :3] for cam, P in P_MATRICES.items()}

def load_json(f):
    with open(f, 'r') as fp:
        return json.load(fp)

def get_transform(entry, sensor):
    return np.array(entry['sensors'][sensor], dtype=np.float32)

def round_ts(ts, decimals=6):
    return round(ts, decimals)

def main():
    print("Carregando arquivos...")
    calib_data = load_json(CALIB_JSON)
    manifest = load_json(MANIFEST_JSON)

    # Dicionário do manifesto: timestamp (6 casas) -> item
    manifest_dict = {}
    for item in manifest:
        ts = round_ts(float(item['timestamp']), 6)
        manifest_dict[ts] = item

    # 1. Coletar as matrizes lidar2cam dos 33 frames calibrados
    lidar2cam_list = {cam: [] for cam in CAMERA_NAMES.keys()}
    for entry in calib_data:
        ts_calib = round_ts(entry['timestamp'], 6)
        # só processa se estiver no manifesto (33)
        if ts_calib not in manifest_dict:
            continue
        T_lidar_base = get_transform(entry, LIDAR_SENSOR)
        for json_cam in CAMERA_NAMES.keys():
            T_cam_base = get_transform(entry, json_cam)
            T_lidar2cam = np.linalg.inv(T_cam_base) @ T_lidar_base
            lidar2cam_list[json_cam].append(T_lidar2cam)

    # 2. Calcular a média para cada câmera
    avg_lidar2cam = {}
    for json_cam, mats in lidar2cam_list.items():
        if len(mats) == 0:
            raise ValueError(f"Nenhuma matriz para {json_cam}")
        avg_lidar2cam[json_cam] = np.mean(mats, axis=0)
        print(f"Média calculada para {json_cam} (baseado em {len(mats)} amostras)")

    # 3. Gerar samples para todos os frames do manifesto
    data_list = []
    total = len(manifest)
    for idx, item in enumerate(manifest):
        if idx % 50 == 0:
            print(f"Processando {idx}/{total}...")
        timestamp = float(item['timestamp'])
        bag = item['bag']

        images = {}
        for json_cam, ds_cam in CAMERA_NAMES.items():
            img_rel = item['files'][ds_cam]
            img_path = os.path.join(DATA_ROOT, img_rel)
            images[ds_cam] = {
                'img_path': img_path,
                'cam2img': INTRINSICS[json_cam],
                'lidar2cam': avg_lidar2cam[json_cam].copy(),
            }

        lidar_rel = item['files']['lidar']
        lidar_path = os.path.join(DATA_ROOT, lidar_rel)

        sample = {
            'timestamp': timestamp,
            'lidar_path': lidar_path,
            'images': images,
            'box_type_3d': 'LiDAR',
            'box_mode_3d': 'LIDAR',
        }
        data_list.append(sample)

    print(f"Total de samples gerados: {len(data_list)}")
    mmengine.dump(data_list, OUTPUT_PKL)
    print(f"Dataset salvo em {OUTPUT_PKL}")

if __name__ == "__main__":
    main()