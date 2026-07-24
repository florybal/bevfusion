import json
import os
import numpy as np
import mmengine

DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
CALIB_JSON = os.path.join(DATA_ROOT, "matrizes_finais_bev.json")
MANIFEST_JSON = os.path.join(DATA_ROOT, "manifests/manifest.json")
OUTPUT_PKL = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset.pkl"

LIDAR_SENSOR = "lidar_front_link"
CAMERA_NAMES = {
    "camera_left_calib_link": "fisheye_left",
    "camera_right_calib_link": "fisheye_right",
    "camera_zed_left_calib_link": "zed_left",
    "camera_zed_right_calib_link": "zed_right",
}

# Matriz P (projeção estéreo retificada)
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

    # Dicionário com timestamps arredondados (6 casas)
    manifest_dict = {}
    for item in manifest:
        ts = round_ts(float(item['timestamp']), 6)
        manifest_dict[ts] = item

    data_list = []
    matched = 0
    for i, entry in enumerate(calib_data):
        if i % 50 == 0:
            print(f"Processando {i}/{len(calib_data)}...")
        ts_calib = round_ts(entry['timestamp'], 6)
        manifest_item = manifest_dict.get(ts_calib)
        if manifest_item is None:
            # tenta com 5 casas (tolerância)
            ts_alt = round_ts(entry['timestamp'], 5)
            manifest_item = manifest_dict.get(ts_alt)
            if manifest_item is None:
                continue
        matched += 1
        T_lidar_base = get_transform(entry, LIDAR_SENSOR)
        images = {}
        for json_cam, ds_cam in CAMERA_NAMES.items():
            T_cam_base = get_transform(entry, json_cam)
            T_lidar2cam = np.linalg.inv(T_cam_base) @ T_lidar_base
            img_rel = manifest_item['files'][ds_cam]
            img_path = os.path.join(DATA_ROOT, img_rel)
            images[ds_cam] = {
                'img_path': img_path,
                'cam2img': INTRINSICS[json_cam],
                'lidar2cam': T_lidar2cam,
            }
        lidar_rel = manifest_item['files']['lidar']
        lidar_path = os.path.join(DATA_ROOT, lidar_rel)
        sample = {
            'timestamp': entry['timestamp'],
            'lidar_path': lidar_path,
            'images': images,
            'box_type_3d': 'LiDAR',
            'box_mode_3d': 'LIDAR',
        }
        data_list.append(sample)

    print(f"Total calib: {len(calib_data)}, matches: {matched}, samples: {len(data_list)}")
    if data_list:
        mmengine.dump(data_list, OUTPUT_PKL)
        print(f"Salvo em {OUTPUT_PKL}")
    else:
        print("Nenhum sample gerado.")

if __name__ == "__main__":
    main()