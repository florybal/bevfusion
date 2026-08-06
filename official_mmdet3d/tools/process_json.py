import os
import json
import numpy as np
import mmengine

DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning"
PKL_PATH = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset.pkl"
CALIB_JSON = os.path.join(DATA_ROOT, "matrizes_finais_bev.json")
BEV_DIR = os.path.join(DATA_ROOT, "output_bevs/output")

MAX_TIME_DIFF = 0.15

LIDAR_SENSOR = "lidar_front_link"

CAMERA_NAMES = {
    "camera_left_calib_link": "fisheye_left",
    "camera_right_calib_link": "fisheye_right",
    "camera_zed_left_calib_link": "zed_left",
    "camera_zed_right_calib_link": "zed_right",
}

P_MATRICES = {
    "camera_left_calib_link": np.array([[500.391,0,343.318,0],
                                        [0,545.179,280.577,0],
                                        [0,0,1,0]],dtype=np.float32),

    "camera_right_calib_link": np.array([[492.441,0,645.295,0],
                                         [0,491.652,348.315,0],
                                         [0,0,1,0]],dtype=np.float32),

    "camera_zed_left_calib_link": np.array([[534.755,0,644.505,0],
                                            [0,534.790,347.236,0],
                                            [0,0,1,0]],dtype=np.float32),

    "camera_zed_right_calib_link": np.array([[534.250,0,638.765,0],
                                             [0,534.285,338.202,0],
                                             [0,0,1,0]],dtype=np.float32),
}

INTRINSICS = {k:v[:,:3] for k,v in P_MATRICES.items()}

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_transform(entry, sensor):
    return np.array(entry["sensors"][sensor], dtype=np.float32)

print("Carregando PKL...")
dataset = mmengine.load(PKL_PATH)

print("Carregando calibração...")
calib = load_json(CALIB_JSON)

calib_ts = np.array(
    [float(x["timestamp"]) for x in calib],
    dtype=np.float64
)

print("Indexando máscaras BEV...")

bev_files = []

for f in os.listdir(BEV_DIR):
    if f.endswith("_bev_label.npy"):
        ts = float(f.replace("_bev_label.npy",""))
        bev_files.append((ts,f))

bev_files.sort()

def get_bev(ts):

    if not bev_files:
        return None

    closest = min(
        bev_files,
        key=lambda x: abs(x[0]-ts)
    )

    if abs(closest[0]-ts) > MAX_TIME_DIFF:
        return None

    return os.path.join(BEV_DIR, closest[1])

updated = 0

for sample in dataset:

    ts = float(sample["timestamp"])
    
    idx = np.argmin(np.abs(calib_ts - ts))
    calib_entry = calib[idx]

    if abs(calib_ts[idx] - ts) > MAX_TIME_DIFF:
        print("SEM CALIB:", ts)
        continue

    T_lidar = get_transform(calib_entry, LIDAR_SENSOR)

    for calib_name, ds_name in CAMERA_NAMES.items():

        if ds_name not in sample["images"]:
            continue

        T_cam = get_transform(calib_entry, calib_name)
        lidar2cam = np.linalg.inv(T_cam) @ T_lidar

        sample["images"][ds_name]["cam2img"] = INTRINSICS[calib_name].tolist()
        sample["images"][ds_name]["lidar2cam"] = lidar2cam.tolist()

    sample["gt_bev_seg"] = get_bev(ts)

    updated += 1

print(f"Atualizados: {updated}/{len(dataset)}")

mmengine.dump(dataset, PKL_PATH)
print("PKL salvo.")