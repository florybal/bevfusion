import os
import random
import numpy as np
import mmengine
import open3d as o3d
from pathlib import Path


# ======================================================
# CONFIG
# ======================================================

DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning"
OUTPUT_ROOT = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning"
MANIFEST = os.path.join(DATA_ROOT,"manifests/manifest.json")
BEV_FOLDER = os.path.join(DATA_ROOT,"output_bevs/output")
BIN_FOLDER = os.path.join(OUTPUT_ROOT,"bin")

os.makedirs(BIN_FOLDER, exist_ok=True)

# ======================================================
# LOAD MANIFEST
# ======================================================

manifest = mmengine.load(MANIFEST)
print("Frames no manifesto:", len(manifest))

CALIB_JSON = os.path.join(DATA_ROOT, "matrizes_finais_bev.json")

calib = mmengine.load(CALIB_JSON)

calib_ts = np.array(
    [float(x["timestamp"]) for x in calib],
    dtype=np.float64
)

MAX_TIME_DIFF = 0.15

dataset = []

missing_bev = 0
missing_pcd = 0

# ======================================================
# PROCESS
# ======================================================

for i,item in enumerate(manifest):

    timestamp = float(item["timestamp"])

    print(f"\n[{i+1}/{len(manifest)}] {timestamp}")

    # PCD
    pcd_rel = item["files"]["lidar"]

    pcd_path = os.path.join(DATA_ROOT, pcd_rel)
    
    if not os.path.exists(pcd_path):
        print("SEM PCD:", pcd_path)
        missing_pcd += 1
        continue

    # -------------------------
    # BEV MASK
    # -------------------------
    if i < 10:
        print("Manifest:", f"{timestamp:.6f}")

        import glob

        arquivos = sorted(glob.glob(os.path.join(BEV_FOLDER, "*.npy")))

        for a in arquivos[:5]:
            print("BEV:", os.path.basename(a))
    bev_path = os.path.join(BEV_FOLDER,f"{timestamp:.6f}_bev_label.npy")
    if not os.path.exists(bev_path):
        print("SEM BEV:", bev_path)

        missing_bev += 1
        continue
    timestamp = float(item["timestamp"])

    idx = np.argmin(np.abs(calib_ts - timestamp))

    diff = abs(calib_ts[idx] - timestamp)

    if diff > MAX_TIME_DIFF:
        print("SEM CALIB:", timestamp)
        continue

    # -------------------------
    # CONVERTE PCD -> BIN
    # -------------------------

    bin_path = os.path.join(BIN_FOLDER, f"{timestamp:.6f}.bin")


    if not os.path.exists(bin_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)

        if pcd.has_colors():
            colors = np.asarray( pcd.colors, dtype=np.float32)
            intensity = (colors[:,0] + colors[:,1] + colors[:,2]) / 3.0
        else:
            intensity = np.zeros(xyz.shape[0],dtype=np.float32)

        points = np.column_stack((xyz,intensity))
        points.astype(np.float32).tofile(bin_path)

    # IMAGES
    images = {}

    for cam in [
        "fisheye_left",
        "fisheye_right",
        "zed_left",
        "zed_right"
    ]:
        img_rel = item["files"][cam]
        images[cam] = { "img_path": os.path.join(DATA_ROOT, img_rel) }


    # -------------------------
    # SAMPLE FINAL
    # -------------------------

    sample = {
        "timestamp": timestamp,
        "lidar_path": bin_path,
        "lidar_points":
        {
            "lidar_path": bin_path,
            "num_pts_feats": 4
        },
        "images":
            images,
        # SEGMENTAÇÃO BEV
        "gt_bev_seg":
            bev_path,
        "box_type_3d": "LiDAR",
        "box_mode_3d": "LIDAR"
    }
    dataset.append(sample)

print("\n======================")
print("RESULTADO")
print("======================")

print("Samples:", len(dataset) )
print("Sem BEV:", missing_bev)
print("Sem PCD:", missing_pcd )

# ======================================================
# SAVE FULL
# ======================================================

FULL = os.path.join( OUTPUT_ROOT, "bevfusion_dataset.pkl" )
mmengine.dump( dataset, FULL)
print("\nSalvo:", FULL)

# ======================================================
# SPLIT
# ======================================================

random.seed(42)
random.shuffle(dataset)
n=len(dataset)

train_end=int(n*0.70)
val_end=int(n*0.85 )

train = dataset[:train_end]
val = dataset[train_end:val_end]
test = dataset[val_end:]

mmengine.dump(
    train,
    os.path.join(
        OUTPUT_ROOT, "bevfusion_dataset_train.pkl"
    )
)

mmengine.dump(
    val,
    os.path.join(
        OUTPUT_ROOT, "bevfusion_dataset_val.pkl"
    )
)

mmengine.dump(
    test,
    os.path.join(
        OUTPUT_ROOT, "bevfusion_dataset_test.pkl"
    )
)

print("\nSPLIT")
print( "Train:", len(train) )
print( "Val:", len(val))
print( "Test:", len(test) )