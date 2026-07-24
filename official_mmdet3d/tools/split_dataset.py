import json
import mmengine
import os
from collections import defaultdict

# Caminhos
PKL_PATH = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset.pkl"
DATA_ROOT = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/"
OUTPUT_DIR = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/"

# Carregar splits (listas de IDs)
train_ids = json.load(open(os.path.join(DATA_ROOT, "manifests/train.json")))
val_ids = json.load(open(os.path.join(DATA_ROOT, "manifests/val.json")))
test_ids = json.load(open(os.path.join(DATA_ROOT, "manifests/test.json")))

# Carregar dataset completo
data = mmengine.load(PKL_PATH)
print(f"Total samples no dataset completo: {len(data)}")

# Construir dicionário: timestamp arredondado (6 casas) -> sample (completo)
timestamp_to_sample = {}
for sample in data:
    ts = round(sample['timestamp'], 6)
    timestamp_to_sample[ts] = sample

def id_to_timestamp(id_str):
    """Extrai o timestamp do ID (formato 'record_.../timestamp')."""
    parts = id_str.split('/')
    if len(parts) == 2:
        return float(parts[1])
    return None

def filter_samples(id_list):
    """Filtra samples pelo ID, preservando o sample completo (incluindo annos)."""
    filtered = []
    for id_str in id_list:
        ts = id_to_timestamp(id_str)
        if ts is not None:
            ts_rounded = round(ts, 6)
            sample = timestamp_to_sample.get(ts_rounded)
            if sample is not None:
                filtered.append(sample)  # mantém o sample original intacto
    return filtered

# Gerar splits
train_samples = filter_samples(train_ids)
val_samples = filter_samples(val_ids)
test_samples = filter_samples(test_ids)

print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")
print(f"Test samples: {len(test_samples)}")

# Salvar
mmengine.dump(train_samples, os.path.join(OUTPUT_DIR, "bevfusion_dataset_train.pkl"))
mmengine.dump(val_samples, os.path.join(OUTPUT_DIR, "bevfusion_dataset_val.pkl"))
mmengine.dump(test_samples, os.path.join(OUTPUT_DIR, "bevfusion_dataset_test.pkl"))

print("Splits salvos com sucesso.")