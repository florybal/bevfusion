# ==============================================================
# BEVFusion - Divide o dataset .pkl em treino, validação e teste
# ==============================================================

import mmengine
import random

data = mmengine.load("/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset.pkl")

random.seed(42)
random.shuffle(data)

n = len(data)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

mmengine.dump(train, "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_train.pkl")
mmengine.dump(val, "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_val.pkl")
mmengine.dump(test, "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_test.pkl")

print(len(train), len(val), len(test))