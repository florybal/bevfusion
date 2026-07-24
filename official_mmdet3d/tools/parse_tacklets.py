import xml.etree.ElementTree as ET
import numpy as np
import mmengine
import os
from collections import defaultdict

PKL_PATH = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset_all.pkl"
XML_PATH = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/record_2025-10-15_09-54-30/annotations/pointclouds/tracklet_labels.xml"
FRAME_LIST_PATH = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/record_2025-10-15_09-54-30/annotations/pointclouds/frame_list.txt"
OUTPUT_PKL = PKL_PATH  # sobrescreve

CLASS_TO_ID = {
    'obstrucao': 0,
    'empilhadeira': 1,
    'carga': 2,
    'maquina': 3,
    'humano': 4,
    'navegavel': 5,
    'estrutura': 6,
    'portapalete': 7,
}

def load_frame_list(frame_list_path):
    """Retorna lista de timestamps (float) na ordem dos frames."""
    with open(frame_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Assume que cada linha é um nome de arquivo, ex: "1760532874.604852.pcd"
    timestamps = []
    for line in lines:
        # Extrai o timestamp: remove extensão .pcd
        if line.endswith('.pcd'):
            ts_str = line[:-4]  # remove '.pcd'
        else:
            ts_str = line
        timestamps.append(float(ts_str))
    return timestamps

def parse_xml(xml_path, timestamps):
    """Retorna dicionário timestamp -> lista de (bbox, label)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = defaultdict(list)
    
    # Encontrar todos os itens (tracklets)
    # A estrutura: <tracklets> ... <item> ... </item> ... </tracklets>
    # O root é <tracklets>, e seus filhos são <item>?
    # Na amostra, vemos <tracklets version="0" tracking_level="0" class_id="0">
    # e depois <count>4015</count>, <item_version>1</item_version>, e vários <item>.
    for item in root.findall('item'):
        object_type = item.find('objectType').text
        if object_type not in CLASS_TO_ID:
            print(f"Classe desconhecida: {object_type}")
            continue
        label = CLASS_TO_ID[object_type]
        
        h = float(item.find('h').text)
        w = float(item.find('w').text)
        l = float(item.find('l').text)
        first_frame = int(item.find('first_frame').text)
        
        # Acessar poses
        poses = item.find('poses')
        if poses is None:
            continue
        # Cada pose é um <item> dentro de <poses>
        pose_items = poses.findall('item')
        if not pose_items:
            continue
        # Pegamos a primeira pose (assumindo que cada tracklet tem uma pose)
        pose = pose_items[0]
        tx = float(pose.find('tx').text)
        ty = float(pose.find('ty').text)
        tz = float(pose.find('tz').text)
        # rx, ry, rz
        rx = float(pose.find('rx').text)
        ry = float(pose.find('ry').text)
        rz = float(pose.find('rz').text)
        # Ignoramos rx, ry (provavelmente rotação de cabeçalho? Mas assumimos que a orientação é rz)
        yaw = rz  # Ângulo no plano XY
        
        # Construir bbox: [x, y, z, w, l, h, yaw]
        bbox = np.array([tx, ty, tz, w, l, h, yaw], dtype=np.float32)
        
        # Verificar se first_frame está dentro do range
        if first_frame < 0 or first_frame >= len(timestamps):
            print(f"first_frame {first_frame} fora do range (0-{len(timestamps)-1})")
            continue
        ts = timestamps[first_frame]
        annotations[ts].append((bbox, label))
    
    return annotations

def main():
    print("Carregando dataset...")
    data = mmengine.load(PKL_PATH)
    print(f"Total samples: {len(data)}")
    
    print("Carregando frame_list...")
    timestamps = load_frame_list(FRAME_LIST_PATH)
    print(f"Total frames: {len(timestamps)}")
    
    print("Parseando XML...")
    annotations = parse_xml(XML_PATH, timestamps)
    print(f"Frames com anotações: {len(annotations)}")
    
    # Atualizar samples
    updated = 0
    for sample in data:
        ts = sample['timestamp']
        ts_rounded = round(ts, 6)
        ann_list = annotations.get(ts_rounded, [])
        if ann_list:
            bboxes = np.array([ann[0] for ann in ann_list], dtype=np.float32)
            labels = np.array([ann[1] for ann in ann_list], dtype=np.int64)
            sample['annos'] = {
                'bboxes_3d': bboxes,
                'labels_3d': labels,
            }
            updated += 1
        else:
            sample['annos'] = {
                'bboxes_3d': np.zeros((0, 7), dtype=np.float32),
                'labels_3d': np.zeros((0,), dtype=np.int64),
            }
    
    print(f"Samples atualizados: {updated}")
    mmengine.dump(data, OUTPUT_PKL)
    print(f"Salvo em {OUTPUT_PKL}")

if __name__ == "__main__":
    main()