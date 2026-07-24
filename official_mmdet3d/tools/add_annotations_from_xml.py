import xml.etree.ElementTree as ET
import numpy as np
import mmengine
from collections import defaultdict

PKL_PATH = "/mnt/53cbd82b-cb4d-4d12-af28-db5560fa258d/datasets/BEVLOG/finetunning/bevfusion_dataset.pkl"
FRAME_LIST_PATH = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/record_2025-10-15_09-54-30/annotations/pointclouds/frame_list.txt"
XML_PATH = "/workspace/official_mmdet3d/data/BEVLOG/finetunning/record_2025-10-15_09-54-30/annotations/pointclouds/tracklet_labels.xml"
OUTPUT_PKL = PKL_PATH

CLASS_MAPPING = {
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
    frame_to_ts = {}
    with open(frame_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            ts = float(parts[1])
            frame_to_ts[idx] = round(ts, 6)
    return frame_to_ts

def parse_xml(xml_path, frame_to_ts):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = defaultdict(list)
    total_tracklets = 0
    skipped = 0
    
    # Encontra TODOS os elementos <item> em qualquer lugar do XML
    for elem in root.findall('.//item'):
        # Verifica se este <item> é um tracklet (tem objectType, first_frame e poses)
        object_type_elem = elem.find('objectType')
        first_frame_elem = elem.find('first_frame')
        poses_elem = elem.find('poses')
        
        if object_type_elem is None or first_frame_elem is None or poses_elem is None:
            # Não é um tracklet (provavelmente é um <item> interno da pose ou versão)
            continue
        
        total_tracklets += 1
        
        # Extrai campos
        obj_type = object_type_elem.text
        label = CLASS_MAPPING.get(obj_type, -1)
        if label == -1:
            skipped += 1
            continue
        
        frame_idx = int(first_frame_elem.text)
        timestamp = frame_to_ts.get(frame_idx)
        if timestamp is None:
            skipped += 1
            continue
        
        h = float(elem.find('h').text)
        w = float(elem.find('w').text)
        l = float(elem.find('l').text)
        
        # Pose: dentro de <poses> há um <item> com tx, ty, tz, rx, ry, rz
        pose_item = poses_elem.find('item')
        if pose_item is None:
            skipped += 1
            continue
        
        tx = float(pose_item.find('tx').text)
        ty = float(pose_item.find('ty').text)
        tz = float(pose_item.find('tz').text)
        rx = float(pose_item.find('rx').text)
        ry = float(pose_item.find('ry').text)
        rz = float(pose_item.find('rz').text)
        
        yaw = rz  # rotação em torno de Z
        
        bbox = np.array([tx, ty, tz, w, l, h, yaw], dtype=np.float32)
        annotations[timestamp].append((bbox, label))
    
    print(f"Total tracklets: {total_tracklets}, skipped: {skipped}, válidos: {sum(len(v) for v in annotations.values())}")
    return annotations

def main():
    print("Carregando frame_list...")
    frame_to_ts = load_frame_list(FRAME_LIST_PATH)
    print(f"Frames carregados: {len(frame_to_ts)}")
    
    print("Parseando XML...")
    annotations = parse_xml(XML_PATH, frame_to_ts)
    print(f"Timestamps com anotações: {len(annotations)}")
    
    print("Carregando dataset .pkl...")
    data = mmengine.load(PKL_PATH)
    print(f"Total samples: {len(data)}")
    
    updated = 0
    for sample in data:
        ts = round(sample['timestamp'], 6)
        ann_list = annotations.get(ts, [])
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
    
    print(f"Samples atualizados com anotações: {updated}")
    mmengine.dump(data, OUTPUT_PKL)
    print(f"Dataset salvo em {OUTPUT_PKL}")

if __name__ == "__main__":
    main()