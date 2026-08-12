import os.path as osp

import numpy as np
import torch
import mmengine

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .det3d_dataset import Det3DDataset

@DATASETS.register_module()  
class PlastipakDataset(Det3DDataset):
    CAMERA_NAMES = {
        'camera_left_calib_link': 'fisheye_left',
        'camera_right_calib_link': 'fisheye_right',
        'camera_zed_left_calib_link': 'zed_left',
        'camera_zed_right_calib_link': 'zed_right',
    }

    P_MATRICES = {
        'camera_left_calib_link': np.array([[500.391, 0, 343.318, 0],
                                            [0, 545.179, 280.577, 0],
                                            [0, 0, 1, 0]], dtype=np.float32),
        'camera_right_calib_link': np.array([[492.441, 0, 645.295, 0],
                                             [0, 491.652, 348.315, 0],
                                             [0, 0, 1, 0]], dtype=np.float32),
        'camera_zed_left_calib_link': np.array([[534.755, 0, 644.505, 0],
                                                [0, 534.79, 347.236, 0],
                                                [0, 0, 1, 0]], dtype=np.float32),
        'camera_zed_right_calib_link': np.array([[534.25, 0, 638.765, 0],
                                                 [0, 534.285, 338.202, 0],
                                                 [0, 0, 1, 0]], dtype=np.float32),
    }

    METAINFO = {
        'classes': (
            'obstrucao', 'empilhadeira', 'carga', 'maquina',
            'humano', 'navegavel', 'estrutura', 'portapalete'
        ),
            'palette': [
            [255, 0, 0],       # obstrucao
            [255, 140, 0],     # empilhadeira
            [139, 43, 226],    # carga
            [0, 0, 255],       # maquina
            [0, 170, 255],     # humano
            [0, 255, 0],       # navegavel
            [255, 255, 0],     # estrutura
            [205, 133, 63],    # portapalete
        ]
    }

    def __init__(self, *args, **kwargs):
        kwargs['filter_empty_gt'] = False
        kwargs['serialize_data'] = False
        super().__init__(*args, **kwargs)
        self._calib_data = None
        self._calib_ts = None

    def _load_calibration(self):
        if self._calib_data is not None:
            return

        if not getattr(self, 'data_root', None):
            self._calib_data = []
            self._calib_ts = np.zeros((0,), dtype=np.float64)
            return

        calib_path = osp.join(self.data_root, 'matrizes_finais_bev.json')
        if not osp.exists(calib_path):
            self._calib_data = []
            self._calib_ts = np.zeros((0,), dtype=np.float64)
            return

        self._calib_data = mmengine.load(calib_path)
        self._calib_ts = np.array(
            [float(entry['timestamp']) for entry in self._calib_data],
            dtype=np.float64)

    def _attach_calibration(self, data_info):
        images = data_info.get('images')
        if not images:
            return

        needs_calib = any(
            'lidar2cam' not in cam_item or 'cam2img' not in cam_item
            for cam_item in images.values())
        if not needs_calib:
            return

        self._load_calibration()
        if self._calib_data is None or len(self._calib_data) == 0:
            return

        timestamp = float(data_info.get('timestamp', 0.0))
        nearest_idx = int(np.argmin(np.abs(self._calib_ts - timestamp)))
        if abs(self._calib_ts[nearest_idx] - timestamp) > 1.2:
            return

        calib_entry = self._calib_data[nearest_idx]
        lidar_t = np.array(
            calib_entry['sensors']['lidar_front_link'],
            dtype=np.float32)

        for calib_name, ds_name in self.CAMERA_NAMES.items():
            cam_item = images.get(ds_name)
            if cam_item is None:
                continue
            if 'lidar2cam' in cam_item and 'cam2img' in cam_item:
                continue

            cam_t = np.array(calib_entry['sensors'][calib_name], dtype=np.float32)
            lidar2cam = np.linalg.inv(cam_t) @ lidar_t
            cam_item.setdefault('cam2img', self.P_MATRICES[calib_name][:, :3].tolist())
            cam_item.setdefault('lidar2cam', lidar2cam.tolist())

    def full_init(self):
        print("=== BEFORE FULL INIT ===")
        super().full_init()
        print("=== AFTER FULL INIT ===")
        print("data_list:", len(self.data_list))

    def load_data_list(self):
        data = mmengine.load(self.ann_file)

        if isinstance(data, dict) and 'data_list' in data:
            data_list = data['data_list']
        elif isinstance(data, list):
            data_list = data
        else:
            raise TypeError(f'Unsupported annotation type: {type(data)}')

        # Convert annos -> instances for MMDetection3D filtering
        for item in data_list:
            if 'annos' in item:
                annos = item['annos']
                instances = []
                boxes = annos.get('gt_bboxes_3d', [])   
                labels = annos.get('gt_labels_3d', [])  

                for box, label in zip(boxes, labels):
                    instances.append({
                        'bbox_3d': box,
                        'bbox_label_3d': int(label),
                        'bbox_label': int(label)
                    })
                item['instances'] = instances

        print("Loaded samples:", len(data_list))
        if data_list:
            print("First sample keys:", data_list[0].keys())
            print("Instances:", len(data_list[0].get('instances', [])))
        print("Total instances across all samples:", sum(len(item.get('instances', [])) for item in data_list))

        return data_list
    def get_data_info(self, idx):

        data_info = self.data_list[idx].copy()

        # Required by LoadPointsFromFile
        data_info['lidar_points'] = {
            'lidar_path': data_info['lidar_path']
        }
        if 'lidar_path' in data_info:
            data_info['lidar_points'] = {
                'lidar_path': data_info['lidar_path']
            }
        # Required by image loaders if cameras exist
        if 'images' in data_info:
            data_info['images'] = data_info['images']

        self._attach_calibration(data_info)

        data_info['box_type_3d'] = self.box_type_3d
        data_info['box_mode_3d'] = self.box_mode_3d

        # .pkl não tem 'sample_idx', mas o KittiMetric (e o
        # pipeline em geral) exige esse campo para casar predição com GT.
        if 'sample_idx' not in data_info:
            data_info['sample_idx'] = idx

        return data_info

    def get_ann_info(self, idx):
        data_info = self.data_list[idx]
        annos = data_info.get('annos', {})
        bboxes = annos.get('gt_bboxes_3d', np.zeros((0, 7), dtype=np.float32))
        labels = annos.get('gt_labels_3d', np.zeros((0,), dtype=np.int64))
        
        # Converte labels para nomes (gt_names)
        names = [self.METAINFO['classes'][int(label)] for label in labels if int(label) < len(self.METAINFO['classes'])]
        
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(bboxes_tensor, box_dim=7, with_yaw=True)

        print(f"Sample {idx}: labels={labels.tolist() if len(labels) else []}, names={names}")  # <-- ADICIONE

        return {
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': torch.tensor(labels, dtype=torch.long),
            'gt_names': names 
        }
    def filter_data(self):
        print("Before filter:", len(self.data_list))

        data_list = super().filter_data()

        print("After filter:", len(data_list))

        return data_list
    
    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        ann_info = self.get_ann_info(idx)
        data_info.update(ann_info)  # insere 'gt_bboxes_3d' e 'gt_labels_3d'

        # CORREÇÃO: o Pack3DDetInputs procura especificamente por
        # 'eval_ann_info' (independente da lista `keys` do test_pipeline)
        # para anexar ao data_sample e permitir avaliação. Sem isso,
        # data_sample.eval_ann_info fica None.
        if self.test_mode:
            data_info['eval_ann_info'] = ann_info

        print("DEBUG KEYS:", data_info.keys())
        print("LIDAR USED:", data_info['lidar_points']['lidar_path'])
        return self.pipeline(data_info)