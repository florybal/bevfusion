import torch
import numpy as np
import mmengine
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes

@DATASETS.register_module()
class PlastipakDataset(Det3DDataset):
    METAINFO = {
        'classes': ('obstrucao', 'empilhadeira', 'carga', 'maquina',
                    'humano', 'navegavel', 'estrutura', 'portapalete'),
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
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()

    def load_data_list(self):
        data_list = mmengine.load(self.ann_file)
        if isinstance(data_list, list):
            return data_list
        elif isinstance(data_list, dict) and 'data_list' in data_list:
            return data_list['data_list']
        else:
            raise TypeError(f"Arquivo de anotação inválido: esperado lista, obtido {type(data_list)}")

    def get_data_info(self, idx):
        data_info = self.data_list[idx].copy()
        if 'lidar_path' in data_info:
            data_info['lidar_points'] = {'lidar_path': data_info['lidar_path']}
        if 'images' in data_info:
            data_info['img'] = data_info['images']
        # Define box_type_3d como a classe real
        data_info['box_type_3d'] = LiDARInstance3DBoxes
        return data_info

    def get_ann_info(self, idx):
        data_info = self.data_list[idx]
        annos = data_info.get('annos', None)

        if annos is None:
            empty_bboxes = LiDARInstance3DBoxes(
                torch.zeros((0, 8), dtype=torch.float32),
                box_dim=8,
                with_yaw=True
            )
            return {
                'gt_bboxes_3d': empty_bboxes,
                'gt_labels_3d': np.zeros((0,), dtype=np.int64),
                'gt_names': [],
            }

        bboxes = annos.get('bboxes_3d', np.zeros((0, 7), dtype=np.float32))
        labels = annos.get('labels_3d', np.zeros((0,), dtype=np.int64))

        if isinstance(bboxes, np.ndarray):
            bboxes_tensor = torch.from_numpy(bboxes).float()
        else:
            bboxes_tensor = bboxes.clone().float()

        # Se tiver 7 colunas, adiciona sin e cos
        if bboxes_tensor.shape[-1] == 7:
            # Extrai yaw
            yaw = bboxes_tensor[:, 6]
            sin_yaw = torch.sin(yaw)
            cos_yaw = torch.cos(yaw)
            # Concatena: [x, y, z, w, l, h, sin, cos]
            bboxes_tensor = torch.cat([
                bboxes_tensor[:, :6],   # x, y, z, w, l, h
                sin_yaw.unsqueeze(1),   # sin(yaw)
                cos_yaw.unsqueeze(1)    # cos(yaw)
            ], dim=1)   # agora tem 8 colunas

        bboxes_3d = LiDARInstance3DBoxes(
            bboxes_tensor,
            box_dim=8,
            with_yaw=True
        )

        names = [self.METAINFO['classes'][l] for l in labels] if len(labels) > 0 else []
        return {
            'gt_bboxes_3d': bboxes_3d,
            'gt_labels_3d': labels,
            'gt_names': names,
        }

    def prepare_data(self, idx):
        ori_input_dict = self.get_data_info(idx)
        ann_info = self.get_ann_info(idx)
        # Garantia extra: recria com 7 dimensões se necessário
        gt_bboxes = ann_info['gt_bboxes_3d']
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            tensor = gt_bboxes.tensor.clone().float()
            if tensor.shape[-1] != 7:
                if tensor.shape[-1] < 7:
                    pad = torch.zeros((tensor.shape[0], 7 - tensor.shape[-1]))
                    tensor = torch.cat([tensor, pad], dim=1)
                else:
                    tensor = tensor[:, :7]
                ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                    tensor,
                    box_dim=7,
                    with_yaw=True
                )
        elif isinstance(gt_bboxes, torch.Tensor):
            tensor = gt_bboxes.clone().float()
            if tensor.shape[-1] != 7:
                if tensor.shape[-1] < 7:
                    pad = torch.zeros((tensor.shape[0], 7 - tensor.shape[-1]))
                    tensor = torch.cat([tensor, pad], dim=1)
                else:
                    tensor = tensor[:, :7]
                ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                    tensor,
                    box_dim=7,
                    with_yaw=True
                )
        # Se for np.ndarray, já foi convertido no get_ann_info

        ori_input_dict['ann_info'] = ann_info

        if 'lidar_path' in ori_input_dict and 'lidar_points' not in ori_input_dict:
            ori_input_dict['lidar_points'] = {'lidar_path': ori_input_dict['lidar_path']}
        if 'images' in ori_input_dict and 'img' not in ori_input_dict:
            ori_input_dict['img'] = ori_input_dict['images']

        return self.pipeline(ori_input_dict)