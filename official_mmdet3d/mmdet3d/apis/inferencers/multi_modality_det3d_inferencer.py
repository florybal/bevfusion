# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import warnings
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import mmengine
import numpy as np
from mmengine.dataset import Compose
from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData

from mmdet3d.registry import INFERENCERS
from mmdet3d.utils import ConfigType
from .base_3d_inferencer import Base3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


@INFERENCERS.register_module(name='det3d-multi_modality')
@INFERENCERS.register_module()
class MultiModalityDet3DInferencer(Base3DInferencer):
    """The inferencer of multi-modality detection.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. For example, it could be
            "pointpillars_kitti-3class" or
            "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py". # noqa: E501
            If model is not specified, user must provide the
            `weights` saved by MMEngine which contains the config string.
            Defaults to None.
        weights (str, optional): Path to the checkpoint. If it is not specified
            and model is a model name of metafile, the weights will be loaded
            from metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the available
            device will be automatically used. Defaults to None.
        scope (str): The scope of registry. Defaults to 'mmdet3d'.
        palette (str): The palette of visualization. Defaults to 'none'.
    """

    def __init__(self,
                 model: Union[ModelType, str, None] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmdet3d',
                 palette: str = 'none') -> None:
        # A global counter tracking the number of frames processed, for
        # naming of the output results
        self.num_visualized_frames = 0
        super(MultiModalityDet3DInferencer, self).__init__(
            model=model,
            weights=weights,
            device=device,
            scope=scope,
            palette=palette)

    def _inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type: str = 'CAM2',
                        **kwargs) -> list:
        """Preprocess the inputs to a list.

        Supports both single‑view (cam_type != 'all') and multi‑view (cam_type == 'all').
        """
        if isinstance(inputs, dict):
            assert 'infos' in inputs, 'Missing "infos" in input dict'
            infos = inputs.pop('infos')
            info_list = mmengine.load(infos)['data_list']

            # ----- MULTI‑VIEW BRANCH -----
            if cam_type == 'all':
                img_root = inputs['img']
                pcd_root = inputs['points']
                samples = []
                for data_info in info_list:
                    images = {}
                    for cam_name, cam_info in data_info['images'].items():
                        full_path = os.path.join(img_root, cam_info['img_path'])
                        images[cam_name] = {**cam_info, 'img_path': full_path}
                    lidar_path = os.path.join(pcd_root, data_info['lidar_path'])
                    sample = {
                        'images': images,
                        'lidar_points': dict(lidar_path=lidar_path),
                        'timestamp': data_info.get('timestamp', 1),
                        'box_type_3d': self.box_type_3d,
                        'box_mode_3d': self.box_mode_3d,
                        # Adiciona 'img' para compatibilidade com visualize
                        'img': images,  # será usado pelo visualize
                    }
                    samples.append(sample)
                return samples   # <-- SAÍDA ANTES DO SINGLE‑VIEW

            # ----- SINGLE‑VIEW BRANCH -----
            # Agora, se não for multi‑view, processa normalmente
            # Se inputs for um único dicionário, converte para lista
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            # Agora inputs é uma lista, e info_list também é uma lista
            # Não fazemos assert len(info_list) == len(inputs) porque inputs é um dicionário
            # Vamos processar cada item
            for index, sample in enumerate(inputs):
                data_info = info_list[index]
                img_path_from_info = data_info['images'][cam_type]['img_path']
                if isinstance(sample['img'], str) and \
                        osp.basename(img_path_from_info) != osp.basename(sample['img']):
                    raise ValueError(
                        f'the info file of {img_path_from_info} is not provided.')
                cam2img = np.asarray(
                    data_info['images'][cam_type]['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(
                    data_info['images'][cam_type]['lidar2cam'], dtype=np.float32)
                if 'lidar2img' in data_info['images'][cam_type]:
                    lidar2img = np.asarray(
                        data_info['images'][cam_type]['lidar2img'], dtype=np.float32)
                else:
                    lidar2img = cam2img @ lidar2cam
                sample['cam2img'] = cam2img
                sample['lidar2cam'] = lidar2cam
                sample['lidar2img'] = lidar2img
            return list(inputs)

        elif isinstance(inputs, (list, tuple)):
            # Para listas, processa cada item (não usado no demo)
            raise NotImplementedError(
                'List of inputs is not fully supported. Please pass a single dict.')
        else:
            raise TypeError(f'Unsupported input type: {type(inputs)}')

    def _init_pipeline(self, cfg: ConfigType) -> Compose:
        """Initialize the test pipeline."""
        pipeline_cfg = cfg.test_dataloader.dataset.pipeline

        load_point_idx = self._get_transform_idx(pipeline_cfg, 'LoadPointsFromFile')
        load_img_idx = self._get_transform_idx(pipeline_cfg, 'LoadImageFromFile')
        load_mv_img_idx = self._get_transform_idx(pipeline_cfg, 'BEVLoadMultiViewImageFromFiles')

        # Requer um loader de pontos e pelo menos um loader de imagens
        if load_point_idx == -1 or (load_img_idx == -1 and load_mv_img_idx == -1):
            raise ValueError(
                'Both LoadPointsFromFile and (LoadImageFromFile or BEVLoadMultiViewImageFromFiles) '
                'must be specified in the pipeline, but LoadPointsFromFile is '
                f'{load_point_idx == -1} and image loader is missing')

        # Extrai parâmetros do loader de pontos para visualização
        load_cfg = pipeline_cfg[load_point_idx]
        self.coord_type = load_cfg['coord_type']
        self.load_dim = load_cfg['load_dim']
        self.use_dim = list(range(load_cfg['use_dim'])) if isinstance(
            load_cfg['use_dim'], int) else load_cfg['use_dim']

        # Se o pipeline usa LoadImageFromFile, substitui ambos os loaders pelo loader customizado
        # Se usa BEVLoadMultiViewImageFromFiles, mantém o pipeline intacto
        if load_img_idx != -1:
            load_point_args = pipeline_cfg[load_point_idx].copy()
            load_point_args.pop('type')
            load_img_args = pipeline_cfg[load_img_idx].copy()
            load_img_args.pop('type')

            idx_to_remove = sorted([load_point_idx, load_img_idx], reverse=True)
            for idx in idx_to_remove:
                pipeline_cfg.pop(idx)

            pipeline_cfg.insert(
                min(load_point_idx, load_img_idx),
                dict(type='MultiModalityDet3DInferencerLoader',
                    load_point_args=load_point_args,
                    load_img_args=load_img_args)
            )

        return Compose(pipeline_cfg)
    
    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (PredType): Predictions of the model.
            return_vis (bool): Whether to return the visualization result.
                Defaults to False.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            draw_pred (bool): Whether to draw predicted bounding boxes.
                Defaults to True.
            no_save_vis (bool): Whether to save visualization results.
            pred_score_thr (float): Minimum score of bboxes to draw.
                Defaults to 0.3.
            img_out_dir (str): Output directory of visualization results.
                If left as empty, no file will be saved. Defaults to ''.

        Returns:
            List[np.ndarray] or None: Returns visualization results only if
            applicable.
        """
        if no_save_vis is True:
            img_out_dir = ''

        if not show and img_out_dir == '' and not return_vis:
            return None

        if getattr(self, 'visualizer') is None:
            raise ValueError('Visualization needs the "visualizer" term'
                             'defined in the config, but got None.')

        results = []

        for single_input, pred in zip(inputs, preds):
            points_input = single_input['points']
            if isinstance(points_input, str):
                pts_bytes = mmengine.fileio.get(points_input)
                points = np.frombuffer(pts_bytes, dtype=np.float32)
                points = points.reshape(-1, self.load_dim)
                points = points[:, self.use_dim]
                pc_name = osp.basename(points_input).split('.bin')[0]
                pc_name = f'{pc_name}.png'
            elif isinstance(points_input, np.ndarray):
                points = points_input.copy()
                pc_num = str(self.num_visualized_frames).zfill(8)
                pc_name = f'{pc_num}.png'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(points_input)}')

            if img_out_dir != '' and show:
                o3d_save_path = osp.join(img_out_dir, 'vis_lidar', pc_name)
                mmengine.mkdir_or_exist(osp.dirname(o3d_save_path))
            else:
                o3d_save_path = None

            img_input = single_input['img']
            if isinstance(single_input['img'], str):
                img_bytes = mmengine.fileio.get(img_input)
                img = mmcv.imfrombytes(img_bytes)
                img = img[:, :, ::-1]
                img_name = osp.basename(img_input)
            elif isinstance(img_input, np.ndarray):
                img = img_input.copy()
                img_num = str(self.num_visualized_frames).zfill(8)
                img_name = f'{img_num}.jpg'
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(img_input)}')

            out_file = osp.join(img_out_dir, 'vis_camera', cam_type_dir,
                                img_name) if img_out_dir != '' else None

            data_input = dict(points=points, img=img)
            self.visualizer.add_datasample(
                pc_name,
                data_input,
                pred,
                show=show,
                wait_time=wait_time,
                draw_gt=False,
                draw_pred=draw_pred,
                pred_score_thr=pred_score_thr,
                o3d_save_path=o3d_save_path,
                out_file=out_file,
                vis_task='multi-modality_det',
            )
            results.append(points)
            self.num_visualized_frames += 1

        return results