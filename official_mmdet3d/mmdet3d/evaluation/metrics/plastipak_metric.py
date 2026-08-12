from typing import Dict, List, Sequence
import os.path as osp
import numpy as np

import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


@METRICS.register_module()
class PlastipakMetric(KittiMetric):

    CAMERA_NAMES = {
        'camera_left_calib_link': 'fisheye_left',
        'camera_right_calib_link': 'fisheye_right',
        'camera_zed_left_calib_link': 'zed_left',
        'camera_zed_right_calib_link': 'zed_right',
    }

    DS_TO_CALIB = {v: k for k, v in CAMERA_NAMES.items()}

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

    def __init__(self,
                 ann_file=None,
                 metric='bbox',
                 pklfile_prefix=None,
                 submission_prefix=None,
                 default_cam_key='fisheye_left',
                 collect_device='cpu',
                 backend_args=None,
                 prefix=None):

        super().__init__(
            ann_file=ann_file,
            metric=metric,
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            default_cam_key=default_cam_key,
            collect_device=collect_device,
            backend_args=backend_args,
            prefix=prefix)
        self._calib_data = None
        self._calib_ts = None

    def _load_calibration(self, info=None):
        if self._calib_data is not None:
            return

        candidates = []
        if isinstance(self.ann_file, str):
            ann_dir = osp.dirname(self.ann_file)
            candidates.append(osp.join(ann_dir, 'matrizes_finais_bev.json'))

        if isinstance(info, dict):
            images = info.get('images', {})
            if isinstance(images, dict):
                for cam_info in images.values():
                    if not isinstance(cam_info, dict):
                        continue
                    img_path = cam_info.get('img_path')
                    if not isinstance(img_path, str):
                        continue
                    marker = '/camera/'
                    pos = img_path.find(marker)
                    if pos != -1:
                        root = img_path[:pos]
                        candidates.append(osp.join(root, 'matrizes_finais_bev.json'))

        calib_path = None
        for path in candidates:
            if path and osp.exists(path):
                calib_path = path
                break

        if calib_path is None:
            self._calib_data = []
            self._calib_ts = np.zeros((0,), dtype=np.float64)
            return

        self._calib_data = mmengine.load(calib_path)
        self._calib_ts = np.array(
            [float(entry['timestamp']) for entry in self._calib_data],
            dtype=np.float64)

    def _ensure_camera_fields(self, info):
        images = info.get('images')
        if not isinstance(images, dict):
            return

        self._load_calibration(info)
        timestamp = float(info.get('timestamp', 0.0))
        calib_entry = None

        if self._calib_data is not None and len(self._calib_data) > 0:
            nearest_idx = int(np.argmin(np.abs(self._calib_ts - timestamp)))
            if abs(self._calib_ts[nearest_idx] - timestamp) <= 1.2:
                calib_entry = self._calib_data[nearest_idx]

        lidar_t = None
        if calib_entry is not None:
            lidar_t = np.array(
                calib_entry['sensors']['lidar_front_link'],
                dtype=np.float32)

        for ds_name, cam_info in images.items():
            if not isinstance(cam_info, dict):
                continue

            calib_name = self.DS_TO_CALIB.get(ds_name)
            if calib_name is None:
                continue

            if 'cam2img' not in cam_info:
                cam_info['cam2img'] = self.P_MATRICES[calib_name][:, :3].tolist()

            if 'lidar2cam' not in cam_info:
                if calib_entry is not None and lidar_t is not None:
                    cam_t = np.array(
                        calib_entry['sensors'][calib_name],
                        dtype=np.float32)
                    lidar2cam = np.linalg.inv(cam_t) @ lidar_t
                    cam_info['lidar2cam'] = lidar2cam.tolist()
                else:
                    cam_info['lidar2cam'] = np.eye(4, dtype=np.float32).tolist()

    @staticmethod
    def _infer_hw_from_cam_info(cam_info):
        shape = cam_info.get('img_shape') or cam_info.get('ori_shape')
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            try:
                return int(shape[0]), int(shape[1])
            except (TypeError, ValueError):
                pass

        cam2img = np.asarray(cam_info.get('cam2img', []), dtype=np.float32)
        if cam2img.ndim == 2 and cam2img.shape[0] >= 2 and cam2img.shape[1] >= 3:
            cx = float(cam2img[0, 2])
            cy = float(cam2img[1, 2])
            if cx > 0 and cy > 0:
                return int(round(cy * 2.0)), int(round(cx * 2.0))

        return 720, 1280

    def _ensure_image_hw(self, info):
        images = info.get('images')
        if not isinstance(images, dict):
            return

        for cam_name, cam_info in images.items():
            if not isinstance(cam_info, dict):
                continue
            if 'height' in cam_info and 'width' in cam_info:
                continue

            height, width = self._infer_hw_from_cam_info(cam_info)
            cam_info.setdefault('height', int(height))
            cam_info.setdefault('width', int(width))

    def process(self, data_batch, data_samples):

        for sample in data_samples:

            if 'sample_idx' in sample:
                sample_idx = int(sample['sample_idx'])
            else:
                sample_idx = int(sample.metainfo.get('sample_idx', 0))

            pred_3d = sample['pred_instances_3d']

            if isinstance(pred_3d, dict):
                bboxes_3d = pred_3d['bboxes_3d']
                scores_3d = pred_3d['scores_3d']
                labels_3d = pred_3d['labels_3d']
            else:
                bboxes_3d = pred_3d.bboxes_3d
                scores_3d = pred_3d.scores_3d
                labels_3d = pred_3d.labels_3d

            result = dict()

            result['pred_instances_3d'] = dict(
                bboxes_3d=bboxes_3d.cpu(),
                scores_3d=scores_3d.cpu(),
                labels_3d=labels_3d.cpu())
            result['sample_idx'] = sample_idx

            ############################################################
            # Ground Truth
            ############################################################

            if 'eval_ann_info' in sample:

                result['eval_ann_info'] = sample['eval_ann_info']

            else:

                gt = sample.gt_instances_3d

                result['eval_ann_info'] = dict(
                    gt_bboxes_3d=gt.bboxes_3d.cpu(),
                    gt_labels_3d=gt.labels_3d.cpu())

            self.results.append(result)

    def compute_metrics(self, results):

        pkl_infos = mmengine.load(self.ann_file, backend_args=self.backend_args)
        if isinstance(pkl_infos, list):
            pkl_infos = {'data_list': pkl_infos}

        if 'metainfo' not in pkl_infos:
            classes = self.dataset_meta['classes']
            pkl_infos['metainfo'] = {
                'categories': {name: idx for idx, name in enumerate(classes)}
            }

        for item_idx, item in enumerate(pkl_infos.get('data_list', [])):
            item.setdefault('sample_idx', item_idx)
            self._ensure_camera_fields(item)

            if 'instances' in item:
                self._ensure_image_hw(item)
                continue

            annos = item.get('annos', {})
            boxes_3d = np.asarray(
                annos.get('gt_bboxes_3d', np.zeros((0, 7), dtype=np.float32)))
            labels_3d = np.asarray(
                annos.get('gt_labels_3d', np.zeros((0,), dtype=np.int64)))

            instances = []
            for box_3d, label in zip(boxes_3d, labels_3d):
                instances.append(
                    dict(
                        bbox_3d=np.asarray(box_3d, dtype=np.float32),
                        bbox_label=int(label),
                        bbox_label_3d=int(label),
                        bbox=np.zeros(4, dtype=np.float32),
                        truncated=0.0,
                        occluded=0,
                        alpha=0.0,
                        score=1.0,
                    ))
            item['instances'] = instances
            self._ensure_image_hw(item)

        self.classes = self.dataset_meta['classes']
        self.data_infos = self.convert_annos_to_kitti_annos(pkl_infos)
        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            return metric_dict

        gt_annos = [
            self.data_infos[result['sample_idx']]['kitti_annos']
            for result in results
        ]

        for metric in self.metrics:
            metric_dict.update(
                self.kitti_evaluate(
                    result_dict,
                    gt_annos,
                    metric=metric,
                    classes=self.classes,
                    logger=None))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return metric_dict