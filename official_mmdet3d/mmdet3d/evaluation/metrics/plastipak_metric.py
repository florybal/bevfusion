# ==============================
#  Plastipak 3D Detection Metric
# ==============================

from typing import Dict, List, Optional
import os.path as osp

import numpy as np
import torch
import mmengine

from mmengine.logging import MMLogger, print_log
from mmengine.registry import METRICS

from mmdet3d.evaluation.metrics.kitti_metric import KittiMetric


@METRICS.register_module()
class PlastipakMetric(KittiMetric):
    """
    Metric for Plastipak 3D detection.

    Based on MMDetection3D v1.4 KittiMetric, but does NOT use
    mmdet3d.evaluation.functional.kitti_utils.eval.kitti_eval().

    The KITTI infrastructure is still used for:
        - ann_file
        - loading dataset information
        - result formatting
        - MMEngine metric integration

    Evaluation itself is performed directly on the 3D bounding boxes
    using class-wise IoU matching.

    Classes:
        0 - obstrucao
        1 - empilhadeira
        2 - carga
        3 - maquina
        4 - humano
        5 - navegavel
        6 - estrutura
        7 - portapalete
    """

    CAMERA_NAMES = {
        'camera_left_calib_link': 'fisheye_left',
        'camera_right_calib_link': 'fisheye_right',
        'camera_zed_left_calib_link': 'zed_left',
        'camera_zed_right_calib_link': 'zed_right',
    }

    DS_TO_CALIB = {
        v: k for k, v in CAMERA_NAMES.items()
    }

    P_MATRICES = {
        'camera_left_calib_link': np.array(
            [[500.391, 0, 343.318, 0],
             [0, 545.179, 280.577, 0],
             [0, 0, 1, 0]],
            dtype=np.float32),

        'camera_right_calib_link': np.array(
            [[492.441, 0, 645.295, 0],
             [0, 491.652, 348.315, 0],
             [0, 0, 1, 0]],
            dtype=np.float32),

        'camera_zed_left_calib_link': np.array(
            [[534.755, 0, 644.505, 0],
             [0, 534.79, 347.236, 0],
             [0, 0, 1, 0]],
            dtype=np.float32),

        'camera_zed_right_calib_link': np.array(
            [[534.25, 0, 638.765, 0],
             [0, 534.285, 338.202, 0],
             [0, 0, 1, 0]],
            dtype=np.float32),
    }

    # ==============================================================
    # Plastipak classes
    # ==============================================================

    PLASTIPAK_CLASSES = [
        'obstrucao',
        'empilhadeira',
        'carga',
        'maquina',
        'humano',
        'navegavel',
        'estrutura',
        'portapalete',
    ]

    # Default IoU thresholds.
    #
    # 0.50 is a reasonable starting point for a custom industrial
    # 3D dataset.
    #
    # You can later make this class-dependent.
    DEFAULT_IOU_THRESHOLDS = {
        'obstrucao': 0.01,
        'empilhadeira': 0.01,
        'carga': 0.01,
        'maquina': 0.01,
        'humano': 0.01,
        'navegavel': 0.30,
        'estrutura': 0.50,
        'portapalete': 0.10,
    }

    def __init__(
        self,
        ann_file=None,
        metric='bbox',
        pklfile_prefix=None,
        submission_prefix=None,
        default_cam_key='fisheye_left',
        collect_device='cpu',
        backend_args=None,
        prefix=None,
        iou_thr=0.10,
        score_thr=0.0,
        use_bev=False,
    ):

        super().__init__(
            ann_file=ann_file,
            metric=metric,
            pklfile_prefix=pklfile_prefix,
            submission_prefix=submission_prefix,
            default_cam_key=default_cam_key,
            collect_device=collect_device,
            backend_args=backend_args,
            prefix=prefix,
        )

        self._calib_data = None
        self._calib_ts = None

        self.iou_thr = float(iou_thr)
        self.score_thr = float(score_thr)

        # False = full 3D IoU
        # True  = BEV IoU
        self.use_bev = bool(use_bev)

    # ==============================================================
    # Calibration
    # ==============================================================

    def _load_calibration(self, info=None):

        if self._calib_data is not None:
            return

        candidates = []

        if isinstance(self.ann_file, str):
            ann_dir = osp.dirname(self.ann_file)
            candidates.append(
                osp.join(
                    ann_dir,
                    'matrizes_finais_bev.json'))

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

                        candidates.append(
                            osp.join(
                                root,
                                'matrizes_finais_bev.json'))

        calib_path = None

        for path in candidates:

            if path and osp.exists(path):
                calib_path = path
                break

        if calib_path is None:

            self._calib_data = []
            self._calib_ts = np.zeros(
                (0,),
                dtype=np.float64)

            return

        self._calib_data = mmengine.load(
            calib_path)

        self._calib_ts = np.array(
            [
                float(entry['timestamp'])
                for entry in self._calib_data
            ],
            dtype=np.float64)

    # ==============================================================

    def _ensure_camera_fields(self, info):

        images = info.get('images')

        if not isinstance(images, dict):
            return

        self._load_calibration(info)

        timestamp = float(
            info.get('timestamp', 0.0))

        calib_entry = None

        if (
            self._calib_data is not None
            and len(self._calib_data) > 0
        ):

            nearest_idx = int(
                np.argmin(
                    np.abs(
                        self._calib_ts - timestamp)))

            if (
                abs(
                    self._calib_ts[nearest_idx]
                    - timestamp
                ) <= 1.2
            ):

                calib_entry = \
                    self._calib_data[nearest_idx]

        lidar_t = None

        if calib_entry is not None:

            lidar_t = np.array(
                calib_entry['sensors'][
                    'lidar_front_link'],
                dtype=np.float32)

        for ds_name, cam_info in images.items():

            if not isinstance(cam_info, dict):
                continue

            calib_name = self.DS_TO_CALIB.get(
                ds_name)

            if calib_name is None:
                continue

            if 'cam2img' not in cam_info:

                cam_info['cam2img'] = (
                    self.P_MATRICES[
                        calib_name
                    ][:, :3].tolist()
                )

            if 'lidar2cam' not in cam_info:

                if (
                    calib_entry is not None
                    and lidar_t is not None
                ):

                    cam_t = np.array(
                        calib_entry['sensors'][
                            calib_name],
                        dtype=np.float32)

                    lidar2cam = (
                        np.linalg.inv(cam_t)
                        @ lidar_t
                    )

                    cam_info[
                        'lidar2cam'
                    ] = lidar2cam.tolist()

                else:

                    cam_info[
                        'lidar2cam'
                    ] = np.eye(
                        4,
                        dtype=np.float32
                    ).tolist()

    # ==============================================================

    @staticmethod
    def _infer_hw_from_cam_info(
        cam_info
    ):

        shape = (
            cam_info.get('img_shape')
            or cam_info.get('ori_shape')
        )

        if (
            isinstance(shape, (list, tuple))
            and len(shape) >= 2
        ):

            try:
                return (
                    int(shape[0]),
                    int(shape[1])
                )

            except (
                TypeError,
                ValueError
            ):
                pass

        cam2img = np.asarray(
            cam_info.get(
                'cam2img',
                []
            ),
            dtype=np.float32
        )

        if (
            cam2img.ndim == 2
            and cam2img.shape[0] >= 2
            and cam2img.shape[1] >= 3
        ):

            cx = float(
                cam2img[0, 2])

            cy = float(
                cam2img[1, 2])

            if cx > 0 and cy > 0:

                return (
                    int(round(cy * 2.0)),
                    int(round(cx * 2.0))
                )

        return 720, 1280

    # ==============================================================

    def _ensure_image_hw(self, info):

        images = info.get('images')

        if not isinstance(images, dict):
            return

        for cam_name, cam_info in images.items():

            if not isinstance(cam_info, dict):
                continue

            if (
                'height' in cam_info
                and 'width' in cam_info
            ):
                continue

            height, width = (
                self._infer_hw_from_cam_info(
                    cam_info)
            )

            cam_info.setdefault(
                'height',
                int(height))

            cam_info.setdefault(
                'width',
                int(width))

    # ==============================================================
    # PROCESS
    # ==============================================================

    def process(
        self,
        data_batch,
        data_samples
    ):

        for sample in data_samples:

            if 'sample_idx' in sample:

                sample_idx = int(
                    sample['sample_idx'])

            else:

                sample_idx = int(
                    sample.metainfo.get(
                        'sample_idx',
                        0))

            pred_3d = sample[
                'pred_instances_3d']

            if isinstance(pred_3d, dict):

                bboxes_3d = \
                    pred_3d['bboxes_3d']

                scores_3d = \
                    pred_3d['scores_3d']

                labels_3d = \
                    pred_3d['labels_3d']

            else:

                bboxes_3d = \
                    pred_3d.bboxes_3d

                scores_3d = \
                    pred_3d.scores_3d

                labels_3d = \
                    pred_3d.labels_3d

            result = dict()

            result['pred_instances_3d'] = dict(
                bboxes_3d=bboxes_3d.cpu(),
                scores_3d=scores_3d.cpu(),
                labels_3d=labels_3d.cpu(),
            )

            result['sample_idx'] = sample_idx

            # ------------------------------------------------------
            # GT
            # ------------------------------------------------------

            if 'eval_ann_info' in sample:

                result['eval_ann_info'] = \
                    sample['eval_ann_info']

            else:

                gt = sample.gt_instances_3d

                result['eval_ann_info'] = dict(
                    gt_bboxes_3d=
                        gt.bboxes_3d.cpu(),

                    gt_labels_3d=
                        gt.labels_3d.cpu(),
                )

            self.results.append(result)
    
        gt_boxes = result['eval_ann_info']['gt_bboxes_3d']
        pred_boxes = result['pred_instances_3d']['bboxes_3d']

    # ==============================================================
    # BOX UTILITIES
    # ==============================================================

    @staticmethod
    def _box_tensor(boxes):

        if boxes is None:
            return None

        if hasattr(boxes, 'tensor'):

            tensor = boxes.tensor

        elif torch.is_tensor(boxes):

            tensor = boxes

        else:

            tensor = torch.as_tensor(
                boxes)

        return tensor.detach().cpu().float()

    # ==============================================================

    @staticmethod
    def _box_dimensions(boxes):

        tensor = PlastipakMetric._box_tensor(
            boxes)

        if tensor is None:
            return None

        # Standard MMDetection3D format:
        #
        # x, y, z, dx, dy, dz, yaw
        #
        # If velocity exists:
        #
        # x, y, z, dx, dy, dz, yaw, vx, vy
        #
        return tensor[:, :7]

    # ==============================================================
    # AXIS ALIGNED 3D IOU
    # ==============================================================

    @staticmethod
    def _axis_aligned_iou_3d(
        boxes_a,
        boxes_b
    ):
        """
        Computes 3D IoU using axis-aligned boxes.

        This is intentionally implemented without
        kitti_eval(), so it is independent of KITTI
        class names.

        Boxes:
            x, y, z, dx, dy, dz, yaw

        The yaw is ignored for this first evaluator.

        This is useful as a robust baseline.

        """

        if (
            boxes_a.shape[0] == 0
            or boxes_b.shape[0] == 0
        ):

            return torch.zeros(
                (
                    boxes_a.shape[0],
                    boxes_b.shape[0]
                ),
                dtype=torch.float32
            )

        a = boxes_a[:, :7]
        b = boxes_b[:, :7]

        a_min = a[:, :3] - a[:, 3:6] / 2
        a_max = a[:, :3] + a[:, 3:6] / 2

        b_min = b[:, :3] - b[:, 3:6] / 2
        b_max = b[:, :3] + b[:, 3:6] / 2

        inter_min = torch.maximum(
            a_min[:, None, :],
            b_min[None, :, :]
        )

        inter_max = torch.minimum(
            a_max[:, None, :],
            b_max[None, :, :]
        )

        inter_dim = torch.clamp(
            inter_max - inter_min,
            min=0
        )

        inter_volume = (
            inter_dim[..., 0]
            * inter_dim[..., 1]
            * inter_dim[..., 2]
        )

        volume_a = (
            torch.clamp(a[:, 3], min=0)
            * torch.clamp(a[:, 4], min=0)
            * torch.clamp(a[:, 5], min=0)
        )

        volume_b = (
            torch.clamp(b[:, 3], min=0)
            * torch.clamp(b[:, 4], min=0)
            * torch.clamp(b[:, 5], min=0)
        )

        union = (
            volume_a[:, None]
            + volume_b[None, :]
            - inter_volume
        )

        iou = torch.zeros_like(
            inter_volume)

        valid = union > 0

        iou[valid] = (
            inter_volume[valid]
            / union[valid]
        )

        return iou

    # ==============================================================
    # BEV IOU
    # ==============================================================

    @staticmethod
    def _bev_iou(
        boxes_a,
        boxes_b
    ):
        """
        BEV IoU.

        Uses axis-aligned rectangles in XY.

        dx and dy are used as width/length.
        Yaw is intentionally ignored here.
        """

        if (
            boxes_a.shape[0] == 0
            or boxes_b.shape[0] == 0
        ):

            return torch.zeros(
                (
                    boxes_a.shape[0],
                    boxes_b.shape[0]
                ),
                dtype=torch.float32
            )

        a = boxes_a[:, :7]
        b = boxes_b[:, :7]

        a_min = (
            a[:, :2]
            - a[:, 3:5] / 2
        )

        a_max = (
            a[:, :2]
            + a[:, 3:5] / 2
        )

        b_min = (
            b[:, :2]
            - b[:, 3:5] / 2
        )

        b_max = (
            b[:, :2]
            + b[:, 3:5] / 2
        )

        inter_min = torch.maximum(
            a_min[:, None, :],
            b_min[None, :, :]
        )

        inter_max = torch.minimum(
            a_max[:, None, :],
            b_max[None, :, :]
        )

        inter_dim = torch.clamp(
            inter_max - inter_min,
            min=0
        )

        inter_area = (
            inter_dim[..., 0]
            * inter_dim[..., 1]
        )

        area_a = (
            torch.clamp(a[:, 3], min=0)
            * torch.clamp(a[:, 4], min=0)
        )

        area_b = (
            torch.clamp(b[:, 3], min=0)
            * torch.clamp(b[:, 4], min=0)
        )

        union = (
            area_a[:, None]
            + area_b[None, :]
            - inter_area
        )

        iou = torch.zeros_like(
            inter_area)

        valid = union > 0

        iou[valid] = (
            inter_area[valid]
            / union[valid]
        )

        return iou

    # ==============================================================
    # MATCHING
    # ==============================================================

    @staticmethod
    def _match_predictions(
        gt_boxes,
        gt_labels,
        pred_boxes,
        pred_labels,
        pred_scores,
        class_id,
        iou_thr,
    ):
        """
        Greedy one-to-one matching.

        Predictions are processed in descending score order.

        Returns:
            tp
            fp
            fn
            matched_ious
        """

        gt_mask = (
            gt_labels == class_id)

        pred_mask = (
            pred_labels == class_id)

        gt = gt_boxes[
            gt_mask]

        pred = pred_boxes[
            pred_mask]

        scores = pred_scores[
            pred_mask]

        if gt.shape[0] == 0:

            return (
                0,
                int(pred.shape[0]),
                0,
                []
            )

        if pred.shape[0] == 0:

            return (
                0,
                0,
                int(gt.shape[0]),
                []
            )

        order = torch.argsort(
            scores,
            descending=True
        )

        pred = pred[order]

        scores = scores[order]

        if pred.numel() == 0:

            return (
                0,
                0,
                int(gt.shape[0]),
                []
            )

        ious = PlastipakMetric._axis_aligned_iou_3d(
            gt,
            pred
        )

        matched_gt = set()

        tp = 0
        fp = 0
        matched_ious = []

        for pred_idx in range(
            pred.shape[0]
        ):

            candidate_ious = \
                ious[:, pred_idx].clone()

            # Remove already matched GTs.
            for gt_idx in matched_gt:
                candidate_ious[
                    gt_idx
                ] = -1

            best_iou, best_gt = torch.max(
                candidate_ious,
                dim=0
            )

            best_iou = float(
                best_iou.item())

            best_gt = int(
                best_gt.item())

            if best_iou >= iou_thr:

                matched_gt.add(
                    best_gt)

                tp += 1

                matched_ious.append(
                    best_iou)

            else:

                fp += 1

        fn = (
            int(gt.shape[0])
            - tp
        )

        return (
            tp,
            fp,
            fn,
            matched_ious
        )

    # ==============================================================
    # AP
    # ==============================================================

    @staticmethod
    def _compute_ap(
        scores,
        tp_flags,
        fp_flags,
        num_gt
    ):
        """
        Compute AP using the all-points
        precision-recall integral.

        This is intentionally independent
        of KITTI's AP implementation.
        """

        if num_gt <= 0:
            return 0.0

        if len(scores) == 0:
            return 0.0

        order = np.argsort(
            -np.asarray(scores))

        tp = np.asarray(
            tp_flags,
            dtype=np.float64
        )[order]

        fp = np.asarray(
            fp_flags,
            dtype=np.float64
        )[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = (
            tp_cum
            / max(float(num_gt), 1.0)
        )

        precision = (
            tp_cum
            / np.maximum(
                tp_cum + fp_cum,
                1e-12
            )
        )

        # Precision envelope.
        mrec = np.concatenate(
            ([0.0], recall, [1.0])
        )

        mpre = np.concatenate(
            ([0.0], precision, [0.0])
        )

        for i in range(
            len(mpre) - 2,
            -1,
            -1
        ):

            mpre[i] = max(
                mpre[i],
                mpre[i + 1]
            )

        idx = np.where(
            mrec[1:] != mrec[:-1]
        )[0]

        ap = np.sum(
            (
                mrec[idx + 1]
                - mrec[idx]
            )
            * mpre[idx + 1]
        )

        return float(ap)

    # ==============================================================
    # CLASS EVALUATION
    # ==============================================================

    def _evaluate_class(
        self,
        results,
        class_id,
        iou_thr
    ):
        """
        Evaluate one class across all samples.

        Important:
        Matching is done independently per sample.
        """

        class_name = \
            self.PLASTIPAK_CLASSES[
                class_id
            ]

        all_scores = []
        all_tp = []
        all_fp = []

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        matched_ious = []

        for result in results:

            pred = result[
                'pred_instances_3d']

            pred_boxes = self._box_dimensions(
                pred['bboxes_3d'])

            pred_scores = (
                pred['scores_3d']
                .detach()
                .cpu()
                .float()
            )

            pred_labels = (
                pred['labels_3d']
                .detach()
                .cpu()
                .long()
            )

            gt_info = result[
                'eval_ann_info']

            gt_boxes = self._box_dimensions(
                gt_info['gt_bboxes_3d'])

            gt_labels = (
                gt_info['gt_labels_3d']
                .detach()
                .cpu()
                .long()
            )

            if pred_boxes is None:
                continue

            if gt_boxes is None:
                continue

            # ------------------------------------------------------
            # Filter score
            # ------------------------------------------------------

            score_mask = (
                pred_scores
                >= self.score_thr
            )

            pred_boxes = \
                pred_boxes[score_mask]

            pred_scores = \
                pred_scores[score_mask]

            pred_labels = \
                pred_labels[score_mask]

            # ------------------------------------------------------
            # Class filtering
            # ------------------------------------------------------

            gt_mask = (
                gt_labels == class_id)

            pred_mask = (
                pred_labels == class_id)

            gt_cls = \
                gt_boxes[gt_mask]

            pred_cls = \
                pred_boxes[pred_mask]

            pred_scores_cls = \
                pred_scores[pred_mask]

            total_gt += \
                int(gt_cls.shape[0])

            # ------------------------------------------------------
            # No GT
            # ------------------------------------------------------

            if gt_cls.shape[0] == 0:

                for score in pred_scores_cls:

                    all_scores.append(
                        float(score))

                    all_tp.append(0)
                    all_fp.append(1)

                total_fp += \
                    int(pred_cls.shape[0])

                continue

            # ------------------------------------------------------
            # No prediction
            # ------------------------------------------------------

            if pred_cls.shape[0] == 0:

                total_fn += \
                    int(gt_cls.shape[0])

                continue

            # ------------------------------------------------------
            # IoU
            # ------------------------------------------------------

            ious = self._axis_aligned_iou_3d(
                gt_cls,
                pred_cls
            )

            order = torch.argsort(
                pred_scores_cls,
                descending=True
            )

            matched_gt = set()

            for pred_idx in order.tolist():

                score = float(
                    pred_scores_cls[
                        pred_idx
                    ])

                candidate = \
                    ious[:, pred_idx].clone()

                for gt_idx in matched_gt:

                    candidate[
                        gt_idx
                    ] = -1

                best_iou, best_gt = \
                    torch.max(
                        candidate,
                        dim=0
                    )

                best_iou = float(
                    best_iou)

                best_gt = int(
                    best_gt)

                all_scores.append(
                    score)

                if best_iou >= iou_thr:

                    matched_gt.add(
                        best_gt)

                    all_tp.append(1)
                    all_fp.append(0)

                    total_tp += 1

                    matched_ious.append(
                        best_iou)

                else:

                    all_tp.append(0)
                    all_fp.append(1)

                    total_fp += 1

            sample_fn = (
                gt_cls.shape[0]
                - len(matched_gt)
            )

            total_fn += \
                int(sample_fn)

        # ----------------------------------------------------------
        # Metrics
        # ----------------------------------------------------------

        precision = (
            total_tp
            / max(
                total_tp + total_fp,
                1
            )
        )

        recall = (
            total_tp
            / max(
                total_tp + total_fn,
                1
            )
        )

        ap = self._compute_ap(
            all_scores,
            all_tp,
            all_fp,
            total_gt
        )

        mean_iou = (
            float(np.mean(
                matched_ious))
            if matched_ious
            else 0.0
        )

        return dict(
            class_name=class_name,
            ap=ap,
            precision=float(
                precision),
            recall=float(
                recall),
            tp=int(total_tp),
            fp=int(total_fp),
            fn=int(total_fn),
            num_gt=int(total_gt),
            num_predictions=int(
                total_tp + total_fp),
            matched_iou=mean_iou,
        )

    # ==============================================================
    # PLASTIPAK EVALUATION
    # ==============================================================

    def plastipak_evaluate(
        self,
        results,
        logger=None
    ):
        """
        Evaluate all Plastipak classes.
        """

        metric_dict = {}

        aps = []

        print_log('',logger=logger)
        print_log('======================================',logger=logger)
        print_log('Plastipak 3D Detection Evaluation',logger=logger)
        print_log('======================================',logger=logger)
        print_log(f'IoU threshold: {self.iou_thr:.2f}',logger=logger)
        print_log(f'Score threshold: {self.score_thr:.2f}',logger=logger)
        print_log('',logger=logger)

        for class_id, class_name in enumerate(self.PLASTIPAK_CLASSES):

            result = self._evaluate_class(
                results,
                class_id,
                self.iou_thr
            )

            ap = result['ap']
            aps.append(ap)

            prefix = (f'{class_name}')

            metric_dict[f'{prefix}/AP'] = float(ap)
            metric_dict[f'{prefix}/precision'] = float(result['precision'])
            metric_dict[f'{prefix}/recall'] = float(result['recall'])
            metric_dict[f'{prefix}/TP'] = float(result['tp'])
            metric_dict[f'{prefix}/FP'] = float(result['fp'])
            metric_dict[f'{prefix}/FN'] = float(result['fn'])
            metric_dict[f'{prefix}/GT'] = float(result['num_gt'])
            metric_dict[f'{prefix}/predictions'] = float(result['num_predictions'])

            metric_dict[f'{prefix}/matched_IoU'] = float(result['matched_iou'])

            print_log(
                (
                    f'{class_name:15s} '
                    f'AP={ap:.4f} '
                    f'P={result["precision"]:.4f} '
                    f'R={result["recall"]:.4f} '
                    f'TP={result["tp"]} '
                    f'FP={result["fp"]} '
                    f'FN={result["fn"]} '
                    f'GT={result["num_gt"]} '
                    f'IoU={result["matched_iou"]:.4f}'
                ),
                logger=logger
            )

        # ----------------------------------------------------------
        # mAP
        # ----------------------------------------------------------

        valid_aps = [
            x for x in aps
            if np.isfinite(x)
        ]

        mAP = (
            float(np.mean(valid_aps))
            if valid_aps
            else 0.0
        )

        metric_dict['mAP'] = mAP

        print_log('',logger=logger)
        print_log(f'mAP = {mAP:.4f}',logger=logger)
        print_log('======================================',logger=logger)

        return metric_dict

    # ==============================================================
    # COMPUTE METRICS
    # ==============================================================

    def compute_metrics(
        self,
        results
    ):

        if not results:

            return {}

        # ----------------------------------------------------------
        # Load annotation file
        # ----------------------------------------------------------

        pkl_infos = mmengine.load(
            self.ann_file,
            backend_args=self.backend_args
        )

        if isinstance(
            pkl_infos,
            list
        ):

            pkl_infos = {
                'data_list': pkl_infos
            }

        # ----------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------

        if 'metainfo' not in pkl_infos:

            pkl_infos['metainfo'] = {
                'categories': {
                    name: idx
                    for idx, name in enumerate(
                        self.PLASTIPAK_CLASSES)
                }
            }

        # ----------------------------------------------------------
        # Prepare dataset information
        # ----------------------------------------------------------

        for item_idx, item in enumerate(
            pkl_infos.get(
                'data_list',
                [])
        ):

            item.setdefault(
                'sample_idx',
                item_idx
            )

            self._ensure_camera_fields(
                item
            )

            # ------------------------------------------------------
            # Existing instances
            # ------------------------------------------------------

            if 'instances' in item:

                self._ensure_image_hw(item)

                continue

            # ------------------------------------------------------
            # Legacy annos
            # ------------------------------------------------------

            annos = item.get('annos',{})

            boxes_3d = np.asarray(
                annos.get(
                    'gt_bboxes_3d',
                    np.zeros(
                        (0, 7),
                        dtype=np.float32
                    )
                ),
                dtype=np.float32
            )

            labels_3d = np.asarray(
                annos.get(
                    'gt_labels_3d',
                    np.zeros(
                        (0,),
                        dtype=np.int64
                    )
                ),
                dtype=np.int64
            )

            instances = []

            for box_3d, label in zip(
                boxes_3d,
                labels_3d
            ):

                instances.append(
                    dict(
                        bbox_3d=np.asarray(
                            box_3d,
                            dtype=np.float32
                        ),

                        bbox_label=int(
                            label),

                        bbox_label_3d=int(
                            label),

                        bbox=np.zeros(
                            4,
                            dtype=np.float32
                        ),

                        truncated=0.0,
                        occluded=0,
                        alpha=0.0,
                        score=1.0,
                    )
                )

            item['instances'] = instances

            self._ensure_image_hw(item)

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # We DO NOT call:
        #
        #     self.kitti_evaluate()
        #
        # because that eventually calls:
        #
        #     kitti_eval()
        #
        # which only knows KITTI classes.
        # ----------------------------------------------------------

        metric_dict = self.plastipak_evaluate(results,logger=None)

        return metric_dict