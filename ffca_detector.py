"""Standalone FFCA-YOLO inference runtime bundled with ``tello_autonomy``."""

import sys
from pathlib import Path

import numpy as np
import torch


RUNTIME_ROOT = Path(__file__).resolve().parent / 'ffca_yolo'
DEFAULT_WEIGHTS = RUNTIME_ROOT / 'weights' / 'best.pt'
DEFAULT_DATA = RUNTIME_ROOT / 'data' / 'AITOD.yaml'

if str(RUNTIME_ROOT) not in sys.path:
    # The inherited YOLOv5 sources use top-level imports such as
    # ``from models.common`` and checkpoints are pickled with those names.
    sys.path.insert(0, str(RUNTIME_ROOT))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import LOGGER, check_img_size, non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device, smart_inference_mode, time_sync  # noqa: E402


class FFCAYoloDetector:
    """FFCA-YOLO/TS-RPST detector without any Bebop camera dependency."""

    def __init__(
        self,
        weights=DEFAULT_WEIGHTS,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        device='0',
        half=False,
        data=DEFAULT_DATA,
        max_det=1000,
        classes=None,
        agnostic_nms=False,
    ):
        self.weights = str(weights)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device_arg = device
        self.half = half
        self.data = str(data)
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms

        self.model = None
        self.stride = None
        self.names = None
        self.pt = None

    def load_model(self):
        if not Path(self.weights).is_file():
            raise FileNotFoundError(f'FFCA-YOLO weights not found: {self.weights}')
        if not Path(self.data).is_file():
            raise FileNotFoundError(f'FFCA-YOLO data config not found: {self.data}')

        device = select_device(self.device_arg)
        fp16 = bool(self.half and device.type == 'cuda')
        if self.half and not fp16:
            LOGGER.warning('WARNING: --half requested without CUDA; using FP32 inference.')

        self.model = DetectMultiBackend(
            self.weights,
            device=device,
            dnn=False,
            data=self.data,
            fp16=fp16,
        )
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)
        try:
            self.model.warmup(imgsz=(1, 3, *self.imgsz))
        except (RuntimeError, ValueError) as exc:
            if not fp16:
                raise
            LOGGER.warning(f'WARNING: FP16 warmup failed ({exc}); reloading in FP32.')
            self.model = DetectMultiBackend(
                self.weights,
                device=device,
                dnn=False,
                data=self.data,
                fp16=False,
            )
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def preprocess(self, frame):
        if self.model is None:
            raise RuntimeError('load_model() must be called before inference')
        image = letterbox(frame, self.imgsz, stride=self.stride, auto=self.pt)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.model.device)
        image = image.half() if self.model.fp16 else image.float()
        image /= 255.0
        if len(image.shape) == 3:
            image = image[None]
        return image

    @smart_inference_mode()
    def infer(self, frame):
        image = self.preprocess(frame)
        started = time_sync()
        prediction = self.model(image)
        inferred = time_sync()
        prediction = non_max_suppression(
            prediction,
            self.conf_thres,
            self.iou_thres,
            self.classes,
            self.agnostic_nms,
            max_det=self.max_det,
        )
        completed = time_sync()
        return (
            prediction[0],
            image.shape[2:],
            (inferred - started) * 1e3,
            (completed - inferred) * 1e3,
        )

    @staticmethod
    def scale_detections(raw, input_shape, frame_shape):
        scaled = raw.clone()
        scaled[:, :4] = scale_boxes(input_shape, scaled[:, :4], frame_shape).round()
        return scaled
