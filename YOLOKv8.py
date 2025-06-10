"""
Enhanced YOLOKv8 keypoint‑detection module
------------------------------------------
This version aligns its output structure with the official Roboflow‑style
`KeypointsDetectionInferenceResponse` definition (see reference `1.py`) so
that calling the model returns the same rich, nested dataclass objects you
see in *export.txt*.

Key improvements
----------------
* **Dataclass response model** – lightweight local replicas of
  `Keypoint`, `KeypointsPrediction`, `InferenceResponseImage`, and
  `KeypointsDetectionInferenceResponse` are declared so there are **no external
  dependencies** on the Roboflow SDK.
* **Class & keypoint metadata loading** – `keypoints_metadata.json` is read at
  init‑time to obtain object‑class names and per‑class keypoint labels.
* **Automatic coordinate rescaling** – keypoint (x, y) pairs are remapped from
  model‑space (input tensor) back to the original image resolution, matching
  the bbox rescaling logic that already existed.
* **Single call interface** – `__call__(image)` (or `detect_objects(image)`)
  now returns a **list[KeypointsDetectionInferenceResponse]**, identical to the
  structure produced by `1.py`.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime

from yolov8.utils import xywh2xyxy, draw_detections, multiclass_nms

# ---------------------------------------------------------------------------
# Lightweight replicas of the Roboflow response dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class KeypointsPrediction:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    class_confidence: Optional[float] = None
    class_id: Optional[int] = None
    tracker_id: Optional[str] = None
    detection_id: Optional[str] = None
    parent_id: Optional[str] = None
    keypoints: List[Keypoint] = None


@dataclass
class InferenceResponseImage:
    width: int
    height: int


@dataclass
class KeypointsDetectionInferenceResponse:
    visualization: Optional[bytes] = None
    inference_id: Optional[str] = None
    frame_id: Optional[int] = None
    time: Optional[float] = None
    image: InferenceResponseImage = None
    predictions: List[KeypointsPrediction] = None


# ---------------------------------------------------------------------------
# Main model wrapper – API kept compatible with the original YOLOKv8 class
# ---------------------------------------------------------------------------

class YOLOKv8:
    """ONNX‑runtime inference wrapper that outputs Roboflow‑style responses."""

    def __init__(
        self,
        path: str,
        conf_thres: float = 0.7,
        iou_thres: float = 0.5,
        num_classes: Optional[int] = None,
        metadata_path: str = "keypoints_metadata.json",
    ) -> None:
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # ------------------------------------------------------------------
        # Load class & keypoint metadata (if available)
        # ------------------------------------------------------------------
        self.class_names: List[str] = []
        self.keypoints_mapping: dict[int, str] = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf‑8") as f:
                md = json.load(f)
            # One entry per object class – we support multi‑class JSON too
            self.class_names = [entry["object_class"] for entry in md]
            # Build global keypoint map (assumes identical ordering for all classes)
            # If classes have different keypoint sets, adapt as needed.
            self.keypoints_mapping = {
                int(idx): name
                for entry in md
                for idx, name in entry["keypoints"].items()
            }
        # Fallback if the JSON is missing
        if not self.class_names:
            # Either provided explictly or default sequential names
            self.class_names = [f"class_{i}" for i in range(num_classes or 1)]

        self.num_classes = num_classes or len(self.class_names)

        # ------------------------------------------------------------------
        # Initialize ONNX model session
        # ------------------------------------------------------------------
        self._initialize_model(path)

    # ----------------------------------------------------------------------
    # Public interface – calling the instance runs inference and returns a
    # list[KeypointsDetectionInferenceResponse]
    # ----------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> List[KeypointsDetectionInferenceResponse]:
        return self.detect_objects(image)

    # ------------------------------------------------------------------
    # Model setup helpers
    # ------------------------------------------------------------------

    def _initialize_model(self, path: str) -> None:
        # Prefer CUDA if available, else fall back automatically
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        self._get_io_details()

    def _get_io_details(self) -> None:
        # Input
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape  # (b, c, h, w)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        # Output
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    # ------------------------------------------------------------------
    # Inference & preprocessing
    # ------------------------------------------------------------------

    def detect_objects(self, image: np.ndarray) -> List[KeypointsDetectionInferenceResponse]:
        """Runs a complete inference‑>postprocess cycle and returns the response."""
        self.img_height, self.img_width = image.shape[:2]
        input_tensor = self._prepare_input(image)
        raw_outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return self._postprocess(raw_outputs)

    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        # BGR ➜ RGB, resize & normalize to [0,1]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC ➜ CHW
        return img[np.newaxis, :, :, :]

    # ------------------------------------------------------------------
    # Post‑processing helpers
    # ------------------------------------------------------------------

    def _postprocess(self, output: List[np.ndarray]) -> List[KeypointsDetectionInferenceResponse]:
        # ------------------------------------------------------------------
        # 1. Filter by confidence
        # ------------------------------------------------------------------
        preds = np.squeeze(output[0]).T  # (num_boxes, 5 + num_classes + num_kpts*3)
        obj_scores = np.max(preds[:, 4 : 4 + self.num_classes], axis=1)
        keep = obj_scores > self.conf_threshold
        preds = preds[keep]
        obj_scores = obj_scores[keep]
        if preds.size == 0:
            return []

        # ------------------------------------------------------------------
        # 2. Parse fields
        # ------------------------------------------------------------------
        class_ids = np.argmax(preds[:, 4 : 4 + self.num_classes], axis=1)
        keypoints_raw = preds[:, 4 + self.num_classes :]
        boxes = self._extract_boxes(preds[:, :4])

        # ------------------------------------------------------------------
        # 3. Per‑class NMS
        # ------------------------------------------------------------------
        keep_indices = multiclass_nms(boxes, obj_scores, class_ids, self.iou_threshold)
        boxes = boxes[keep_indices]
        obj_scores = obj_scores[keep_indices]
        class_ids = class_ids[keep_indices]
        keypoints_raw = keypoints_raw[keep_indices]

        # ------------------------------------------------------------------
        # 4. Keypoint coordinate rescaling
        # ------------------------------------------------------------------
        keypoints_raw = self._rescale_keypoints(keypoints_raw)

        # ------------------------------------------------------------------
        # 5. Build dataclass response (matches `1.py` output)
        # ------------------------------------------------------------------
        return self._build_response(boxes, obj_scores, class_ids, keypoints_raw)

    # ------------------------------------------------------------------
    # Utility: extract & rescale boxes
    # ------------------------------------------------------------------

    def _extract_boxes(self, xywh: np.ndarray) -> np.ndarray:
        """Converts [cx,cy,w,h] ➜ [x1,y1,x2,y2] and rescales to original image."""
        boxes = xywh2xyxy(xywh)
        scale = np.array([self.input_width, self.input_height, self.input_width, self.input_height], dtype=np.float32)
        boxes = boxes / scale  # back to 0‑1
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height], dtype=np.float32)
        return boxes

    def _rescale_keypoints(self, kpts: np.ndarray) -> np.ndarray:
        """Rescales keypoints from model‑space to original image coords."""
        if kpts.size == 0:
            return kpts
        kpts = kpts.copy().astype(np.float32)
        n_kpts = kpts.shape[1] // 3
        x_idx = np.arange(0, 3 * n_kpts, 3)
        y_idx = x_idx + 1
        # Normalize ➜ absolute pixel coords
        kpts[:, x_idx] = kpts[:, x_idx] / self.input_width * self.img_width
        kpts[:, y_idx] = kpts[:, y_idx] / self.input_height * self.img_height
        return kpts

    # ------------------------------------------------------------------
    # Response builder (Roboflow format)
    # ------------------------------------------------------------------

    def _build_response(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        keypoints: np.ndarray,
    ) -> List[KeypointsDetectionInferenceResponse]:
        """Constructs a list with a single KeypointsDetectionInferenceResponse."""
        # Create KeypointsPrediction list
        preds: List[KeypointsPrediction] = []
        num_kpts = keypoints.shape[1] // 3 if keypoints.size else 0
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            kp_flat = keypoints[i]
            kp_objs: List[Keypoint] = []
            for k in range(num_kpts):
                kp_x = float(kp_flat[k * 3])
                kp_y = float(kp_flat[k * 3 + 1])
                kp_conf = float(kp_flat[k * 3 + 2])
                kp_objs.append(
                    Keypoint(
                        x=kp_x,
                        y=kp_y,
                        confidence=kp_conf,
                        class_id=k,
                        class_name=self.keypoints_mapping.get(k, str(k)),
                    )
                )

            preds.append(
                KeypointsPrediction(
                    x=float((x1 + x2) / 2),
                    y=float((y1 + y2) / 2),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    confidence=float(scores[i]),
                    class_name=self.class_names[int(class_ids[i])],
                    class_id=int(class_ids[i]),
                    detection_id=str(uuid.uuid4()),
                    keypoints=kp_objs,
                )
            )

        response = KeypointsDetectionInferenceResponse(
            image=InferenceResponseImage(width=self.img_width, height=self.img_height),
            predictions=preds,
        )
        return [response]

    # ------------------------------------------------------------------
    # Optional: draw detections on the image (unchanged)
    # ------------------------------------------------------------------

    def draw_detections(self, image: np.ndarray, mask_alpha: float = 0.4):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)
