from __future__ import annotations

"""TFLite-based YOLOv8 experiment.

This variant avoids heavy PyTorch/Ultralytics dependencies by running an
integer-quantised YOLOv8 model with the lightweight *tflite-runtime* backend.
It follows the same public interface as :pyclass:`experiments.impl.yolo_detect.YoloDetect`
so the higher-level scripts can switch runtimes with only the *model_path*
argument.
"""

from pathlib import Path
from typing import Tuple, List
import os
import time

import cv2  # type: ignore
import numpy as np  # type: ignore
import picamera2  # type: ignore
import json

from experiments.core.experiment import BaseExperiment


# ---------------------------------------------------------------------------
# Helper – lightweight TFLite wrapper extracted from the user-supplied script
# ---------------------------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:  # pragma: no cover – fallback only
    try:
        import tensorflow.lite as tflite  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Could not import tflite_runtime or tensorflow.lite") from exc


class YOLOTFLite:  # noqa: D101 – user-provided class, kept verbatim
    """Minimal YOLOv8 TFLite wrapper (int8 or float32 models supported).

    Parameters
    ----------
    model_path
        Path to the `.tflite` model file (integer-quantised recommended on Pi).
    conf_thres
        Confidence threshold – predictions with class score below this are
        discarded *before* non-maximum suppression (NMS).
    iou_thres
        IoU threshold for NMS (suppress overlapping detections).
    """

    def __init__(self, model_path: os.PathLike | str, labelmap_path: str, conf_thres: float = 0.5, iou_thres: float = 0.5):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Load interpreter + allocate tensors
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        # Cache tensor meta-data – assuming single-input single-output model
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]["shape"]  # e.g. [1, 320, 320, 3]
        self.input_h, self.input_w = int(self.input_shape[1]), int(self.input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]

        # Output dequantization params (needed for int8-quantised models)
        out_quant = self.output_details[0].get("quantization_parameters", {})
        self.out_scales = out_quant.get("scales", np.array([1.0]))
        self.out_zero_points = out_quant.get("zero_points", np.array([0]))
        self.out_dtype = self.output_details[0]["dtype"]
        # Per-axis quantization: output shape is (84, N), axis 0 = 84 channels
        self.per_channel = len(self.out_scales) > 1

        print(f"Input shape: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"Output dtype: {self.out_dtype}, per_channel: {self.per_channel}")
        print(f"  scales ({len(self.out_scales)}): {self.out_scales[:6]} ...")
        print(f"  zero_points ({len(self.out_zero_points)}): {self.out_zero_points[:6]} ...")

        self.labelmap = self.load_labelmap(labelmap_path)
        print(f"Labelmap: {self.labelmap}")

    def load_labelmap(self, labelmap_path: str = "models/labelmap.json") -> dict[int, str]:
        if not os.path.exists(labelmap_path):
            raise FileNotFoundError(f"Labelmap file not found: {labelmap_path}")
        with open(labelmap_path, "r") as f:
            labelmap = json.load(f)
            labelmap = labelmap.get("class", {})
        return {int(k): v for k, v in labelmap.items()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> List[dict[str, float | list[int]]]:
        """Run a forward pass on *image* and return filtered detections."""
        # Pre-process -------------------------------------------------------
        # Stretch-resize to model input dims (no padding, simpler post-proc)
        img_h, img_w = image.shape[:2]
        resized = cv2.resize(image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)

        if self.input_dtype == np.float32:
            input_data = (resized / 255.0).astype(np.float32)
        else:
            input_data = resized.astype(self.input_dtype)
        input_data = np.expand_dims(input_data, axis=0)


        # Inference ---------------------------------------------------------
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Post-process ------------------------------------------------------
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]  # shape (84, N)

        # Dequantize if the output tensor is integer-quantised
        if self.out_dtype != np.float32:
            output = output.astype(np.float32)
            if self.per_channel:
                # Per-axis: scales/zp have shape (84,) → reshape to (84, 1)
                output = (output - self.out_zero_points.reshape(-1, 1)) * self.out_scales.reshape(-1, 1)
            else:
                output = (output - self.out_zero_points[0]) * self.out_scales[0]

        output = output.T  # → (N, 84)

        # Box coords are normalised [0,1]; scale to input pixel space
        output[:, 0] *= self.input_w   # cx
        output[:, 1] *= self.input_h   # cy
        output[:, 2] *= self.input_w   # w
        output[:, 3] *= self.input_h   # h

        boxes, scores, class_ids = [], [], []
        max_scores = np.max(output[:, 4:], axis=1)
        valid = max_scores > self.conf_thres
        if not np.any(valid):
            return []

        # Scale factors to map model coords back to original image
        sx = img_w / self.input_w
        sy = img_h / self.input_h

        for det, score in zip(output[valid], max_scores[valid]):
            cx, cy, w, h = det[:4]
            x1 = int((cx - w / 2) * sx)
            y1 = int((cy - h / 2) * sy)
            w_orig = int(w * sx)
            h_orig = int(h * sy)
            boxes.append([x1, y1, w_orig, h_orig])
            scores.append(float(score))
            class_ids.append(int(np.argmax(det[4:])))

        # NMS ---------------------------------------------------------------
        idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        results: list[dict[str, float | list[int]]] = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                results.append({"box": boxes[i], "conf": scores[i], "class_id": class_ids[i]})
        return results


# ---------------------------------------------------------------------------
# Experiment implementation
# ---------------------------------------------------------------------------

class YoloTFLiteDetect(BaseExperiment):
    """Run a TFLite-quantised YOLOv8 model on live camera frames."""

    NAME = "yolo_tflite_detect"

    def __init__(
        self,
        camera: picamera2.Picamera2,
        fps: int,
        size: Tuple[int, int],
        model_path: str | os.PathLike,
        labelmap_path: str,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        experiment_duration: int = 60,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        self.fps = fps
        self.size = size
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

        self.detector = YOLOTFLite(model_path, labelmap_path, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        self.frame_idx = 0

        self.output_dir = Path(output_dir or f"output/{self.NAME}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(self.NAME, camera, experiment_duration)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def configure_camera(self) -> None:
        period = int(1_000_000 / self.fps)
        config = self.camera.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},
            controls={"FrameDurationLimits": (period, period)},
        )
        self.camera.configure(config)
        self.camera.start()


    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_image(self, image: np.ndarray) -> None:
        """Run the TFLite detector, draw results, and persist frame."""
        results = self.detector.predict(image)
        print(f"Frame {self.frame_idx}: {len(results)} detections")
        # picamera2 can hand back a read-only buffer; copy so cv2 can draw
        frame = image.copy()
        for det in results:
            x, y, w, h = det["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"Class ID: {det['class_id']}")
            label = f"{self.detector.labelmap[det['class_id']]} ({int(det['conf'] * 100)}%)"
            print(f"Label: {label}")
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        fname = self.output_dir / f"{self.frame_idx:06d}.jpg"
        cv2.imwrite(str(fname), frame)
        self.frame_idx += 1
