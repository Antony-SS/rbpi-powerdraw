from __future__ import annotations

"""TFLite-based MobileNet SSD v2 detection experiment.

MobileNet SSD v2 (COCO, uint8-quantised) is considerably lighter than
YOLOv8n on a Raspberry Pi 3 and typically achieves single-frame latency in
the 100–200 ms range at 300×300 input.

The TFLite variant of the SSD model exposes **four output tensors**:

0. detection_boxes    – [1, N, 4] normalised [ymin, xmin, ymax, xmax]
1. detection_classes  – [1, N]    class indices (float, 1-indexed for COCO)
2. detection_scores   – [1, N]    confidence scores
3. detection_count    – [1]       number of valid detections
"""

from pathlib import Path
from typing import Tuple, List
import os
import json
import time

import cv2  # type: ignore
import numpy as np  # type: ignore
import picamera2  # type: ignore

from experiments.core.experiment import BaseExperiment

# ---------------------------------------------------------------------------
# TFLite backend import
# ---------------------------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite  # type: ignore
except ImportError:
    try:
        import tensorflow.lite as tflite  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import tflite_runtime or tensorflow.lite"
        ) from exc


class MobileNetSSDTFLite:
    """Minimal MobileNet SSD v2 TFLite wrapper (uint8 quantised).

    Parameters
    ----------
    model_path : str or os.PathLike
        Path to the ``.tflite`` model file.
    labelmap_path : str
        Path to a JSON label-map file (same format as labelmap.json).
    conf_thres : float
        Minimum confidence to keep a detection.
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        labelmap_path: str,
        conf_thres: float = 0.5,
    ) -> None:
        self.conf_thres = conf_thres

        # Load interpreter
        self.interpreter = tflite.Interpreter(
            model_path=str(model_path), num_threads=4
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]  # e.g. [1, 300, 300, 3]
        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]

        print(f"[MobileNetSSD] input shape: {self.input_shape}, dtype: {self.input_dtype}")
        for i, od in enumerate(self.output_details):
            print(f"  output[{i}]: shape={od['shape']}, dtype={od['dtype']}, name={od['name']}")

        self.labelmap = self._load_labelmap(labelmap_path)
        print(f"[MobileNetSSD] labelmap entries: {len(self.labelmap)}")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_labelmap(path: str) -> dict[int, str]:
        """Load label map from JSON file.

        Parameters
        ----------
        path : str
            Path to labelmap JSON.

        Returns
        -------
        dict[int, str]
            Mapping from class index to label string.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Labelmap file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
            classes = data.get("class", {})
        return {int(k): v for k, v in classes.items()}

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> List[dict]:
        """Run inference on *image* and return filtered detections.

        Parameters
        ----------
        image : np.ndarray
            BGR or RGB image of any size.

        Returns
        -------
        list[dict]
            Each dict has keys ``box`` ([x, y, w, h] in original image
            coords), ``conf`` (float), and ``class_id`` (int).
        """
        img_h, img_w = image.shape[:2]

        # Pre-process
        resized = cv2.resize(
            image, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        if self.input_dtype == np.float32:
            input_data = (resized / 127.5 - 1.0).astype(np.float32)
        else:
            input_data = resized.astype(self.input_dtype)
        input_data = np.expand_dims(input_data, axis=0)

        # Inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # --- Read outputs --------------------------------------------------
        # Standard SSD TFLite export order:
        #   0: boxes  [1, N, 4]  (ymin, xmin, ymax, xmax) normalised 0-1
        #   1: classes [1, N]    (float, 1-indexed for COCO)
        #   2: scores  [1, N]
        #   3: count   [1]
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        class_ids = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        num_det = int(
            self.interpreter.get_tensor(self.output_details[3]["index"])[0]
        )

        results: list[dict] = []
        for i in range(num_det):
            score = float(scores[i])
            if score < self.conf_thres:
                continue
            # SSD boxes are [ymin, xmin, ymax, xmax] normalised to [0, 1]
            ymin, xmin, ymax, xmax = boxes[i]
            x = int(xmin * img_w)
            y = int(ymin * img_h)
            w = int((xmax - xmin) * img_w)
            h = int((ymax - ymin) * img_h)
            # COCO SSD classes are 1-indexed; our labelmap is 0-indexed
            cid = int(class_ids[i]) - 1
            results.append({"box": [x, y, w, h], "conf": score, "class_id": cid, "label": self.labelmap.get(cid, str(cid))})

        return results


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class MobilenetDetect(BaseExperiment):
    """Run a TFLite MobileNet SSD v2 model on live camera frames.

    Parameters
    ----------
    camera : picamera2.Picamera2
        Initialised camera instance.
    fps : int
        Target frames per second.
    size : tuple[int, int]
        Capture resolution (width, height).
    model_path : str or os.PathLike
        Path to the ``.tflite`` model file.
    labelmap_path : str
        Path to a JSON label-map file.
    conf_thres : float
        Minimum detection confidence.
    experiment_duration : int
        Run length in seconds.
    output_dir : str or os.PathLike or None
        Directory for annotated frame output.
    """

    NAME = "mobilenet_detect"

    def __init__(
        self,
        camera: picamera2.Picamera2,
        fps: int,
        size: Tuple[int, int],
        model_path: str | os.PathLike,
        labelmap_path: str,
        conf_thres: float = 0.5,
        experiment_duration: int = 60,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        self.fps = fps
        self.size = size
        self.conf_thres = float(conf_thres)

        self.detector = MobileNetSSDTFLite(
            model_path, labelmap_path, conf_thres=self.conf_thres
        )
        self.frame_idx = 0
        self.total_infer_time = 0.0
        self.total_imwrite_time = 0.0
        self.exp_start_time = None

        self.output_dir = Path(output_dir or f"output/{self.NAME}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(self.NAME, camera, experiment_duration)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def configure_camera(self) -> None:
        """Configure the camera for the requested resolution / FPS."""
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
        """Run the MobileNet SSD detector, draw results, and save frame."""
        if self.exp_start_time is None:
            self.exp_start_time = time.monotonic()

        t0 = time.monotonic()
        results = self.detector.predict(image)
        t_infer = time.monotonic() - t0
        self.total_infer_time += t_infer

        print(
            f"Frame {self.frame_idx}: {len(results)} det, "
            f"infer={t_infer*1000:.0f}ms",
            end="",
        )

        frame = image.copy()
        for det in results:
            x, y, w, h = det["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_str = self.detector.labelmap.get(det["class_id"], str(det["class_id"]))
            label = f"{label_str} ({int(det['conf'] * 100)}%)"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        t0 = time.monotonic()
        fname = self.output_dir / f"{self.frame_idx:06d}.jpg"
        cv2.imwrite(str(fname), frame)
        t_write = time.monotonic() - t0
        self.total_imwrite_time += t_write

        print(f", imwrite={t_write*1000:.0f}ms")
        self.frame_idx += 1

    def unconfigure_camera(self) -> None:
        """Print timing summary and release camera."""
        elapsed = time.monotonic() - (self.exp_start_time or time.monotonic())
        n = max(self.frame_idx, 1)
        avg_hz = n / elapsed if elapsed > 0 else 0
        print("\n===== Timing Summary =====")
        print(f"Frames processed:  {n}")
        print(f"Elapsed time:      {elapsed:.1f}s")
        print(f"Avg inference:     {self.total_infer_time/n*1000:.0f}ms")
        print(f"Avg imwrite:       {self.total_imwrite_time/n*1000:.0f}ms")
        print(f"Average Hz:        {avg_hz:.2f}")
        print("==========================")
        super().unconfigure_camera()
