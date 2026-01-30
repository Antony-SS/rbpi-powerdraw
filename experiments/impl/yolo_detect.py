"""YOLO object-detection experiment.

Runs the lightest-weight Ultralytics YOLO model ("yolov8n") on each captured
frame, draws bounding boxes, and saves annotated results. The *complexity*
parameter controls the confidence threshold—higher complexity lowers the
threshold (detects more objects, slower post-NMS), lower complexity raises it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import picamera2
from ultralytics import YOLO  # type: ignore

from experiments.core.experiment import BaseExperiment


class YoloDetect(BaseExperiment):
    """Run YOLOv8n on live frames.

    Parameters
    ----------
    camera : picamera2.Picamera2
        Active camera instance.
    fps : int
        Desired frames per second.
    size : Tuple[int, int]
        Width, height of the RGB stream.
    model_level : int, optional
        1 → YOLOv8n (nano, ~3 MB), 2 → YOLOv8s (small, ~11 MB). Default 1.
    conf_thres : float, optional
        YOLO confidence threshold (default 0.25).
    experiment_duration : int, default 60
        Run length in seconds.
    output_dir : str | os.PathLike | None
        Where to write annotated JPEGs. Default auto-generated.
    """

    NAME = "yolo_detect"

    def __init__(
        self,
        camera: picamera2.Picamera2,
        fps: int,
        size: Tuple[int, int],
        model_level: int = 1,
        conf_thres: float = 0.25,
        experiment_duration: int = 60,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        self.fps = fps
        self.size = size
        self.conf_thres = float(conf_thres)

        self.output_dir = Path(output_dir or f"output/{self.NAME}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Select TFLite model based on level (nano vs small)
        model_name = "yolov8n.tflite" if model_level == 1 else "yolov8s.tflite"
        self.model = YOLO(model_name)  # Ultralytics auto-selects TFLite backend

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
    def process_image(self, image: np.ndarray) -> None:  # noqa: D401
        """Run YOLO and save annotated frame."""
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            iou=0.5,
            verbose=False,
            imgsz=max(self.size),
        )
        annotated = results[0].plot()
        fname = self.output_dir / f"{time.time():.6f}.jpg"
        cv2.imwrite(str(fname), annotated)
