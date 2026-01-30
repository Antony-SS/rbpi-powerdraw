"""Edge detection experiment implementation.

Each captured frame is converted to grayscale, optionally blurred, and then
passed through Canny edge detection. The aggressiveness / sensitivity of the
edge detector can be tuned via the *complexity* parameter (0‒1 scale).

Higher complexity ⇒ wider Gaussian kernel and lower Canny thresholds ⇒ more
edges (and more CPU time).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import picamera2

from experiments.core.experiment import BaseExperiment


class EdgeDetect(BaseExperiment):
    """Detect edges in video frames using Canny.

    Parameters
    ----------
    camera : picamera2.Picamera2
        Initialised camera instance.
    fps : int
        Target frames per second.
    size : tuple[int, int]
        Output resolution (width, height).
    complexity : float, optional
        Value in the interval [0, 1]. 0 → very light; 1 → heavy. Controls the
        Gaussian blur kernel size *and* the Canny thresholds. Default is 0.3.
    experiment_duration : int, default 60
        Run time in seconds.
    output_dir : str | Path | None, optional
        Where to save processed images. Defaults to ``output/edge_detect``.
    """

    NAME = "edge_detect"

    def __init__(
        self,
        camera: picamera2.Picamera2,
        fps: int,
        size: tuple[int, int],
        complexity: float = 0.3,
        experiment_duration: int = 60,
        output_dir: str | os.PathLike | None = None,
    ) -> None:
        complexity = float(np.clip(complexity, 0.0, 1.0))
        self.fps = fps
        self.size = size
        self.complexity = complexity

        self.output_dir = Path(output_dir or f"output/{self.NAME}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(self.NAME, camera, experiment_duration)

    # ---------------------------------------------------------------------
    # Camera setup
    # ---------------------------------------------------------------------
    def configure_camera(self) -> None:
        """Configure the camera for the requested resolution/FPS."""
        period_us = int(1_000_000 / self.fps)
        cfg = self.camera.create_video_configuration(
            main={"size": self.size, "format": "RGB888"},
            controls={"FrameDurationLimits": (period_us, period_us)},
        )
        self.camera.configure(cfg)
        self.camera.start()

    # ---------------------------------------------------------------------
    # Frame processing
    # ---------------------------------------------------------------------
    def process_image(self, image: np.ndarray) -> None:  # noqa: D401
        """Apply Canny edge detection and save result.

        The complexity parameter controls:
        * Gaussian blur kernel size k = 1 + 2*round(1 + complexity*3)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive blur kernel
        k = 1 + 2 * int(round(1 + self.complexity * 10))  # odd 3..9
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        # Adaptive thresholds
        low = 10
        high = 150

        edges = cv2.Canny(gray, low, high)

        # Save side-by-side composite for easy inspection
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        stacked = np.hstack((image, edges_rgb))

        fname = self.output_dir / f"{time.time():.6f}.jpg"
        cv2.imwrite(str(fname), stacked)
