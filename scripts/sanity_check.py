#!/usr/bin/env python3
"""sanity_check.py

Quick sanity script for Raspberry Pi Camera V2/IMX219.

This script initialises the camera with a basic 640 × 480 RGB video
configuration locked to **10 frames per second**, then continuously captures
frames and writes them as JPEGs into a local *test_output* directory.
The loop runs until you press *Ctrl-C*.

Notes
-----
* Requires ``picamera2`` and ``opencv-python-headless`` in the active
  environment (already specified in *environment.yml*).
* Output filenames are epoch timestamps so consecutive runs never clash.
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np  # noqa: F401  # implicit dependency via Picamera2 arrays
from picamera2 import Picamera2


OUTPUT_DIR = Path("test_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FPS: int = 10
FRAME_PERIOD_US: int = int(1_000_000 / FPS)  # micro-seconds per frame


def main() -> None:
    """Grab frames at 10 Hz and save JPEGs to *test_output* until interrupted."""
    camera = Picamera2()

    config = camera.create_video_configuration(
        main={
            "size": (640, 480),
            "format": "RGB888",
        },
        controls={
            "FrameDurationLimits": (FRAME_PERIOD_US, FRAME_PERIOD_US),
        },
    )

    camera.configure(config)
    camera.start()

    # Allow clean shutdown on Ctrl-C.
    def _sig_handler(signum, frame):  # noqa: D401, ANN001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sig_handler)

    print("[sanity_check] Capturing at 10 fps — press Ctrl-C to stop …")
    try:
        while True:
            frame = camera.capture_array("main", wait=True)
            if frame is None:
                continue  # should not happen, but be safe

            frame = np.asarray(frame)
            timestamp = time.time()
            filename = OUTPUT_DIR / f"{timestamp:.6f}.jpg"
            cv2.imwrite(str(filename), frame)
            # Sleep to match target rate; adjust for time spent in capture/write.
            # This is *approximate*; a production app would sync to frame
            # timestamps instead.
            time.sleep(1.0 / FPS)
    except KeyboardInterrupt:
        print("\n[sanity_check] Stopping …")
    finally:
        camera.close()
        print("[sanity_check] Camera closed. Images saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
