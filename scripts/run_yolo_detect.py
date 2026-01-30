#!/usr/bin/env python3
"""CLI runner for the YOLO detection experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import picamera2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.impl.yolo_detect import YoloDetect  # noqa: E402


def _parse_size(arg: str):
    try:
        w, h = (int(x) for x in arg.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--size must be WxH e.g. 640x480") from exc
    return w, h


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLOv8n detection on Pi camera")
    p.add_argument("--fps", type=int, required=True)
    p.add_argument("--size", type=_parse_size, required=True, help="Resolution WxH")
    p.add_argument("--model_level", type=int, choices=[1, 2], default=1, help="1=yolov8n.tflite, 2=yolov8s.tflite")
    p.add_argument("--conf_thres", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--experiment_duration", type=int, default=60)
    p.add_argument("--output_dir", type=str)
    return p


def main() -> None:
    args = build_parser().parse_args()
    width, height = args.size

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(
            f"output/{YoloDetect.NAME}_fps{args.fps}_size{width}x{height}_c{args.complexity}"
        )
    )

    exp = YoloDetect(
        camera=picamera2.Picamera2(),
        fps=args.fps,
        size=(width, height),
        model_level=args.model_level,
        conf_thres=args.conf_thres,
        experiment_duration=args.experiment_duration,
        output_dir=out_dir,
    )
    exp.run_experiment()


if __name__ == "__main__":
    main()
