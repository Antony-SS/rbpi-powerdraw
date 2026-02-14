#!/usr/bin/env python3
"""CLI runner for the TFLite MobileNet SSD v2 detection experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import picamera2  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.impl.mobilenet_detect import MobilenetDetect  # noqa: E402


def _parse_size(arg: str):
    try:
        w, h = (int(x) for x in arg.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--size must be WxH e.g. 640x480") from exc
    return w, h


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Run MobileNet SSD v2 detection on Pi camera using TFLite runtime",
    )
    p.add_argument("--fps", type=int, required=True, help="Frames per second")
    p.add_argument("--size", type=_parse_size, required=True, help="Resolution WxH")
    p.add_argument("--model_path", type=str, required=True,
                    help="Path to the .tflite model")
    p.add_argument("--labelmap_path", type=str, default="models/labelmap.json",
                    help="Path to the labelmap.json file")
    p.add_argument("--conf_thres", type=float, default=0.5,
                    help="Confidence threshold")
    p.add_argument("--experiment_duration", type=int, default=60,
                    help="Run length in seconds")
    p.add_argument("--output_dir", type=str,
                    help="Directory to write annotated JPEGs")
    return p


def main() -> None:
    """Entry point."""
    args = build_parser().parse_args()
    width, height = args.size

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(
            f"output/{MobilenetDetect.NAME}_fps{args.fps}"
            f"_size{width}x{height}"
            f"_model{Path(args.model_path).stem}"
        )
    )

    exp = MobilenetDetect(
        camera=picamera2.Picamera2(),
        fps=args.fps,
        size=(width, height),
        model_path=args.model_path,
        labelmap_path=args.labelmap_path,
        conf_thres=args.conf_thres,
        experiment_duration=args.experiment_duration,
        output_dir=out_dir,
    )
    exp.run_experiment()


if __name__ == "__main__":
    main()
