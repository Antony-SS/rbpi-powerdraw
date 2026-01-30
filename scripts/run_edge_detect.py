#!/usr/bin/env python3
"""Run the EdgeDetect experiment from the command line.

Usage example
-------------
$ python scripts/run_edge_detect.py \
    --fps 20 \
    --size 640x480 \
    --complexity 0.4 \
    --experiment_duration 30
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import picamera2

# Ensure project root is on sys.path so "experiments" package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.impl.edge_detect import EdgeDetect


def _parse_size(s: str) -> tuple[int, int]:
    """Convert "WxH" string into an (width, height) tuple."""
    try:
        w, h = (int(x) for x in s.lower().split("x"))
    except ValueError as exc:  # noqa: D401
        raise argparse.ArgumentTypeError("--size must be like 640x480") from exc
    return w, h


def build_arg_parser() -> argparse.ArgumentParser:  # noqa: D401
    parser = argparse.ArgumentParser(description="Edge detection experiment runner")
    parser.add_argument("--fps", type=int, required=True, help="Target camera FPS")
    parser.add_argument("--size", type=_parse_size, required=True, help="Resolution WxH, e.g. 640x480")
    parser.add_argument(
        "--complexity",
        type=float,
        default=0.3,
        help="Edge detection complexity 0‒1 (default 0.3)",
    )
    parser.add_argument(
        "--experiment_duration",
        type=int,
        default=60,
        help="Duration in seconds (default 60)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save images (default auto-generated)",
    )
    return parser


def main() -> None:  # noqa: D401
    args = build_arg_parser().parse_args()
    width, height = args.size

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(
            f"output/{EdgeDetect.NAME}_fps{args.fps}_size{width}x{height}_c{args.complexity}"
        )

    exp = EdgeDetect(
        camera=picamera2.Picamera2(),
        fps=args.fps,
        size=(width, height),
        complexity=args.complexity,
        experiment_duration=args.experiment_duration,
        output_dir=out_dir,
    )

    exp.run_experiment()


if __name__ == "__main__":
    main()
